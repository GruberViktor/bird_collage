"""Step 3 of the pipeline: pack the day's birds into one collage.

    1. pregen.py   draw the bird on a flat cream ground
    2. cutout.py   segment it -> hard mask + exact SVG outline
    3. collage.py  THIS FILE - size, pack and composite them

Takes a BirdNET-Go daily-species list (see daily_species_example.py) and
returns a single portrait PNG.

    from collage import build_collage
    build_collage(response, Path("collage.png"))

WHAT IT DOES

- Sorts the species by count, loudest first.
- Picks each bird's pose at random, perched or flying.
- Generates the illustration and its cutout ON DEMAND, and only for the
  pose it actually picked. Nothing is pre-rendered, so a species you never
  hear never costs anything. Both stages cache to disk, so the second run
  of a day is free.
- Scales each bird by its count, then packs them out from the centre along
  a spiral, then scales the whole thing to fill the canvas.

COST. Every new (species, pose) costs one Gemini 3 Pro image (~$0.024) plus
a cheap mask call. Everything downstream of that is local pixels. So the
first collage of a fresh day costs a few cents and the rest are free -
which is only true because we pass overwrite=False. get_bird_illustration
defaults to overwrite=True, and left at that default every rebuild of the
same day would silently re-bill every bird.

SIZING. Counts are brutally skewed (665 house sparrows, 1 barn swallow), so
raw count cannot drive size. Counts map onto a bounded range: the smallest
bird is 1x and the largest is at most MAX_SIZE_RATIO x, no matter how far
apart their counts are.

PACKING. Collision uses the SVG outline, not the bounding box - the whole
reason cutout.py emits an exact outline. Birds are mostly diagonal wings
and long tails, so bounding boxes would leave huge gaps and the birds would
never actually nestle. Each placed bird is dilated by SPACING before it
goes into the occupancy grid, so a candidate only has to test its own raw
footprint: no overlap against the dilated set == at least SPACING of clear
air to the nearest bird.
"""

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cutout import cutout, mask_path, svg_path
from pregen import POSES, get_bird_illustration, illustration_path

# Portrait. NOTE: "1600x1200 portrait" is ambiguous - read as WxH it is
# landscape - so this is 1200 wide by 1600 tall. Swap if you meant the other.
CANVAS_W, CANVAS_H = 1200, 1600

MARGIN = 50  # the collage must reach within this of the canvas edge
SPACING = 40  # clear air between the closest points of two birds
MIN_BIRD_HEIGHT = 150  # the smallest bird's height, before the global fit scale
MAX_SIZE_RATIO = 4  # the largest bird is at most this many times the smallest

# Collision runs on a downsampled grid: the spiral tests thousands of
# candidate positions per bird and full resolution is wasted on that. But not
# too downsampled - the grid cell IS the quantisation error on SPACING. At
# 0.25 (a 4px cell) the measured gap came out 6.6px against a 10px target;
# at 0.5 (a 2px cell) it lands within a pixel or so, which is what "about
# 10px" asks for, and packing is still fast.
COLLISION_SCALE = 0.5

# The spiral. STEP is how far apart candidate positions are along the arc;
# GROWTH is how much the radius opens per full turn. Both in canvas pixels.
# GROWTH below a typical bird height is what lets small birds tuck into the
# gaps of the cluster instead of orbiting outside it.
SPIRAL_STEP = 8.0
SPIRAL_GROWTH = 30.0

# Room for the collage to overflow the canvas mid-search, before it gets
# scaled to fit. The spiral needs somewhere to put birds while we find out
# the scale is too big.
WORK_W, WORK_H = CANVAS_W * 2, CANVAS_H * 2

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Bird:
    sci: str
    com: str
    count: int
    pose: str
    rgb: np.ndarray  # cropped tight to the silhouette, BGR
    alpha: np.ndarray  # cropped identically, 0 or 255
    poly: np.ndarray  # SVG outline, in the same cropped coordinates
    factor: float  # size multiple, 1.0 .. MAX_SIZE_RATIO


@dataclass
class Placement:
    bird: Bird
    x: int  # top-left in work coordinates
    y: int
    rgb: np.ndarray  # scaled to its final size
    alpha: np.ndarray


# ---- sizing ----


def size_factors(
    counts: list[int], curve: str = "linear", ratio: float = MAX_SIZE_RATIO
) -> list[float]:
    """Map counts onto 1.0 .. ratio.

    A 665-count sparrow must not be 665x a 1-count swallow, so the ratio is
    capped. `curve` shapes how the middle of the range fills in: "linear" is
    literal (with counts this skewed it leaves the long tail all at minimum
    size, which is honest but flat), while "sqrt" and "log" give the rare
    birds more presence.
    """
    lo, hi = min(counts), max(counts)
    if lo == hi:
        return [1.0] * len(counts)

    if curve == "linear":
        f = lambda c: c  # noqa: E731
    elif curve == "sqrt":
        f = math.sqrt
    elif curve == "log":
        f = lambda c: math.log(c + 1)  # noqa: E731
    else:
        raise ValueError(f"unknown curve {curve!r}")

    flo, fhi = f(lo), f(hi)
    return [1.0 + (ratio - 1) * (f(c) - flo) / (fhi - flo) for c in counts]


def fit_ratio(birds: list[Bird], curve: str, verbose: bool = True) -> float:
    """Pick the biggest size ratio (up to MAX_SIZE_RATIO) that still leaves the
    smallest bird at least MIN_BIRD_HEIGHT tall.

    The two rules collide. "Largest bird up to 10x the smallest" and "smallest
    bird at least 100px" cannot both hold on a skewed day: at 10x the biggest
    bird alone is wider than the canvas, so the fit-to-canvas scale drags
    everything down and the smallest lands around 63px. MAX_SIZE_RATIO is a
    CAP, though, not a target - so we take the largest ratio under that cap
    which still respects the 100px floor, and both rules hold.

    Monotone (a smaller ratio can only make the smallest bird bigger), so a
    binary search is safe.
    """
    counts = [b.count for b in birds]

    def smallest_height(ratio: float) -> int:
        for b, f in zip(birds, size_factors(counts, curve, ratio)):
            b.factor = f
        scale, _ = fit_scale(birds, verbose=False)
        return min(round(MIN_BIRD_HEIGHT * b.factor * scale) for b in birds)

    if smallest_height(MAX_SIZE_RATIO) >= MIN_BIRD_HEIGHT:
        return MAX_SIZE_RATIO  # no conflict today

    lo, hi = 1.0, float(MAX_SIZE_RATIO)
    for _ in range(7):
        mid = (lo + hi) / 2
        if smallest_height(mid) >= MIN_BIRD_HEIGHT:
            lo = mid  # still room to spread the sizes out
        else:
            hi = mid
    if verbose:
        print(
            f"\nsize ratio: {lo:.1f}x (capped at {MAX_SIZE_RATIO}x, but that "
            f"would push the smallest bird under {MIN_BIRD_HEIGHT}px)"
        )
    return lo


# ---- loading ----


def _parse_svg_polys(svg: str) -> list[np.ndarray]:
    out = []
    for d in re.findall(r'd="M ([^"]+) Z"', svg):
        pts = [[float(x), float(y)] for x, y in (p.split() for p in d.split(" L "))]
        out.append(np.array(pts, dtype=np.float32))
    return out


def prepare_bird(entry: dict, pose: str, verbose: bool = True) -> Bird | None:
    """Generate (if needed) and load one bird, cropped tight to its silhouette.

    Returns None if the species could not be drawn or segmented - one bad
    species should cost us that bird, not the whole collage.
    """
    sci, com = entry["scientific_name"], entry["common_name"]
    try:
        # overwrite=False is what makes a rebuild free. See the module docstring.
        get_bird_illustration(sci, com, pose=pose, overwrite=False, verbose=verbose)
        cutout(sci, pose, verbose=verbose)
    except (RuntimeError, OSError) as e:
        print(f"  [{sci}, {pose}] SKIPPED: {e}")
        return None

    src = illustration_path(sci, pose)
    img = cv2.imread(str(src))
    mask = cv2.imread(str(mask_path(src.stem)), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"  [{sci}, {pose}] SKIPPED: illustration or mask missing")
        return None

    # Erode a touch: the model's mask edge can sit a pixel off the ink line,
    # and a surviving rim of cream reads as a halo once it is on white.
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))

    x, y, w, h = cv2.boundingRect(mask)
    if w < 2 or h < 2:
        print(f"  [{sci}, {pose}] SKIPPED: empty mask")
        return None

    polys = _parse_svg_polys(svg_path(src.stem).read_text())
    poly = max(polys, key=len) if polys else None
    if poly is None:
        print(f"  [{sci}, {pose}] SKIPPED: no outline in svg")
        return None

    return Bird(
        sci=sci,
        com=com,
        count=entry["count"],
        pose=pose,
        rgb=img[y : y + h, x : x + w],
        alpha=mask[y : y + h, x : x + w],
        poly=poly - np.array([x, y], dtype=np.float32),  # into cropped coords
        factor=1.0,  # filled in by build_collage
    )


# ---- packing ----


def _footprint(bird: Bird, height: int) -> tuple[np.ndarray, int, int]:
    """The bird's collision shape at a given rendered height, from its SVG
    outline. Returns (small binary mask, full-res width, full-res height)."""
    scale = height / bird.alpha.shape[0]
    w = max(1, round(bird.alpha.shape[1] * scale))
    h = max(1, height)

    cw = max(1, round(w * COLLISION_SCALE))
    ch = max(1, round(h * COLLISION_SCALE))
    fp = np.zeros((ch, cw), np.uint8)
    pts = np.round(bird.poly * scale * COLLISION_SCALE).astype(np.int32)
    cv2.fillPoly(fp, [pts], 255)
    return fp, w, h


def _spiral(max_radius: float, aspect: float = CANVAS_H / CANVAS_W):
    """Candidate offsets from the centre, spiralling outward.

    The spiral is stretched by the canvas aspect, which matters more than it
    sounds. A round spiral grows a round cluster, and a round cluster in a
    portrait canvas fills the width, hits the margin, and stops - leaving the
    top and bottom thirds empty and forcing a smaller scale on everything.
    Opening the spiral faster vertically grows a portrait-shaped cluster that
    reaches all four margins, so every bird ends up bigger.
    """
    yield 0.0, 0.0
    theta = 0.0
    while True:
        theta += SPIRAL_STEP / max(SPIRAL_GROWTH * theta / (2 * math.pi), 1.0)
        r = SPIRAL_GROWTH * theta / (2 * math.pi)
        if r > max_radius:
            return
        yield r * math.cos(theta), r * math.sin(theta) * aspect


def pack(birds: list[Bird], scale: float) -> list[Placement] | None:
    """Place every bird, biggest first, spiralling out from the centre.

    Returns None if any bird could not be placed at all. Positions are in
    work coordinates, which are deliberately bigger than the canvas: at this
    point we are still finding out whether `scale` is too large.
    """
    occupied = np.zeros(
        (round(WORK_H * COLLISION_SCALE), round(WORK_W * COLLISION_SCALE)), np.uint8
    )
    gap = max(1, round(SPACING * COLLISION_SCALE))
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))

    cx, cy = WORK_W / 2, WORK_H / 2
    placements: list[Placement] = []

    for bird in birds:
        height = max(1, round(MIN_BIRD_HEIGHT * bird.factor * scale))
        fp, w, h = _footprint(bird, height)
        fh, fw = fp.shape
        placed = False

        for dx, dy in _spiral(max_radius=min(WORK_W, WORK_H) / 2):
            # top-left, full-res work coords, then the collision-grid version
            x = round(cx + dx - w / 2)
            y = round(cy + dy - h / 2)
            gx = round(x * COLLISION_SCALE)
            gy = round(y * COLLISION_SCALE)
            if (
                gx < 0
                or gy < 0
                or gx + fw > occupied.shape[1]
                or gy + fh > occupied.shape[0]
            ):
                continue
            if np.any(cv2.bitwise_and(occupied[gy : gy + fh, gx : gx + fw], fp)):
                continue

            # Commit. Dilating on the way in means every later candidate only
            # has to test its own raw shape to be SPACING clear of this one.
            #
            # The dilated ring extends BEYOND the footprint's bounding box, so
            # it has to be written into a region padded by `gap`. Writing it
            # back into the unpadded box instead silently clips the ring off
            # on every side the silhouette touches - which is most of them, a
            # bird's box being defined by its wingtips - and the gap collapses
            # to a third of what was asked for.
            pad = cv2.copyMakeBorder(
                fp, gap, gap, gap, gap, cv2.BORDER_CONSTANT, value=0
            )
            dil = cv2.dilate(pad, grow)
            ys0, xs0 = gy - gap, gx - gap
            cy0, cy1 = max(0, ys0), min(occupied.shape[0], ys0 + dil.shape[0])
            cx0, cx1 = max(0, xs0), min(occupied.shape[1], xs0 + dil.shape[1])
            occupied[cy0:cy1, cx0:cx1] |= dil[
                cy0 - ys0 : cy1 - ys0, cx0 - xs0 : cx1 - xs0
            ]
            rgb = cv2.resize(bird.rgb, (w, h), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(bird.alpha, (w, h), interpolation=cv2.INTER_AREA)
            placements.append(Placement(bird, x, y, rgb, alpha))
            placed = True
            break

        if not placed:
            return None

    return placements


def _bbox(placements: list[Placement]) -> tuple[int, int, int, int]:
    """Tight box around all placed birds, in work coords."""
    xs0 = min(p.x for p in placements)
    ys0 = min(p.y for p in placements)
    xs1 = max(p.x + p.rgb.shape[1] for p in placements)
    ys1 = max(p.y + p.rgb.shape[0] for p in placements)
    return xs0, ys0, xs1, ys1


def _fits(placements: list[Placement]) -> bool:
    x0, y0, x1, y1 = _bbox(placements)
    return (x1 - x0) <= CANVAS_W - 2 * MARGIN and (y1 - y0) <= CANVAS_H - 2 * MARGIN


def fit_scale(birds: list[Bird], verbose: bool = True) -> tuple[float, list[Placement]]:
    """Find the largest uniform scale whose packing still fits the canvas.

    This is both of the caller's rules at once: shrink if the collage spills
    out of the canvas, grow if it fails to reach within MARGIN of the edges.
    The largest scale that fits does both, and a binary search finds it -
    packing is deterministic in `scale`, so the search is stable.
    """
    lo, hi = 0.02, 4.0

    best = None
    for _ in range(14):
        mid = (lo + hi) / 2
        placements = pack(birds, mid)
        if placements and _fits(placements):
            best = (mid, placements)
            lo = mid  # room to grow
        else:
            hi = mid  # spilled out, shrink

    if best is None:
        # Even the smallest scale spills. Take it anyway and say so, rather
        # than shrinking forever or silently cropping birds off the edge.
        placements = pack(birds, lo)
        if not placements:
            raise RuntimeError("could not place the birds at any scale")
        print(f"  WARNING: the collage does not fit even at scale {lo:.3f}")
        return lo, placements

    scale, placements = best
    if verbose:
        x0, y0, x1, y1 = _bbox(placements)
        smallest = min(round(MIN_BIRD_HEIGHT * b.factor * scale) for b in birds)
        largest = max(round(MIN_BIRD_HEIGHT * b.factor * scale) for b in birds)
        avail = (CANVAS_W - 2 * MARGIN) * (CANVAS_H - 2 * MARGIN)
        print(
            f"\nfit: scale {scale:.3f}, collage {x1 - x0}x{y1 - y0}px in "
            f"{CANVAS_W}x{CANVAS_H} (margin {MARGIN}), "
            f"{(x1 - x0) * (y1 - y0) / avail:.0%} of the usable canvas"
        )
        print(f"     bird heights {smallest}-{largest}px")
    return scale, placements


# ---- rendering ----


def render(placements: list[Placement], outline: bool = False) -> np.ndarray:
    """Composite the placed birds onto white, centred on the canvas."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 255, np.uint8)

    x0, y0, x1, y1 = _bbox(placements)
    # Centre the collage's bounding box on the canvas.
    ox = round((CANVAS_W - (x1 - x0)) / 2) - x0
    oy = round((CANVAS_H - (y1 - y0)) / 2) - y0

    for p in placements:
        h, w = p.alpha.shape
        dx, dy = p.x + ox, p.y + oy

        # Clip, in case a bird still hangs off the edge.
        sx0, sy0 = max(0, -dx), max(0, -dy)
        dx0, dy0 = max(0, dx), max(0, dy)
        dx1, dy1 = min(CANVAS_W, dx + w), min(CANVAS_H, dy + h)
        if dx1 <= dx0 or dy1 <= dy0:
            continue
        sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)

        rgb = p.rgb[sy0:sy1, sx0:sx1].astype(np.float32)
        a = (p.alpha[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0)[..., None]
        back = canvas[dy0:dy1, dx0:dx1].astype(np.float32)
        canvas[dy0:dy1, dx0:dx1] = (rgb * a + back * (1 - a)).astype(np.uint8)

        if outline:  # the packing geometry, for eyeballing the gaps
            s = h / p.bird.alpha.shape[0]
            pts = np.round(p.bird.poly * s + [dx, dy]).astype(np.int32)
            cv2.polylines(canvas, [pts], True, (0, 0, 255), 1)

    return canvas


# ---- entry point ----


def build_collage(
    species: list[dict],
    out: Path = BASE_DIR / "collage.png",
    seed: int | None = None,
    curve: str = "linear",
    outline: bool = False,
    verbose: bool = True,
) -> Path:
    """Build the day's collage from a BirdNET-Go daily-species response."""
    rng = random.Random(seed)
    ranked = sorted(species, key=lambda e: e["count"], reverse=True)

    print(f"preparing {len(ranked)} species (generating only what is missing)")
    birds: list[Bird] = []
    for entry in ranked:
        pose = rng.choice(list(POSES))
        bird = prepare_bird(entry, pose, verbose=verbose)
        if bird:
            birds.append(bird)
    if not birds:
        raise RuntimeError("no birds could be prepared")

    ratio = fit_ratio(birds, curve, verbose=verbose)
    for bird, f in zip(birds, size_factors([b.count for b in birds], curve, ratio)):
        bird.factor = f

    scale, placements = fit_scale(birds, verbose=verbose)

    if verbose:
        print("\nsizes (by count):")
        for b in birds:
            h = round(MIN_BIRD_HEIGHT * b.factor * scale)
            print(f"  {b.count:>4}x {b.sci:<24} {b.pose:<8} {h:>4}px")
    canvas = render(placements, outline=outline)
    cv2.imwrite(str(out), canvas)
    print(f"\nwrote {out} ({CANVAS_W}x{CANVAS_H}, {len(placements)} birds)")
    return out


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(BASE_DIR.parent))
    from daily_species_example import response

    build_collage(response, seed=1)
