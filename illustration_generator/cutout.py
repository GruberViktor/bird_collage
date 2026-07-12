"""Step 2 of the pipeline: turn each illustration into a mask and an outline.

    1. pregen.py   draw the bird on a flat cream ground
    2. cutout.py   THIS FILE - segment the bird out of that ground
    3. render      composite the collage

For every illustrations/<stem>.png this writes two files into
illustration_masks/:

    <stem>.png   a hard black-and-white mask - white bird, black ground.
                 Multiply it into the alpha channel at render time to drop
                 the background. (Deliberately NOT done here: this file
                 produces the mask, not the transparent PNG.)
    <stem>.svg   a simplified outline of the same silhouette, for packing
                 the birds into a collage without overlapping.

WHY A MODEL AND NOT A FLOOD FILL. The cream ground is flat and uniform
(std ~2-5 across the whole frame), so colour-keying it looks like a
one-liner - and it is a trap. In real woodblock practice the model paints
a bird's WHITE areas by leaving the paper showing, so a magpie's white
primaries and belly are literally the same cream as the background, and
they connect out to it through the gaps between the quill strokes. A
corner flood fill at any tolerance either eats the magpie's wing, belly
and tail or keeps background noise. Measured, not guessed: at tol=28 the
magpie came out 6.2% foreground with its whole right wing missing.

So the mask has to come from something that knows what a bird IS. A cheap
image model does. It costs a fraction of the Pro call that drew the bird,
and it runs once per illustration because the result is cached on disk.

The model's mask is a REGENERATED image, so it is semantically right but
not pixel-exact - its edge can sit a pixel or two off the ink line. That
is acceptable for both consumers here: erode the mask slightly when you
build the alpha so no halo of cream survives, and the packing outline is
approximate by design (holes filled, which is what a packer wants - one
conservative solid silhouette).
"""

import base64
import re
from pathlib import Path

import cv2
import numpy as np

from pregen import (
    OUT_DIR,
    _data_url,
    _post_with_retry,
    illustration_path,
)

# Cheap on purpose. This traces a silhouette; it does not need to know what
# a house martin's throat looks like - that battle was already fought in
# pregen.py. Keep it well away from the expensive IMAGE_MODEL.
MASK_MODEL = "google/gemini-3.1-flash-image"

BASE_DIR = Path(__file__).resolve().parent
SOURCE_IMAGE_DIR = BASE_DIR / "illustrations"
MASK_DIR = BASE_DIR / "illustration_masks"

MASK_PROMPT = (
    "Produce a SEGMENTATION MASK of the bird in this image.\n\n"
    "Output a pure black-and-white image, the same size and composition as "
    "the input and precisely registered with it:\n"
    "- Every pixel belonging to the BIRD is PURE WHITE (#FFFFFF).\n"
    "- Every other pixel is PURE BLACK (#000000).\n\n"
    "The bird counts in FULL: body, head, beak, eye, both wings out to every "
    "wingtip, the whole tail to its tip, legs, feet and every claw.\n\n"
    "CRITICAL: parts of the bird are painted in a pale cream or white that is "
    "almost exactly the colour of the paper background - white wing feathers, "
    "a white belly, a white rump. Those pale areas ARE the bird and MUST be "
    "WHITE in the mask. Do not follow colour; follow the SHAPE of the bird. "
    "The silhouette must be one solid filled shape, with no holes punched "
    "through it wherever the plumage happens to be pale.\n\n"
    "Fill the silhouette solidly. No grey, no anti-aliasing, no soft edges, no "
    "shadow, no outline stroke, no texture, no caption. Only a solid white "
    "bird on a solid black field."
)

# A mask that is nearly all black or nearly all white is the model having
# failed, not a strange bird. Worth one retry before giving up.
MIN_FG, MAX_FG = 0.005, 0.90


def mask_path(stem: str) -> Path:
    return MASK_DIR / f"{stem}.png"


def svg_path(stem: str) -> Path:
    return MASK_DIR / f"{stem}.svg"


def _request_mask(png: bytes) -> np.ndarray:
    """Ask the cheap image model for a silhouette. Returns it grayscale, not
    yet binarised and not yet registered to the source size."""
    resp = _post_with_retry(
        {
            "model": MASK_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": MASK_PROMPT},
                        {"type": "image_url", "image_url": {"url": _data_url(png)}},
                    ],
                }
            ],
            "modalities": ["image", "text"],
        }
    )
    for choice in resp.get("choices", []):
        for img in choice.get("message", {}).get("images") or []:
            url = (img.get("image_url") or {}).get("url", "")
            if url.startswith("data:") and ";base64," in url:
                raw = base64.b64decode(url.split(";base64,", 1)[1])
                arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
                if arr is not None:
                    return arr
    # No image - surface the finish/error reason so users know what to fix.
    finish = (resp.get("choices") or [{}])[0].get("finish_reason", "?")
    error = resp.get("error", {}).get("message", "")
    raise RuntimeError(f"no mask image (finish={finish} error={error})")


def _clean(raw: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Binarise the model's mask, register it to the source, and tidy it.

    The model regenerates rather than edits, so the mask can come back at a
    different size and with soft or dithered edges. Otsu binarises it however
    its brightness drifted; the morphology closes the pinholes that dithering
    leaves behind.
    """
    h, w = shape
    _, m = cv2.threshold(raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)  # close pinholes in the bird
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)  # drop specks in the ground

    # Keep only the bird: a stray blob in a corner would otherwise become a
    # second contour and confuse the packer.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.where(labels == biggest, 255, 0).astype(np.uint8)

    # Fill enclosed holes: flood the ground in from the border, and anything
    # black the flood cannot reach must have been enclosed by the bird.
    ff = np.zeros((h + 2, w + 2), np.uint8)
    flooded = m.copy()
    cv2.floodFill(flooded, ff, (0, 0), 255)
    return m | cv2.bitwise_not(flooded)


def build_mask(png: bytes, attempts: int = 2) -> np.ndarray:
    """Segment the bird. Returns a uint8 mask: 255 = bird, 0 = ground."""
    src = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if src is None:
        raise RuntimeError("could not decode the illustration")
    shape = src.shape[:2]

    last = 0.0
    for _ in range(attempts):
        m = _clean(_request_mask(png), shape)
        last = float(m.mean()) / 255.0
        if MIN_FG <= last <= MAX_FG:
            return m
    raise RuntimeError(
        f"mask implausible after {attempts} attempts (foreground {last:.1%}, "
        f"expected {MIN_FG:.1%}-{MAX_FG:.0%}) - the model probably returned a "
        f"blank or inverted image"
    )


def rasterize_svg(svg: str, shape: tuple[int, int]) -> np.ndarray:
    """Render an outline SVG back to a bitmap, the way a real renderer would.

    Used to PROVE the outline and the mask agree. The half-pixel matters: an
    SVG coordinate is a pixel CORNER, so a renderer covers pixel (i,j) iff the
    point (i+0.5, j+0.5) falls inside the path. OpenCV's fillPoly instead
    samples integer pixel CENTRES, so we shift by -0.5 to emulate the renderer
    rather than to flatter ourselves.
    """
    m = np.zeros(shape, np.uint8)
    polys = []
    for d in re.findall(r'd="M ([^"]+) Z"', svg):
        pts = [[float(x), float(y)] for x, y in (p.split() for p in d.split(" L "))]
        polys.append(np.round((np.array(pts) - 0.5) * 16).astype(np.int32))
    if polys:
        cv2.fillPoly(m, polys, 255, shift=4)  # 4 fractional bits = 1/16 px
    return m


def mask_to_svg(mask: np.ndarray, smooth: float = 0.0, pad: int = 0) -> str:
    """Trace the mask into an SVG outline that overlays the mask EXACTLY.

    The outline lives in the same pixel space as the mask and the
    illustration: width, height and viewBox all equal the image size and there
    is no transform, so drawing the SVG at (0,0) at its natural size lands on
    the same pixels the mask marks. Verified, not assumed - see
    rasterize_svg() and the exactness check in cutout_file().

    Only the OUTER contour is emitted, so interior holes are filled: a packer
    wants one solid silhouette, not a shape another bird could nest inside.

    smooth: Douglas-Peucker epsilon, as a fraction of the contour perimeter.
        DEFAULT 0 = exact. This is not a knob to turn casually. At the 0.0015
        that seemed reasonable, the outline sliced 1.29% of the bird OFF -
        wingtips and tail first, since they are thin - and it cut INWARD,
        which is the dangerous direction: a packer trusting that outline would
        let birds overlap. Exactness is nearly free anyway, because
        CHAIN_APPROX_SIMPLE already collapses straight runs losslessly.
        If you do set it, any simplification that would cut into the bird is
        rejected here and the exact contour is used instead - the outline is
        guaranteed to be a SUPERSET of the mask, never a subset.

    pad: grow the silhouette by N pixels. A deliberate margin for packing, so
        birds do not touch. Unlike `smooth`, this only ever grows the shape.
    """
    h, w = mask.shape[:2]
    if pad > 0:
        k = 2 * pad + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.0005 * h * w  # ignore specks that survived the cleanup
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        raise RuntimeError("no contour found in the mask")

    def build(polys: list[np.ndarray]) -> str:
        body = "\n".join(
            '    <path d="M '
            + " L ".join(f"{x} {y}" for x, y in p.reshape(-1, 2))
            + ' Z"/>'
            for p in polys
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">\n'
            f"  <!-- Bird silhouette, holes filled. Same pixel space as the mask:\n"
            f"       draw at (0,0) unscaled and it overlays the mask exactly. -->\n"
            f'  <g fill="#000000" fill-rule="evenodd">\n{body}\n  </g>\n</svg>\n'
        )

    exact = build(contours)
    if not smooth:
        return exact

    simplified = [
        cv2.approxPolyDP(c, smooth * cv2.arcLength(c, True), True) for c in contours
    ]
    simplified = [p for p in simplified if len(p) >= 3]
    if not simplified:
        return exact
    # Only accept the simplification if it still covers every bird pixel.
    lost = (mask == 255) & (rasterize_svg(build(simplified), (h, w)) == 0)
    return exact if lost.any() else build(simplified)


def cutout_file(
    src: Path, overwrite: bool = False, verbose: bool = True
) -> tuple[Path, Path]:
    """Mask and vectorise one illustration. Returns (mask_path, svg_path)."""
    MASK_DIR.mkdir(exist_ok=True)
    mp, sp = mask_path(src.stem), svg_path(src.stem)

    if not overwrite and mp.exists() and sp.exists():
        if verbose:
            print(f"  [{src.stem}] already cut out, skipping (no API call)")
        return mp, sp

    mask = build_mask(src.read_bytes())
    svg = mask_to_svg(mask)

    # Prove the two agree before writing them, rather than trusting that they
    # do. They are consumed by different code paths - the mask becomes alpha,
    # the outline drives packing - and a silent drift between them would place
    # a bird by an outline that does not match the pixels it then draws.
    agreement = _agreement(svg, mask)
    if agreement < 1.0:
        raise RuntimeError(
            f"outline does not overlay the mask ({agreement:.2%} of the bird "
            f"covered) - packing and alpha would disagree"
        )

    cv2.imwrite(str(mp), mask)
    sp.write_text(svg)
    if verbose:
        points = svg.count(" L ") + svg.count("M ")
        print(
            f"  [{src.stem}] bird = {mask.mean() / 255:.1%} of frame, "
            f"outline = {points} points, overlays mask exactly"
        )
    return mp, sp


def _agreement(svg: str, mask: np.ndarray) -> float:
    """Fraction of the mask's bird pixels the outline actually covers."""
    bird = mask == 255
    covered = bird & (rasterize_svg(svg, mask.shape[:2]) == 255)
    return float(covered.sum()) / float(bird.sum() or 1)


def cutout(sci_name: str, pose: str, **kw) -> tuple[Path, Path]:
    """Mask and vectorise one (species, pose), named as pregen names them."""
    return cutout_file(illustration_path(sci_name, pose), **kw)


def cutout_all(overwrite: bool = False) -> None:
    """Mask and vectorise every illustration that does not have one yet."""
    srcs = sorted(OUT_DIR.glob("*.png"))
    if not srcs:
        print(f"no illustrations found in {OUT_DIR}")
        return

    print(f"cutting out {len(srcs)} illustration(s) -> {MASK_DIR}/")
    failed = 0
    for src in srcs:
        try:
            cutout_file(src, overwrite=overwrite)
        except (RuntimeError, OSError) as e:
            failed += 1
            print(f"  [{src.stem}] FAILED: {e}")
    print(f"\ndone: {len(srcs) - failed} ok, {failed} failed")


if __name__ == "__main__":
    cutout_all()
