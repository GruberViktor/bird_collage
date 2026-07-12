"""Generate a kachō-e style bird illustration for a single species.

Call get_bird_illustration(sci_name, com_name). It returns the PNG bytes
and saves them to illustrations/<sci_name>.png. Set OPENROUTER_API_KEY.

WHAT COSTS MONEY. One generated image from Gemini 3 Pro Image costs
~$0.024, which is 40x everything else in this file put together. So the
design spends freely on cheap models to make the FIRST Pro call land, and
guards the second one jealously:

    per bird, typical    1 Pro image + 1 critique      (~$0.025)
    per bird, worst case 2 Pro images + 1 critique     (~$0.049)
    once per species     photo filter                  (~$0.0005, then cached)
    once per species     description + photos          (free, then cached)

A retry fires ONLY when the critique finds an error bad enough to make
the bird the wrong species, or to break the downstream cutout step. It
never fires for a stylistic quibble, and it never fires twice.

Note the counter-intuitive bit: the cheap photo filter is NOT there to
save input tokens (it saves ~$0.0006, i.e. nothing). It is there because
a junk reference can cause a bad image, and a bad image costs a $0.024
retry. Cheap calls that de-risk the expensive call are always worth it;
cheap calls that merely trim its input are not.

WHY IT USED TO GET SPECIES WRONG. Gemini's prior for a species collapses
toward its famous relative: a Common House Martin came out with a Barn
Swallow's dark throat-collar and no white rump at all, even though the
attached reference photos showed both correctly. So it was not failing to
SEE the reference, it was overriding it. Two things fixed that:

1. THE MODEL. gemini-2.5-flash-image ignored the references and the
   prompt's accuracy rules. gemini-3-pro-image follows both. This was the
   whole fix, and it is why we pay for Pro.

2. GROUNDING TEXT. The species' Wikipedia "Description" section is
   attached verbatim. It reads like a field guide ("steel-blue above with
   a white rump, and white underparts") and usually names the lookalikes
   the species is confused with - so the anti-lookalike nudge comes for
   free, per species, with no hand-maintained table.

   (A richer variant was tried and REJECTED: having a model derive a
   part-by-part field-marks checklist from the photos. It hallucinated a
   white forehead onto the house martin, and mandating an invented field
   mark is worse than saying nothing. Wikipedia prose is not exhaustive,
   but it does not make things up.)

Reference photos come from the species' Wikipedia article, several of
them, so a diagnostic feature is visible from more than one angle. That
media list is a grab-bag - for Delichon urbicum it also holds museum egg
specimens, chicks in a nest and a microscope slide of a house-martin flea
- so a cheap model vets them first (see select_bird_photos). By the time
Pro sees a photo, it is an adult bird.

Other learnings kept from the AvianVisitors pregen pipeline:

- References are downscaled to 384px on the long side. A full-size photo
  dominates as a style signal even though the prompt says it is anatomy
  only; 384px still resolves plumage zones, which are large features.
- The bird sits on a flat warm CREAM ground, not a transparent one: the
  model cannot cut transparency cleanly, but a flat known ground is easy
  to remove afterwards.
- Image bytes are MIME-sniffed from magic bytes; a PNG labelled as JPEG
  gets the reference silently rejected.
- Both modalities are requested so the model can surface refusal text
  without rejecting the request shape, and 429/5xx are retried.

Note on sources: eBird's public API serves observations and taxonomy
only - no photos, no appearance text. Its photos live in Cornell's
Macaulay Library, which is bot-protected (403) and all-rights-reserved.
Wikipedia/Wikimedia is CC-licensed and serves both photos and prose.
"""

import base64
import json
import os
import re
import time
import urllib.parse
from pathlib import Path

import cv2
import httpx
import jinja2
import numpy as np

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# The expensive one. Worth it: gemini-2.5-flash-image drew a Barn Swallow
# no matter what it was shown or told. Do not downgrade without re-checking
# a house martin's throat and rump.
IMAGE_MODEL = "google/gemini-3-pro-image"

# The cheap one. Only ever used to check the expensive one's work, so a
# critique costs a rounding error next to the image it might reject.
CRITIC_MODEL = "google/gemini-2.5-flash"

# Wikimedia's robot policy rejects generic User-Agents with a 403. It wants
# a descriptive one with a contact address.
WIKI_UA = "BirdCollage/1.0 (viktor@luvifermente.eu)"

REF_MAX_SIDE = 384  # long-side px of each attached reference photo
REF_COUNT = 4  # article photos attached; not all show a bird (see fetch_photos)

# The two poses. The key is what goes in the filename and drives the
# conditional blocks in prompt.txt; the value is the phrase handed to the
# model. Anything outside this set is a caller error, not a free-text pose:
# a typo'd pose would otherwise silently produce a file nothing else can find.
POSES = {
    "perched": "perched",
    "flying": "in flight, with both wings fully spread",
}

BASE_DIR = Path(__file__).resolve().parent
REFS_DIR = BASE_DIR / "wikipedia"
OUT_DIR = BASE_DIR / "illustrations"

prompt_template = jinja2.Template((BASE_DIR / "prompt.txt").read_text())


# ---- OpenRouter ----


def _post_with_retry(payload: dict) -> dict:
    """POST to OpenRouter with bounded retry on 429 + transient 5xx. The API
    key goes in the header, NOT the URL - keeps it out of request logs,
    proxy logs and shell history."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Title": "Bird Collage Generator",
    }
    backoff = 4.0
    for attempt in range(4):
        try:
            r = httpx.post(OPENROUTER_URL, json=payload, headers=headers, timeout=300)
        except httpx.TransportError:
            if attempt == 3:
                raise
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
                r.raise_for_status()
            try:
                backoff = float(r.headers.get("Retry-After", ""))
            except ValueError:
                pass  # missing or HTTP-date Retry-After, keep the backoff
        time.sleep(backoff)
        backoff *= 2
    raise RuntimeError("unreachable")


def _data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def _downscaled(path: Path) -> str:
    """Encode a reference as a data: URL, downscaled to REF_MAX_SIDE on the
    long side so no single photo dominates the output as a style signal."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # Undecodable by OpenCV - ship the raw bytes with their real MIME.
        mime = "image/png" if path.suffix == ".png" else "image/jpeg"
        return _data_url(path.read_bytes(), mime)
    h, w = img.shape[:2]
    if max(h, w) > REF_MAX_SIDE:
        scale = REF_MAX_SIDE / max(h, w)
        img = cv2.resize(
            img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"could not re-encode reference {path.name}")
    return _data_url(buf.tobytes())


# ---- Wikipedia ----


def _wiki_titles(sci_name: str, com_name: str) -> list[str]:
    return [sci_name, com_name, com_name.split()[0]]


def fetch_description(sci_name: str, com_name: str) -> str | None:
    """Return the species' Wikipedia 'Description' section as plain text.

    This is the plumage grounding handed to the model. It reads like a
    field guide and typically also names the lookalike species, which is
    what keeps the model from drifting toward them.
    """
    with httpx.Client(
        headers={"User-Agent": WIKI_UA}, follow_redirects=True, timeout=30
    ) as client:
        for title in _wiki_titles(sci_name, com_name):
            try:
                r = client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "prop": "extracts",
                        "explaintext": 1,
                        "redirects": 1,
                        "titles": title,
                        "format": "json",
                    },
                )
                pages = r.raise_for_status().json()["query"]["pages"]
            except (httpx.HTTPError, KeyError):
                continue
            text = next(iter(pages.values())).get("extract") or ""
            if not text:
                continue
            # Usually "Description"; some articles use "Description and taxonomy".
            m = re.search(
                r"^==+ ?Description[^=\n]* ?==+\s*(.+?)(?=^==)", text, re.S | re.M
            )
            section = m.group(1) if m else text[:1500]
            section = re.sub(r"\s+", " ", section).strip()
            if section:
                return section[:1500]
    return None


def fetch_photos(sci_name: str, com_name: str) -> list[tuple[bytes, str]]:
    """Download reference photos from the species' Wikipedia article.

    JPEG-only, which drops the SVG-derived range maps and IUCN status
    badges (they arrive as .png). What survives is still a grab-bag - eggs,
    nests, chicks, parasites - and we deliberately do NOT spend a call
    filtering it: the generation prompt tells the model to use only the
    photos that actually show a bird. Article order, so the lead photo
    (reliably the bird) comes first. Returns [(bytes, ext)].
    """
    out: list[tuple[bytes, str]] = []
    with httpx.Client(
        headers={"User-Agent": WIKI_UA}, follow_redirects=True, timeout=30
    ) as client:
        for title in _wiki_titles(sci_name, com_name):
            try:
                r = client.get(
                    "https://en.wikipedia.org/api/rest_v1/page/media-list/"
                    + urllib.parse.quote(title.replace(" ", "_"))
                )
                items = r.raise_for_status().json().get("items", [])
            except (httpx.HTTPError, KeyError):
                continue
            for item in items:
                if len(out) >= REF_COUNT:
                    break
                if item.get("type") != "image":
                    continue
                srcset = item.get("srcset") or []
                if not srcset:
                    continue
                src = srcset[-1].get("src") or srcset[0].get("src") or ""
                if not src.lower().endswith((".jpg", ".jpeg")):
                    continue
                if src.startswith("//"):
                    src = "https:" + src
                try:
                    data = client.get(src).raise_for_status().content
                except httpx.HTTPError:
                    continue
                # Magic-byte sniff: the URL extension is a hint, the bytes are
                # what the model's MIME check sees.
                if data.startswith(b"\xff\xd8\xff"):
                    out.append((data, ".jpg"))
                elif data.startswith(b"\x89PNG\r\n\x1a\n"):
                    out.append((data, ".png"))
            if out:
                break
    return out


def select_bird_photos(
    candidates: list[tuple[bytes, str]], sci_name: str, com_name: str
) -> list[int]:
    """Drop the candidates that are not an adult bird, on the CHEAP model.

    The economics, which are not what they look like. Filtering saves only
    ~$0.0006 of image-input tokens on the expensive call - nothing. What it
    actually buys is risk: an egg or a flea slide in the reference set is a
    plausible cause of a BAD generation, and a bad generation costs a
    ~$0.024 Pro retry, 42x the saving. The filter pays for itself if it
    prevents one retry in fifty birds.

    It is also nearly free in the first place, because ensure_references
    caches to disk - so this runs ONCE PER SPECIES, ever, not per
    illustration. And it lets the generation prompt stop spending Pro's
    instruction-following budget on "ignore the photos that aren't birds",
    which is budget we would rather it spent on plumage.

    Returns candidate indices, best first, capped at REF_COUNT.
    """
    if not candidates:
        return []
    parts: list[dict] = [
        {
            "type": "text",
            "text": (
                f"These are candidate reference photos for {com_name} ({sci_name}), "
                f"numbered from 0. They come straight from the species' Wikipedia "
                f"article, so some are not the bird at all.\n\n"
                f"KEEP only images clearly showing a live ADULT bird of this species "
                f"with its plumage visible. REJECT eggs, nests, chicks or juveniles, "
                f"museum specimens, skeletons, parasites and insects, maps, diagrams, "
                f"habitat shots, and any other species.\n\n"
                f"Rank the keepers best first, preferring a sharp, well-lit bird seen "
                f"side-on with the head, throat and underparts visible.\n\n"
                f'Reply with JSON only: {{"keep": [<indices, best first>]}}'
            ),
        }
    ]
    for i, (data, ext) in enumerate(candidates):
        mime = "image/png" if ext == ".png" else "image/jpeg"
        parts.append({"type": "text", "text": f"Image {i}:"})
        parts.append({"type": "image_url", "image_url": {"url": _data_url(data, mime)}})

    resp = _post_with_retry(
        {"model": CRITIC_MODEL, "messages": [{"role": "user", "content": parts}]}
    )
    choices = resp.get("choices") or [{}]
    reply = (choices[0].get("message", {}).get("content") or "").strip()

    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return [0]  # fall back to the lead photo, which is reliably the bird
    try:
        keep = json.loads(m.group(0)).get("keep", [])
    except json.JSONDecodeError:
        return [0]
    valid = [i for i in keep if isinstance(i, int) and 0 <= i < len(candidates)]
    return valid[:REF_COUNT] or [0]


def ensure_references(sci_name: str, com_name: str) -> list[Path]:
    """Cache-or-fetch the reference photos for a species, filtered to birds.

    Pre-existing wikipedia/<Sci name>.jpg, -2.jpg, -3.jpg (hand-picked or
    previously fetched) are used as-is, with NO network and NO model call.
    That is both the escape hatch when an article's photos are poor - drop
    better ones in under those names - and the reason the filter below is
    affordable: it runs once per species and never again.
    """
    REFS_DIR.mkdir(exist_ok=True)

    def cached() -> list[Path]:
        found = []
        for n in range(1, REF_COUNT + 1):
            stem = sci_name if n == 1 else f"{sci_name}-{n}"
            for ext in (".jpg", ".png"):
                p = REFS_DIR / f"{stem}{ext}"
                if p.exists() and p.stat().st_size > 1024:
                    found.append(p)
                    break
        return found

    if hits := cached():
        return hits

    candidates = fetch_photos(sci_name, com_name)
    paths = []
    for n, idx in enumerate(select_bird_photos(candidates, sci_name, com_name), 1):
        data, ext = candidates[idx]
        stem = sci_name if n == 1 else f"{sci_name}-{n}"
        path = REFS_DIR / f"{stem}{ext}"
        path.write_bytes(data)
        paths.append(path)
    return paths


# ---- Generation ----


def _reference_parts(refs: list[Path], com_name: str) -> list[dict]:
    parts: list[dict] = []
    for i, ref in enumerate(refs, 1):
        parts.append(
            {
                "type": "text",
                "text": (
                    f"REFERENCE PHOTO {i} of {len(refs)} - adult {com_name}, "
                    f"anatomy and colour only:"
                ),
            }
        )
        parts.append({"type": "image_url", "image_url": {"url": _downscaled(ref)}})
    return parts


def critique(
    png: bytes,
    refs: list[Path],
    description: str | None,
    sci_name: str,
    com_name: str,
    pose: str,
) -> str | None:
    """Check a generated illustration, on the cheap model, for errors bad
    enough to justify paying for a second image.

    Returns a correction list, or None to accept. The bar is deliberately
    HIGH: this must not burn an expensive retry on a quibble about a hue or
    a brush stroke. Only two things qualify - a plumage error that makes it
    the wrong species, and a violation of the hard constraints that would
    break the cutout step downstream.
    """
    parts: list[dict] = [
        {
            "type": "text",
            "text": (
                f"You are a birding expert reviewing a stylised woodblock-print "
                f"illustration of {com_name} ({sci_name}). Regenerating it is "
                f"EXPENSIVE, so reject it ONLY for a serious, unambiguous error.\n\n"
                f"REJECT for either of these:\n"
                f"(a) A plumage error that would make a birder call it a DIFFERENT "
                f"species: a diagnostic marking missing, or one present that this "
                f"species does not have, or a colour zone whose EXTENT is plainly "
                f"wrong (a pale throat that stops short of the bill when it should "
                f"reach it, a dark cap that swallows the cheek, a missing pale "
                f"rump). Be strict here: this species must not be drawn as its "
                f"commoner lookalike.\n"
                f"(b) A hard-constraint violation: a branch, perch, twig, leaf or "
                f"any scenery drawn; a border or frame; any body part cropped at "
                f"the frame edge. The background must be plain cream and nothing "
                f"else - anything drawn on it breaks the next pipeline stage.\n\n"
                f"Do NOT reject for style, brushwork, exact hue, or charm. The "
                f"flatness and the simplification are INTENTIONAL.\n\n"
                f"The bird was ASKED for {POSES[pose]}, so do not reject it for "
                f"being in that pose. In particular, a flying bird correctly has "
                f"its feet tucked against the belly or trailed behind the tail, and "
                f"a perched bird correctly floats with no perch drawn under it - "
                f"neither is an error.\n\n"
                f"If it is acceptable, reply with exactly: OK\n"
                f"Otherwise reply with a short bulleted list of the serious errors "
                f"only, each saying what is wrong and what it must be instead."
            ),
        }
    ]
    if description:
        parts.append(
            {
                "type": "text",
                "text": f"Authoritative description of {com_name}:\n{description}",
            }
        )
    parts += _reference_parts(refs, com_name)
    parts.append({"type": "text", "text": "THE ILLUSTRATION UNDER REVIEW:"})
    parts.append({"type": "image_url", "image_url": {"url": _data_url(png)}})

    resp = _post_with_retry(
        {"model": CRITIC_MODEL, "messages": [{"role": "user", "content": parts}]}
    )
    choices = resp.get("choices") or [{}]
    verdict = (choices[0].get("message", {}).get("content") or "").strip()
    # Empty or unparseable verdict means "no evidence of a serious problem".
    # Failing open is right here: a spurious retry costs real money.
    if not verdict or verdict.upper().startswith("OK"):
        return None
    return verdict


def illustration_path(sci_name: str, pose: str) -> Path:
    """Where a (species, pose) illustration lives: 'Hirundo rustica, perched.png'.

    Single source of truth for the naming convention - the cutout and
    collage stages resolve files through this, so the format only ever
    changes in one place.
    """
    if pose not in POSES:
        raise ValueError(f"unknown pose {pose!r}; expected one of {sorted(POSES)}")
    return OUT_DIR / f"{sci_name}, {pose}.png"


def _generate(
    sci_name: str,
    com_name: str,
    pose: str,
    refs: list[Path],
    description: str | None,
    corrections: str | None,
) -> bytes:
    body = prompt_template.render(
        sci_name=sci_name,
        com_name=com_name,
        pose=pose,
        pose_phrase=POSES[pose],
        n_refs=len(refs),
        description=description,
        corrections=corrections,
    )
    content: list[dict] = [{"type": "text", "text": body}]
    content += _reference_parts(refs, com_name)

    resp = _post_with_retry(
        {
            "model": IMAGE_MODEL,
            "messages": [{"role": "user", "content": content}],
            # Both modalities so the model can surface refusal/safety text
            # without rejecting the request shape.
            "modalities": ["image", "text"],
        }
    )
    for choice in resp.get("choices", []):
        for img in choice.get("message", {}).get("images") or []:
            url = (img.get("image_url") or {}).get("url", "")
            if url.startswith("data:") and ";base64," in url:
                return base64.b64decode(url.split(";base64,", 1)[1])
    # No image - surface the finish/error reason so users know what to fix.
    finish = (resp.get("choices") or [{}])[0].get("finish_reason", "?")
    error = resp.get("error", {}).get("message", "")
    raise RuntimeError(f"no image (finish={finish} error={error})")


def get_bird_illustration(
    sci_name: str,
    com_name: str,
    pose: str = "perched",
    verify: bool = True,
    overwrite: bool = True,
    verbose: bool = True,
) -> bytes:
    """Generate a faithful illustration of the species on a flat cream ground.

    Grounds the model with the species' Wikipedia description and reference
    photos so the FIRST image lands. Then one cheap critique; and if - and
    only if - it finds a serious error, exactly ONE retry. Never more: two
    expensive calls is the hard ceiling per bird.

    pose: "perched" or "flying" (see POSES). Each is a separate image, so a
    species in both poses costs two generations.

    Returns the PNG bytes and writes them to
    illustrations/<sci_name>, <pose>.png.

    verify=False skips the critique and the retry (one image call, no safety
    net). overwrite=False returns an already-generated file untouched instead
    of paying to redraw it - use it when re-running over a species list.
    """
    if pose not in POSES:
        # Fail here, before spending anything, rather than writing an image
        # to a filename the rest of the pipeline will never look for.
        raise ValueError(f"unknown pose {pose!r}; expected one of {sorted(POSES)}")

    def say(msg: str) -> None:
        if verbose:
            print(f"  [{com_name}, {pose}] {msg}")

    out = illustration_path(sci_name, pose)
    if not overwrite and out.exists():
        say(f"already exists, skipping (no API call): {out.name}")
        return out.read_bytes()

    description = fetch_description(sci_name, com_name)
    refs = ensure_references(sci_name, com_name)
    say(
        f"grounding: {len(refs)} photo(s), description {'yes' if description else 'MISSING'}"
    )
    if not refs and not description:
        say("WARNING: no photo and no description - the output will be unreliable")

    png = _generate(sci_name, com_name, pose, refs, description, None)

    if verify:
        errors = critique(png, refs, description, sci_name, com_name, pose)
        if errors is None:
            say("verified: accepted on the first attempt")
        else:
            say("rejected, retrying ONCE with corrections:")
            if verbose:
                print("    " + errors.replace("\n", "\n    "))
            png = _generate(sci_name, com_name, pose, refs, description, errors)
            # No second critique: we would not act on it, and it costs money.
            say("retry done (not re-checked - one retry is the ceiling)")

    OUT_DIR.mkdir(exist_ok=True)
    out.write_bytes(png)
    return png


if __name__ == "__main__":
    for pose in POSES:
        png = get_bird_illustration("Turdus merula", "Eurasian Blackbird", pose=pose)
        path = illustration_path("Turdus merula", pose)
        print(f"saved {path.name} ({len(png) // 1024} KB)")
