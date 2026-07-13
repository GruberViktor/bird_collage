"""Build today's bird collage from the BirdNET-Go daily species list.

    uv run python main.py              # today's birds from birdie
    uv run python main.py --example    # the recorded sample, no BirdNET needed

Everything comes together here:

    get_bird_summary()   what BirdNET-Go heard today, with counts
    build_collage()      draw / segment / pack / composite  -> collage.png

Illustrations are generated ON DEMAND, only for the species actually heard
and only in the pose the collage picked for them. Nothing is pre-rendered,
so the ~500 species that might live here cost nothing until one shows up.
Both the illustration and its cutout cache to disk, so the first collage of
a new day costs a few cents and every rebuild of it is free.
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent / "illustration_generator"))
load_dotenv()

from collage import build_collage  # noqa: E402

BIRDNET_URL = "http://localhost:8080/api/v2/analytics/species/daily"


def get_bird_summary() -> list[dict]:
    return httpx.get(BIRDNET_URL, timeout=30).raise_for_status().json()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--example",
        action="store_true",
        help="use the recorded sample response instead of querying BirdNET-Go",
    )
    ap.add_argument("--out", type=Path, default=Path("collage.png"))
    ap.add_argument("--seed", type=int, help="fix the random perched/flying picks")
    ap.add_argument(
        "--curve",
        default="linear",
        choices=["linear", "sqrt", "log"],
        help="how count maps to size; sqrt/log give the rarer birds more presence",
    )
    ap.add_argument(
        "--outline",
        action="store_true",
        help="draw the packing outlines, to eyeball the spacing",
    )
    args = ap.parse_args()

    if args.example:
        from daily_species_example import response as species
    else:
        try:
            species = get_bird_summary()
            species = list(
                filter(lambda s: s.get("species_code") is not None and s.get("count") > 1, species)
            )  # only keep birds, no grasshoppers or other non-birds
        except httpx.HTTPError as e:
            print(f"could not reach BirdNET-Go at {BIRDNET_URL}: {e}")
            print("(run with --example to use the recorded sample instead)")
            raise SystemExit(1)

    print(f"{len(species)} species heard today\n")
    build_collage(
        species,
        out=args.out,
        seed=args.seed,
        curve=args.curve,
        outline=args.outline,
    )


if __name__ == "__main__":
    main()
