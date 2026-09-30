from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


SAMPLE_FILES = (
    "rgba.mp4",
    "segmentation.mp4",
    "depth.mp4",
    "metadata.json",
    "animation_data.pkl",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract one complete simulation sample from a PhyCo tar archive."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--index", type=int, default=0, help="Zero-based sample index")
    parser.add_argument("--output", type=Path, default=Path("validation_data/phyco_sample"))
    args = parser.parse_args()

    if args.index < 0:
        parser.error("--index must be non-negative")
    args.output.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.archive, "r:gz") as archive:
        members = {PurePosixPath(member.name): member for member in archive.getmembers() if member.isfile()}
        sample_roots = sorted(path.parent for path in members if path.name == "rgba.mp4")
        if not sample_roots:
            raise ValueError(f"no rgba.mp4 samples found in {args.archive}")
        if args.index >= len(sample_roots):
            raise IndexError(f"sample index {args.index} is outside 0..{len(sample_roots) - 1}")

        sample_root = sample_roots[args.index]
        extracted: list[str] = []
        for filename in SAMPLE_FILES:
            member = members.get(sample_root / filename)
            if member is None:
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            destination = args.output / filename
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(filename)

    summary = {
        "archive": str(args.archive.resolve()),
        "sample_index": args.index,
        "sample_root": str(sample_root),
        "extracted": extracted,
    }
    (args.output / "sample.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
