from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one official Morpheus real-world clip.")
    parser.add_argument("--output", default="validation_data")
    parser.add_argument("--experiment", default="falling_ball")
    parser.add_argument("--video", default="video_0_fps30")
    args = parser.parse_args()

    deps = Path(__file__).resolve().parents[1] / ".deps"
    if deps.exists():
        sys.path.insert(0, str(deps))
    from huggingface_hub import snapshot_download

    pattern = f"real-world-cropped/{args.experiment}/{args.video}/**"
    path = snapshot_download(
        repo_id="physics-from-video/morpheus-real-world",
        repo_type="dataset",
        local_dir=args.output,
        allow_patterns=[pattern],
        max_workers=4,
    )
    print(path)


if __name__ == "__main__":
    main()

