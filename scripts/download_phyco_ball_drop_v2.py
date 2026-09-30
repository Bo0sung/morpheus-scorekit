from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download only the rigid single-ball free-fall split from PhyCo-Kubric."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_data/phyco_kubric"),
        help="Local dataset root (default: validation_data/phyco_kubric)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    downloaded = snapshot_download(
        repo_id="nnsriram97/phyco_kubric",
        repo_type="dataset",
        allow_patterns=["ball_drop_v2/*"],
        local_dir=args.output,
    )
    print(downloaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
