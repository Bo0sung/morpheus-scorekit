from __future__ import annotations

import argparse
import json
from pathlib import Path

from morpheus_scorekit.freefall_segments import trim_precontact
from morpheus_scorekit.models import Trajectory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trim a tracked falling-ball trajectory before its first estimated contact."
    )
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--minimum-frames", type=int, default=8)
    args = parser.parse_args()

    trajectory = Trajectory.load(args.trajectory)
    trimmed, trim_info = trim_precontact(
        trajectory,
        object_id=args.object_id,
        minimum_freefall_frames=args.minimum_frames,
    )
    path = trimmed.save(args.output)
    print(
        json.dumps(
            {"trajectory": str(path), "trim": trim_info, "quality": trimmed.quality_summary()},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
