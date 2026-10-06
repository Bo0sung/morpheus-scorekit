from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy pre-contact Kubric RGBA frames for Morpheus tracking."
    )
    parser.add_argument("pair_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    pair_dir = args.pair_dir.resolve()
    output_dir = args.output_dir.resolve()
    for split in ("normal", "violation"):
        source = pair_dir / split
        state_path = source / "state.npz"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)

        with np.load(state_path) as state:
            contacts = np.flatnonzero(state["contact_ball_floor"])
            frame_count = len(state["frame"])
        end = int(contacts[0]) if len(contacts) else frame_count
        if end < 3:
            raise ValueError(f"{split}: fewer than three pre-contact frames ({end})")

        destination = output_dir / split
        destination.mkdir(parents=True, exist_ok=True)
        for stale in destination.glob("*.png"):
            stale.unlink()

        for frame in range(end):
            source_frame = source / f"rgba_{frame:05d}.png"
            if not source_frame.is_file():
                raise FileNotFoundError(source_frame)
            shutil.copy2(source_frame, destination / f"{frame:05d}.png")
        print(f"{split}: copied {end} pre-contact frames to {destination}")


if __name__ == "__main__":
    main()
