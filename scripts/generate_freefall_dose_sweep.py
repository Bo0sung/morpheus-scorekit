from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from morpheus_scorekit.freefall_sweep import build_sweep_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate matched Kubric gravity-dose pairs.")
    parser.add_argument("kubric_repo", type=Path)
    parser.add_argument("--count", type=int, default=5, help="Number of seeds")
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--doses", nargs="+", type=float, default=[0.9, 0.7, 0.5, 0.2, 0.0])
    parser.add_argument("--start-frame", type=int, default=8)
    parser.add_argument("--end-frame", type=int, default=20)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.kubric_repo.resolve()
    generator = repo / "scripts" / "generate_pair.sh"
    if not generator.is_file():
        raise FileNotFoundError(generator)
    manifest = build_sweep_manifest(
        repo,
        count=args.count,
        start_seed=args.start_seed,
        doses=args.doses,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    manifest_path = args.manifest or repo / "outputs" / "freefall_sweep_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for index, pair in enumerate(manifest["pairs"], start=1):
        pair_dir = Path(pair["pair_dir"])
        complete = all(
            (pair_dir / variant / "state.npz").is_file() for variant in ("normal", "violation")
        )
        prefix = f"[{index}/{len(manifest['pairs'])}] {pair['scene_id']}"
        if complete:
            print(f"{prefix}: already complete, skipping")
            continue
        command = [
            "bash",
            str(generator),
            pair["scene_id"],
            str(pair["seed"]),
            "gravity_scale",
            str(args.start_frame),
            str(args.end_frame),
            str(pair["dose"]),
        ]
        if args.dry_run:
            print(f"{prefix}: {' '.join(command)}")
        else:
            print(f"{prefix}: generating")
            subprocess.run(command, cwd=repo, env=os.environ.copy(), check=True)

    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
