from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from morpheus_scorekit.extraction import SeededColorTracker
from morpheus_scorekit.scoring import MorpheusTrajectoryScorer, ScoringConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ScoreKit on an official real-world clip.")
    parser.add_argument("--data-root", default="validation_data")
    parser.add_argument("--experiment", default="falling_ball")
    parser.add_argument("--video", default="video_0_fps30")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--output", default="validation_results/falling_ball")
    args = parser.parse_args()

    clip = ROOT / args.data_root / "real-world-cropped" / args.experiment / args.video
    frames = clip / "frames_for_tracking"
    if not frames.exists():
        raise FileNotFoundError(
            f"sample not found: {frames}\n"
            "Run: python scripts/download_validation_sample.py"
        )

    labels_path = ROOT / "vendor" / "Morpheus" / "prompts" / "reference_images" / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    point = labels[args.experiment][args.video]["object_1"]["positive"][0]

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    trajectory = SeededColorTracker().extract(
        frames,
        {1: (float(point[0]), float(point[1]))},
        fps=args.fps,
        experiment=args.experiment,
        overlay_dir=output / "tracking_overlay",
    )
    trajectory_path = trajectory.save(output / "trajectory.npz")

    scorer = MorpheusTrajectoryScorer(ScoringConfig(
        reference_repo=ROOT / "vendor" / "Morpheus",
        epochs=args.epochs,
        source_category="real_world_trajectories",
    ))
    result = scorer.score(trajectory, output_dir=output / "scores")
    summary = {
        "source": str(clip),
        "seed_xy": point,
        "trajectory": str(trajectory_path),
        "quality": trajectory.quality_summary(),
        "scores": {
            "physical_invariance": result.get("physical_score"),
            "dynamical": result.get("statistical_score"),
            "total": result.get("total_score"),
        },
    }
    (output / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

