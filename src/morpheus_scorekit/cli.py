from __future__ import annotations

import argparse
import json
from pathlib import Path

from .auto_seed import suggest_moving_object_seed
from .extraction import SeededColorTracker, TrackingConfig
from .models import Trajectory
from .sam2_extraction import SAM2DepthConfig, SAM2DepthTrajectoryExtractor
from .scoring import MorpheusTrajectoryScorer, ScoringConfig


def _add_sam2_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference-repo", default="vendor/Morpheus")
    parser.add_argument("--checkpoint", help="SAM2.1 checkpoint (.pt); defaults to vendor/Morpheus/checkpoints")
    parser.add_argument("--config-dir", help="Directory containing the SAM2 Hydra YAML")
    parser.add_argument("--model-cfg", default="sam2.1_hiera_l.yaml")
    parser.add_argument("--depth-model", default="nielsr/depth-anything-large",
                        help="Hugging Face model ID or local Depth Anything directory")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--no-depth", action="store_true",
                        help="Track in 2D only and store zero depth (scores use image-plane motion)")


def _sam2_config(args) -> SAM2DepthConfig:
    return SAM2DepthConfig(
        reference_repo=args.reference_repo,
        checkpoint=args.checkpoint,
        config_dir=args.config_dir,
        model_cfg=args.model_cfg,
        depth_model=args.depth_model,
        device=args.device,
        use_depth=not args.no_depth,
        save_masks=bool(getattr(args, "mask_dir", None)),
    )


def _point_maps(args) -> tuple[dict[int, list[tuple[float, float]]], dict[int, list[tuple[float, float]]]]:
    if getattr(args, "auto_seed", False):
        result = suggest_moving_object_seed(args.source, preview_path=args.seed_preview)
        print(json.dumps({
            "auto_seed": {"object_id": 1, "x": result.x, "y": result.y,
                          "area": result.area, "confidence": result.confidence},
            "preview": args.seed_preview,
        }, indent=2, ensure_ascii=False))
        return {1: [(result.x, result.y)]}, {}
    positives: dict[int, list[tuple[float, float]]] = {}
    negatives: dict[int, list[tuple[float, float]]] = {}
    for raw in args.seed:
        object_id, x, y = int(raw[0]), float(raw[1]), float(raw[2])
        positives.setdefault(object_id, []).append((x, y))
    for raw in args.negative or []:
        object_id, x, y = int(raw[0]), float(raw[1]), float(raw[2])
        negatives.setdefault(object_id, []).append((x, y))
    return positives, negatives


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract trajectories and compute Morpheus scores.")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="video/frames -> trajectory NPZ")
    extract.add_argument("source")
    extract.add_argument("--seed", nargs=2, type=float, metavar=("X", "Y"), required=True)
    extract.add_argument("--object-id", type=int, default=1)
    extract.add_argument("--fps", type=float, default=30.0)
    extract.add_argument("--experiment", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--overlay-dir")

    suggest = sub.add_parser(
        "suggest-seed", help="estimate a first-frame click for one moving object"
    )
    suggest.add_argument("source")
    suggest.add_argument("--preview", default="auto_seed_preview.jpg")

    sam_extract = sub.add_parser("extract-sam2", help="video/frames -> trajectory NPZ using SAM2 + Depth Anything")
    sam_extract.add_argument("source")
    sam_extract_seed = sam_extract.add_mutually_exclusive_group(required=True)
    sam_extract_seed.add_argument("--seed", nargs=3, action="append",
                             metavar=("OBJECT_ID", "X", "Y"),
                             help="Positive first-frame click; repeat for multiple points/objects")
    sam_extract_seed.add_argument("--auto-seed", action="store_true",
                                  help="Detect one moving object in a static-camera video")
    sam_extract.add_argument("--seed-preview", default="auto_seed_preview.jpg",
                             help="Annotated first frame written when --auto-seed is used")
    sam_extract.add_argument("--negative", nargs=3, action="append",
                             metavar=("OBJECT_ID", "X", "Y"),
                             help="Negative first-frame click; may be repeated")
    sam_extract.add_argument("--fps", type=float)
    sam_extract.add_argument("--experiment", required=True)
    sam_extract.add_argument("--output", required=True)
    sam_extract.add_argument("--overlay-dir")
    sam_extract.add_argument("--mask-dir")
    _add_sam2_model_arguments(sam_extract)

    doctor = sub.add_parser("sam2-doctor", help="check whether SAM2/Depth Anything assets are connected")
    _add_sam2_model_arguments(doctor)

    score = sub.add_parser("score", help="trajectory NPZ -> Morpheus scores")
    score.add_argument("trajectory")
    score.add_argument("--experiment")
    score.add_argument("--output-dir", required=True)
    score.add_argument("--epochs", type=int, default=200_000)
    score.add_argument("--reference-repo", default="vendor/Morpheus")
    score.add_argument("--generated", action="store_true",
                       help="Use the generated-video 25%% physical-score window profile.")
    score.add_argument("--only-physical", action="store_true")
    score.add_argument("--only-dynamical", action="store_true")

    run = sub.add_parser("run", help="extract then score")
    run.add_argument("source")
    run.add_argument("--seed", nargs=2, type=float, metavar=("X", "Y"), required=True)
    run.add_argument("--fps", type=float, default=30.0)
    run.add_argument("--experiment", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--epochs", type=int, default=10_000)
    run.add_argument("--reference-repo", default="vendor/Morpheus")

    sam_run = sub.add_parser("run-sam2", help="SAM2 extraction followed by Morpheus scoring")
    sam_run.add_argument("source")
    sam_run_seed = sam_run.add_mutually_exclusive_group(required=True)
    sam_run_seed.add_argument("--seed", nargs=3, action="append",
                         metavar=("OBJECT_ID", "X", "Y"))
    sam_run_seed.add_argument("--auto-seed", action="store_true",
                              help="Detect one moving object in a static-camera video")
    sam_run.add_argument("--seed-preview", default="auto_seed_preview.jpg")
    sam_run.add_argument("--negative", nargs=3, action="append",
                         metavar=("OBJECT_ID", "X", "Y"))
    sam_run.add_argument("--fps", type=float)
    sam_run.add_argument("--experiment", required=True)
    sam_run.add_argument("--output-dir", required=True)
    sam_run.add_argument("--epochs", type=int, default=200_000)
    sam_run.add_argument("--generated", action="store_true")
    sam_run.add_argument("--mask-dir")
    _add_sam2_model_arguments(sam_run)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "suggest-seed":
        result = suggest_moving_object_seed(args.source, preview_path=args.preview)
        print(json.dumps({
            "object_id": 1,
            "x": result.x,
            "y": result.y,
            "area": result.area,
            "confidence": result.confidence,
            "preview": args.preview,
        }, indent=2, ensure_ascii=False))
        return 0

    if args.command == "sam2-doctor":
        report = SAM2DepthTrajectoryExtractor(_sam2_config(args)).environment_report()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ready"] else 2

    if args.command == "extract":
        tracker = SeededColorTracker(TrackingConfig())
        trajectory = tracker.extract(
            args.source,
            {args.object_id: tuple(args.seed)},
            fps=args.fps,
            experiment=args.experiment,
            overlay_dir=args.overlay_dir,
        )
        path = trajectory.save(args.output)
        print(json.dumps({"trajectory": str(path), "quality": trajectory.quality_summary()}, indent=2))
        return 0

    if args.command == "extract-sam2":
        positives, negatives = _point_maps(args)
        trajectory = SAM2DepthTrajectoryExtractor(_sam2_config(args)).extract(
            args.source,
            positives,
            negative_points_xy=negatives,
            fps=args.fps,
            experiment=args.experiment,
            overlay_dir=args.overlay_dir,
            mask_dir=args.mask_dir,
        )
        path = trajectory.save(args.output)
        print(json.dumps({"trajectory": str(path), "quality": trajectory.quality_summary()},
                         indent=2, ensure_ascii=False))
        return 0

    if args.command == "score":
        trajectory = Trajectory.load(args.trajectory)
        scorer = MorpheusTrajectoryScorer(ScoringConfig(
            reference_repo=args.reference_repo,
            epochs=args.epochs,
            source_category="VGM_trajectories" if args.generated else "real_world_trajectories",
            compute_physical=not args.only_dynamical,
            compute_dynamical=not args.only_physical,
        ))
        result = scorer.score(trajectory, args.experiment, output_dir=args.output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0


    if args.command == "run-sam2":
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        positives, negatives = _point_maps(args)
        trajectory = SAM2DepthTrajectoryExtractor(_sam2_config(args)).extract(
            args.source,
            positives,
            negative_points_xy=negatives,
            fps=args.fps,
            experiment=args.experiment,
            overlay_dir=output / "tracking_overlay",
            mask_dir=args.mask_dir,
        )
        trajectory_path = trajectory.save(output / "trajectory.npz")
        scorer = MorpheusTrajectoryScorer(ScoringConfig(
            reference_repo=args.reference_repo,
            epochs=args.epochs,
            source_category="VGM_trajectories" if args.generated else "real_world_trajectories",
        ))
        result = scorer.score(trajectory, output_dir=output / "scores")
        print(json.dumps({"trajectory": str(trajectory_path), "quality": trajectory.quality_summary(),
                          "scores": result}, indent=2, ensure_ascii=False))
        return 0

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory = SeededColorTracker().extract(
        args.source,
        {1: tuple(args.seed)},
        fps=args.fps,
        experiment=args.experiment,
        overlay_dir=output / "tracking_overlay",
    )
    trajectory_path = trajectory.save(output / "trajectory.npz")
    scorer = MorpheusTrajectoryScorer(ScoringConfig(
        reference_repo=args.reference_repo,
        epochs=args.epochs,
    ))
    result = scorer.score(trajectory, output_dir=output / "scores")
    print(json.dumps({"trajectory": str(trajectory_path), "scores": result}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
