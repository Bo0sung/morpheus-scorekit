from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import cv2
import numpy as np

from .extraction import _read_frames
from .models import Trajectory


@dataclass(frozen=True)
class SAM2DepthConfig:
    """Model locations for the paper-faithful trajectory extractor.

    No model is loaded while this object is created.  Loading happens on the
    first call to :meth:`SAM2DepthTrajectoryExtractor.extract`.
    """

    reference_repo: str | Path = "vendor/Morpheus"
    checkpoint: str | Path | None = None
    config_dir: str | Path | None = None
    model_cfg: str = "sam2.1_hiera_l.yaml"
    depth_model: str = "nielsr/depth-anything-large"
    device: str | None = None
    use_depth: bool = True
    save_masks: bool = False

    def resolved(self) -> "SAM2DepthConfig":
        repo = Path(self.reference_repo).resolve()
        return SAM2DepthConfig(
            reference_repo=repo,
            checkpoint=Path(self.checkpoint).resolve() if self.checkpoint else repo / "checkpoints" / "sam2.1_hiera_large.pt",
            config_dir=Path(self.config_dir).resolve() if self.config_dir else repo / "configs",
            model_cfg=self.model_cfg,
            depth_model=self.depth_model,
            device=self.device,
            use_depth=self.use_depth,
            save_masks=self.save_masks,
        )


def _normalise_prompts(
    positive_points_xy: Mapping[int, Sequence[tuple[float, float]] | tuple[float, float]],
    negative_points_xy: Mapping[int, Sequence[tuple[float, float]]] | None,
) -> dict[int, dict[str, np.ndarray]]:
    """Convert friendly point dictionaries to the format expected by SAM2."""

    negative_points_xy = negative_points_xy or {}
    object_ids = sorted(set(positive_points_xy) | set(negative_points_xy))
    if not object_ids:
        raise ValueError("at least one positive SAM2 point is required")
    if object_ids != list(range(1, len(object_ids) + 1)):
        raise ValueError("SAM2 object IDs must be consecutive positive integers starting at 1")

    prompts: dict[int, dict[str, np.ndarray]] = {}
    for object_id in object_ids:
        raw_positive = positive_points_xy.get(object_id, [])
        if isinstance(raw_positive, tuple) and len(raw_positive) == 2 and all(
            np.isscalar(value) for value in raw_positive
        ):
            positives = [raw_positive]
        else:
            positives = list(raw_positive)  # type: ignore[arg-type]
        negatives = list(negative_points_xy.get(object_id, []))
        if not positives:
            raise ValueError(f"object {object_id} needs at least one positive point")
        points = np.asarray([*positives, *negatives], dtype=np.float32)
        labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
        prompts[object_id] = {"points": points, "labels": labels}
    return prompts


@contextmanager
def _sam2_frames(source: str | Path) -> Iterator[tuple[Path, list[np.ndarray], float | None]]:
    """Materialise any supported input as SAM2's ``00000.jpg`` frame layout."""

    frames, detected_fps = _read_frames(source)
    with tempfile.TemporaryDirectory(prefix="morpheus_sam2_") as temporary:
        frames_dir = Path(temporary)
        for index, frame in enumerate(frames):
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
            if not ok:
                raise RuntimeError(f"failed to encode frame {index}")
            encoded.tofile(frames_dir / f"{index:05d}.jpg")
        yield frames_dir, frames, detected_fps


class HuggingFaceDepthEstimator:
    """Small lazy wrapper around a configurable Depth Anything checkpoint."""

    def __init__(self, model: str, device: str | None = None):
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - exercised only with optional deps
            raise RuntimeError(
                "Depth Anything requires torch and transformers; install with: pip install -e '.[sam2]'"
            ) from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
            )
        pipeline_device: int | str = 0 if device == "cuda" else ("mps" if device == "mps" else -1)
        self.model = model
        self._pipeline = pipeline("depth-estimation", model=model, device=pipeline_device)

    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        from PIL import Image

        result = self._pipeline(Image.fromarray(rgb_image))
        depth = result["depth"]
        return np.asarray(depth, dtype=np.float32)


class SAM2DepthTrajectoryExtractor:
    """Run SAM2 masks and optional Depth Anything lifting, returning a Trajectory.

    ``tracker`` and ``depth_estimator`` are injectable for testing or for callers
    that already own loaded model instances.  With neither supplied, the pinned
    official Morpheus tracker is loaded from ``reference_repo``.
    """

    def __init__(
        self,
        config: SAM2DepthConfig | None = None,
        *,
        tracker: Any | None = None,
        depth_estimator: Any | None = None,
    ):
        self.config = (config or SAM2DepthConfig()).resolved()
        self._tracker = tracker
        self._depth_estimator = depth_estimator

    def environment_report(self) -> dict[str, Any]:
        cfg = self.config
        repo = Path(cfg.reference_repo)
        checkpoint = Path(cfg.checkpoint)  # type: ignore[arg-type]
        config_file = Path(cfg.config_dir) / cfg.model_cfg  # type: ignore[arg-type]
        modules = {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "torchvision", "transformers", "hydra", "PIL")
        }
        sam2_source = repo / "sam2" / "sam2" / "__init__.py"
        report = {
            "reference_repo": str(repo),
            "reference_repo_exists": repo.is_dir(),
            "checkpoint": str(checkpoint),
            "checkpoint_exists": checkpoint.is_file(),
            "config_file": str(config_file),
            "config_exists": config_file.is_file(),
            "vendored_sam2_exists": sam2_source.is_file(),
            "python_modules": modules,
            "depth_model": cfg.depth_model,
        }
        report["ready"] = bool(
            report["reference_repo_exists"]
            and report["checkpoint_exists"]
            and report["config_exists"]
            and report["vendored_sam2_exists"]
            and modules["torch"]
            and modules["torchvision"]
            and modules["hydra"]
            and modules["PIL"]
            and (not cfg.use_depth or modules["transformers"])
        )
        return report

    def _load_models(self) -> tuple[Any, Any | None]:
        if self._tracker is None:
            report = self.environment_report()
            missing = []
            if not report["checkpoint_exists"]:
                missing.append(f"SAM2 checkpoint: {report['checkpoint']}")
            if not report["config_exists"]:
                missing.append(f"SAM2 config: {report['config_file']}")
            if not report["vendored_sam2_exists"]:
                missing.append("vendored SAM2 submodule")
            if missing:
                raise FileNotFoundError("missing model assets: " + ", ".join(missing))

            repo = Path(self.config.reference_repo)
            for path in (repo, repo / "sam2"):
                path_text = str(path)
                if path_text not in sys.path:
                    sys.path.insert(0, path_text)
            try:
                from morpheus.tracking.sam2_tracker import Sam2Tracker
            except ImportError as exc:
                raise RuntimeError(
                    "SAM2 dependencies are not installed. Run: pip install -e '.[sam2]' and "
                    "pip install -e vendor/Morpheus/sam2"
                ) from exc
            self._tracker = Sam2Tracker(
                checkpoint=str(self.config.checkpoint),
                config_dir=str(self.config.config_dir),
                model_cfg=self.config.model_cfg,
                device=self.config.device,
                load_depth=False,
            )

        if self.config.use_depth and self._depth_estimator is None:
            self._depth_estimator = HuggingFaceDepthEstimator(
                self.config.depth_model, device=self.config.device
            )
        return self._tracker, self._depth_estimator

    def extract(
        self,
        source: str | Path,
        positive_points_xy: Mapping[int, Sequence[tuple[float, float]] | tuple[float, float]],
        *,
        negative_points_xy: Mapping[int, Sequence[tuple[float, float]]] | None = None,
        fps: float | None = None,
        experiment: str | None = None,
        overlay_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
    ) -> Trajectory:
        prompts = _normalise_prompts(positive_points_xy, negative_points_xy)
        tracker, depth_estimator = self._load_models()
        overlays = Path(overlay_dir) if overlay_dir else None
        masks_out = Path(mask_dir) if mask_dir else None
        if overlays:
            overlays.mkdir(parents=True, exist_ok=True)
        if masks_out:
            masks_out.mkdir(parents=True, exist_ok=True)

        with _sam2_frames(source) as (frames_dir, bgr_frames, detected_fps):
            state, first_masks = tracker.initialize_tracking(str(frames_dir), prompts)
            segments = tracker.propagate(state)
            # Some predictor versions yield propagation starting at frame 1.
            if 0 not in segments:
                segments[0] = first_masks

            tracks = {
                object_id: np.full((len(bgr_frames), 3), np.nan, dtype=np.float32)
                for object_id in prompts
            }
            for frame_index, frame in enumerate(bgr_frames):
                frame_masks = segments.get(frame_index, {})
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                depth = depth_estimator.predict(rgb) if depth_estimator is not None and frame_masks else None
                rendered = frame.copy()
                for object_id in prompts:
                    if object_id not in frame_masks:
                        continue
                    mask = np.squeeze(np.asarray(frame_masks[object_id])) > 0
                    if mask.shape != frame.shape[:2]:
                        mask = cv2.resize(
                            mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    indices = np.argwhere(mask)
                    if not len(indices):
                        continue
                    y, x = indices.mean(axis=0)
                    z = 0.0
                    if depth is not None:
                        if depth.shape != mask.shape:
                            depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
                        z = float(np.mean(depth[mask]))
                    tracks[object_id][frame_index] = (float(y), float(x), z)
                    rendered[mask] = (0.55 * rendered[mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
                    cv2.circle(rendered, (int(round(x)), int(round(y))), 5, (0, 0, 255), -1)
                    if masks_out and (self.config.save_masks or mask_dir is not None):
                        encoded_mask = (mask.astype(np.uint8) * 255)
                        ok, encoded = cv2.imencode(".png", encoded_mask)
                        if ok:
                            encoded.tofile(masks_out / f"object_{object_id}_{frame_index:05d}.png")
                if overlays:
                    ok, encoded = cv2.imencode(".jpg", rendered)
                    if ok:
                        encoded.tofile(overlays / f"{frame_index:05d}.jpg")

        actual_fps = float(fps or detected_fps or 30.0)
        trajectory = Trajectory(
            fps=actual_fps,
            objects=tracks,
            experiment=experiment,
            metadata={
                "extractor": "sam2_depth_anything",
                "source": str(Path(source)),
                "coordinate_order": "y_x_depth",
                "paper_faithful_tracker": True,
                "sam2_checkpoint": str(self.config.checkpoint),
                "sam2_model_cfg": self.config.model_cfg,
                "depth_model": self.config.depth_model if self.config.use_depth else None,
                "positive_points_xy": {
                    str(key): np.asarray(value).reshape(-1, 2).tolist()
                    for key, value in positive_points_xy.items()
                },
            },
        )
        validation = trajectory.validate()
        if not validation["valid"]:
            raise ValueError("invalid SAM2 trajectory: " + "; ".join(validation["errors"]))
        return trajectory
