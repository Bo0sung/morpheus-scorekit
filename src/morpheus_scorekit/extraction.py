from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .models import Trajectory


@dataclass(frozen=True)
class TrackingConfig:
    hue_tolerance: int = 8
    saturation_floor: int = 55
    value_floor: int = 45
    patch_radius: int = 12
    min_area: int = 40
    max_area_ratio: float = 6.0
    morphology_size: int = 5
    max_jump_pixels: float = 180.0


def _read_frames(source: str | Path) -> tuple[list[np.ndarray], float | None]:
    source = Path(source)
    if source.is_dir():
        paths = sorted(
            [*source.glob("*.jpg"), *source.glob("*.jpeg"), *source.glob("*.png")],
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.name,
        )
        if not paths:
            raise ValueError(f"no image frames found in {source}")
        # ``cv2.imread`` is unreliable on some Windows builds when a parent path
        # contains non-ASCII characters.  imdecode + fromfile keeps Korean paths safe.
        frames = [cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR) for path in paths]
        if any(frame is None for frame in frames):
            raise ValueError(f"failed to read one or more frames from {source}")
        return frames, None

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or None
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"video contains no readable frames: {source}")
    return frames, fps


def _circular_hue_distance(hue: np.ndarray, target: float) -> np.ndarray:
    direct = np.abs(hue.astype(np.float32) - target)
    return np.minimum(direct, 180.0 - direct)


class SeededColorTracker:
    """Lightweight seeded tracker for controlled, static-camera experiments.

    This is a practical validation backend, not a replacement for paper-faithful SAM2.
    It learns an HSV colour model around a user-provided first-frame point and follows
    the best connected component using motion continuity and area consistency.
    """

    def __init__(self, config: TrackingConfig | None = None):
        self.config = config or TrackingConfig()

    def extract(
        self,
        source: str | Path,
        seed_points_xy: dict[int, tuple[float, float]],
        *,
        fps: float | None = None,
        experiment: str | None = None,
        overlay_dir: str | Path | None = None,
    ) -> Trajectory:
        frames, detected_fps = _read_frames(source)
        actual_fps = float(fps or detected_fps or 30.0)
        if not seed_points_xy:
            raise ValueError("at least one first-frame seed point is required")

        hsv0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)
        models = {
            object_id: self._learn_model(hsv0, point)
            for object_id, point in seed_points_xy.items()
        }
        tracks = {
            object_id: np.full((len(frames), 3), np.nan, dtype=np.float32)
            for object_id in seed_points_xy
        }
        states: dict[int, list[np.ndarray]] = {object_id: [] for object_id in seed_points_xy}

        overlay_path = Path(overlay_dir) if overlay_dir is not None else None
        if overlay_path is not None:
            overlay_path.mkdir(parents=True, exist_ok=True)

        for frame_index, frame in enumerate(frames):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            rendered = frame.copy()
            for object_id, model in models.items():
                center, area, mask = self._locate(hsv, model, states[object_id])
                if center is not None:
                    x, y = center
                    tracks[object_id][frame_index] = (y, x, 0.0)
                    states[object_id].append(np.array([x, y, area], dtype=float))
                    if len(states[object_id]) > 2:
                        states[object_id].pop(0)
                    cv2.circle(rendered, (int(round(x)), int(round(y))), 8, (0, 255, 0), 2)
                    cv2.putText(rendered, f"obj {object_id}", (int(x) + 10, int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                if overlay_path is not None and frame_index == 0:
                    cv2.imwrite(str(overlay_path / f"object_{object_id}_initial_mask.png"), mask)
            if overlay_path is not None:
                cv2.imwrite(str(overlay_path / f"{frame_index:05d}.jpg"), rendered)

        trajectory = Trajectory(
            fps=actual_fps,
            objects=tracks,
            experiment=experiment,
            metadata={
                "extractor": "seeded_hsv_connected_components",
                "source": str(Path(source)),
                "coordinate_order": "y_x_depth",
                "paper_faithful_tracker": False,
            },
        )
        validation = trajectory.validate()
        if not validation["valid"]:
            raise ValueError("invalid extracted trajectory: " + "; ".join(validation["errors"]))
        return trajectory

    def _learn_model(self, hsv: np.ndarray, point_xy: tuple[float, float]) -> dict[str, float]:
        x, y = [int(round(value)) for value in point_xy]
        radius = self.config.patch_radius
        y0, y1 = max(0, y - radius), min(hsv.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(hsv.shape[1], x + radius + 1)
        patch = hsv[y0:y1, x0:x1]
        if patch.size == 0:
            raise ValueError(f"seed point {point_xy} lies outside the first frame")
        saturated = patch[..., 1] >= np.percentile(patch[..., 1], 60)
        pixels = patch[saturated] if saturated.any() else patch.reshape(-1, 3)
        target_hue = float(np.median(pixels[:, 0]))
        saturation_min = max(self.config.saturation_floor, int(np.percentile(pixels[:, 1], 20) * 0.85))
        value_min = max(self.config.value_floor, int(np.percentile(pixels[:, 2], 10) * 0.55))
        return {
            "hue": target_hue,
            "saturation_min": float(saturation_min),
            "value_min": float(value_min),
            "initial_x": float(point_xy[0]),
            "initial_y": float(point_xy[1]),
            "reference_area": 0.0,
        }

    def _locate(
        self,
        hsv: np.ndarray,
        model: dict[str, float],
        state: list[np.ndarray],
    ) -> tuple[tuple[float, float] | None, float, np.ndarray]:
        hue_ok = _circular_hue_distance(hsv[..., 0], model["hue"]) <= self.config.hue_tolerance
        mask = (
            hue_ok
            & (hsv[..., 1] >= model["saturation_min"])
            & (hsv[..., 2] >= model["value_min"])
        ).astype(np.uint8) * 255
        kernel = np.ones((self.config.morphology_size, self.config.morphology_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if state:
            previous = state[-1]
            predicted = previous[:2]
            if len(state) == 2:
                predicted = state[-1][:2] + (state[-1][:2] - state[-2][:2])
        else:
            predicted = np.array([model["initial_x"], model["initial_y"]], dtype=float)

        candidates: list[tuple[float, float, np.ndarray]] = []
        for label_index in range(1, count):
            area = float(stats[label_index, cv2.CC_STAT_AREA])
            if area < self.config.min_area:
                continue
            if model["reference_area"] and area > model["reference_area"] * self.config.max_area_ratio:
                continue
            center = centroids[label_index]
            distance = float(np.linalg.norm(center - predicted))
            if state and distance > self.config.max_jump_pixels:
                continue
            area_penalty = 0.0
            if model["reference_area"]:
                area_penalty = abs(np.log((area + 1e-6) / model["reference_area"])) * 25.0
            candidates.append((distance + area_penalty, area, center))

        if not candidates:
            return None, 0.0, mask
        _, area, center = min(candidates, key=lambda item: item[0])
        if not model["reference_area"]:
            model["reference_area"] = area
        return (float(center[0]), float(center[1])), area, mask


class MorpheusSAM2Adapter:
    """Backward-compatible facade for the executable SAM2 + Depth Anything extractor."""

    def __init__(self, reference_repo: str | Path = "vendor/Morpheus"):
        self.reference_repo = Path(reference_repo).resolve()

    def extractor(self, **kwargs):
        """Create the executable extractor while keeping heavyweight imports lazy."""
        from .sam2_extraction import SAM2DepthConfig, SAM2DepthTrajectoryExtractor

        return SAM2DepthTrajectoryExtractor(
            SAM2DepthConfig(reference_repo=self.reference_repo, **kwargs)
        )

    def extract(self, source, positive_points_xy, **kwargs):
        """Run SAM2 extraction with default model settings."""
        return self.extractor().extract(source, positive_points_xy, **kwargs)

    def command_for_dataset_tree(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        experiment: str,
        *,
        device: str = "cpu",
        calculate_scores: bool = False,
    ) -> list[str]:
        command = [
            "morpheus-track-and-score",
            "--input-dir", str(Path(input_dir).resolve()),
            "--output-dir", str(Path(output_dir).resolve()),
            "--methods", "real-world",
            "--experiments", experiment,
            "--device", device,
        ]
        if calculate_scores:
            command.append("--calculate-scores")
        return command
