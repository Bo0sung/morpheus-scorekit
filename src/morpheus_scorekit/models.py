from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Trajectory:
    """Frame-aligned trajectories in the coordinate order used by Morpheus.

    Each object array has shape ``(frames, 3)`` and stores ``[row_y, col_x, depth]``.
    Missing observations are represented by NaN.  Depth may be all zero when a depth
    estimator is not used; depth is diagnostic and is not part of the aggregate score.
    """

    fps: float
    objects: dict[int, np.ndarray]
    experiment: str | None = None
    distances: dict[int, np.ndarray] | None = None
    angles: dict[int, dict[int, float]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(next(iter(self.objects.values()))) if self.objects else 0

    def validate(self, minimum_frames: int = 8) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        if not np.isfinite(self.fps) or self.fps <= 0:
            errors.append("fps must be a positive finite number")
        if not self.objects:
            errors.append("at least one object trajectory is required")
            return {"valid": False, "errors": errors, "warnings": warnings}

        lengths = {len(np.asarray(v)) for v in self.objects.values()}
        if len(lengths) != 1:
            errors.append("all object trajectories must have the same frame count")
        for object_id, values in self.objects.items():
            arr = np.asarray(values, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3:
                errors.append(f"object {object_id}: expected shape (frames, 3), got {arr.shape}")
                continue
            valid_rows = np.isfinite(arr[:, :2]).all(axis=1)
            if valid_rows.sum() < minimum_frames:
                errors.append(f"object {object_id}: fewer than {minimum_frames} valid positions")
            valid_fraction = float(valid_rows.mean()) if len(valid_rows) else 0.0
            if valid_fraction < 0.15:
                errors.append(f"object {object_id}: valid fraction {valid_fraction:.3f} < 0.15")
            elif valid_fraction < 0.8:
                warnings.append(f"object {object_id}: valid fraction is only {valid_fraction:.3f}")

        if self.distances is not None:
            for object_id, distance in self.distances.items():
                if len(distance) != self.frame_count:
                    errors.append(f"distance {object_id}: frame count mismatch")
        return {"valid": not errors, "errors": errors, "warnings": warnings}

    def quality_summary(self) -> dict[str, Any]:
        per_object: dict[str, Any] = {}
        for object_id, values in self.objects.items():
            arr = np.asarray(values, dtype=float)
            valid = np.isfinite(arr[:, :2]).all(axis=1)
            longest_gap = 0
            current_gap = 0
            for is_valid in valid:
                current_gap = 0 if is_valid else current_gap + 1
                longest_gap = max(longest_gap, current_gap)
            if valid.any():
                rows = arr[valid, :2]
                motion_range = np.ptp(rows, axis=0).tolist()
            else:
                motion_range = [0.0, 0.0]
            per_object[str(object_id)] = {
                "valid_frames": int(valid.sum()),
                "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
                "longest_missing_gap": int(longest_gap),
                "motion_range_yx": motion_range,
            }
        return {"frame_count": self.frame_count, "fps": self.fps, "objects": per_object}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "fps": np.array(self.fps),
            "experiment": np.array(self.experiment or ""),
            "metadata_json": np.array(json.dumps(self.metadata, ensure_ascii=False)),
            "object_ids": np.array(sorted(self.objects), dtype=int),
        }
        for object_id, values in self.objects.items():
            payload[f"object_{object_id}"] = np.asarray(values, dtype=np.float32)
        if self.distances is not None:
            payload["distance_ids"] = np.array(sorted(self.distances), dtype=int)
            for object_id, values in self.distances.items():
                payload[f"distance_{object_id}"] = np.asarray(values, dtype=np.float32)
        if self.angles is not None:
            payload["angles_json"] = np.array(json.dumps(self.angles))
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        with np.load(path, allow_pickle=False) as data:
            object_ids = [int(v) for v in data["object_ids"]]
            objects = {object_id: data[f"object_{object_id}"].copy() for object_id in object_ids}
            distances = None
            if "distance_ids" in data:
                distance_ids = [int(v) for v in data["distance_ids"]]
                distances = {object_id: data[f"distance_{object_id}"].copy() for object_id in distance_ids}
            angles = json.loads(str(data["angles_json"])) if "angles_json" in data else None
            metadata = json.loads(str(data["metadata_json"]))
            experiment = str(data["experiment"]) or None
            return cls(
                fps=float(data["fps"]),
                objects=objects,
                experiment=experiment,
                distances=distances,
                angles=angles,
                metadata=metadata,
            )

