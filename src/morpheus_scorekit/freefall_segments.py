from __future__ import annotations

from typing import Any

import numpy as np

from .models import Trajectory


def estimate_first_contact_frame(
    trajectory: Trajectory,
    *,
    object_id: int = 1,
    minimum_freefall_frames: int = 8,
    smoothing_window: int = 5,
) -> int:
    """Estimate first floor contact from the ball's first image-plane low point.

    Image rows increase downwards. For a static camera, the first maximum row is the
    impact point before a bounce. A short moving average suppresses tracking noise.
    The returned frame is exclusive, so the contact frame itself is omitted.
    """

    if object_id not in trajectory.objects:
        raise ValueError(f"object {object_id} is not present")
    rows = np.asarray(trajectory.objects[object_id], dtype=float)[:, 0]
    if len(rows) <= minimum_freefall_frames:
        raise ValueError("trajectory is too short to isolate pre-contact free fall")
    valid = np.isfinite(rows)
    if valid.sum() < minimum_freefall_frames + 1:
        raise ValueError("not enough valid row coordinates")

    indices = np.arange(len(rows))
    filled = np.interp(indices, indices[valid], rows[valid])
    window = max(1, min(int(smoothing_window), len(filled)))
    if window % 2 == 0:
        window -= 1
    if window > 1:
        radius = window // 2
        padded = np.pad(filled, (radius, radius), mode="edge")
        smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
    else:
        smoothed = filled

    contact = minimum_freefall_frames + int(np.argmax(smoothed[minimum_freefall_frames:]))
    if contact < minimum_freefall_frames:
        raise ValueError("estimated contact leaves too few free-fall frames")
    return contact


def trim_precontact(
    trajectory: Trajectory,
    *,
    object_id: int = 1,
    minimum_freefall_frames: int = 8,
) -> tuple[Trajectory, dict[str, Any]]:
    contact = estimate_first_contact_frame(
        trajectory,
        object_id=object_id,
        minimum_freefall_frames=minimum_freefall_frames,
    )
    objects = {key: np.asarray(value)[:contact].copy() for key, value in trajectory.objects.items()}
    distances = None
    if trajectory.distances is not None:
        distances = {
            key: np.asarray(value)[:contact].copy() for key, value in trajectory.distances.items()
        }
    metadata = dict(trajectory.metadata)
    metadata["freefall_trim"] = {
        "method": "first smoothed maximum image row",
        "object_id": object_id,
        "contact_frame_exclusive": contact,
        "assumption": "static camera and predominantly downward image motion",
    }
    trimmed = Trajectory(
        fps=trajectory.fps,
        objects=objects,
        experiment=trajectory.experiment or "falling_ball",
        distances=distances,
        angles=trajectory.angles,
        metadata=metadata,
    )
    return trimmed, metadata["freefall_trim"]
