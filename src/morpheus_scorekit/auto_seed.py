from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .extraction import _read_frames


@dataclass(frozen=True)
class AutoSeedResult:
    x: float
    y: float
    area: int
    confidence: float


def suggest_moving_object_seed(
    source: str | Path,
    *,
    preview_path: str | Path | None = None,
) -> AutoSeedResult:
    """Find a first-frame point on the dominant moving object.

    The method is intentionally limited to static-camera, single-object scenes such
    as PhyCo ``ball_drop_v2``.  It compares the first frame with a temporal median,
    then selects the strongest reasonably sized changed component.
    """

    frames, _ = _read_frames(source)
    if len(frames) < 3:
        raise ValueError("auto seed needs at least three frames")

    sample_indices = np.linspace(1, len(frames) - 1, min(17, len(frames) - 1), dtype=int)
    sampled = np.stack([frames[index] for index in sample_indices], axis=0)
    temporal_median = np.median(sampled, axis=0).astype(np.uint8)
    difference = cv2.absdiff(frames[0], temporal_median)
    difference_gray = np.max(difference, axis=2).astype(np.uint8)
    difference_gray = cv2.GaussianBlur(difference_gray, (5, 5), 0)

    otsu_threshold, mask = cv2.threshold(
        difference_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(12.0, float(otsu_threshold))
    _, mask = cv2.threshold(difference_gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    frame_area = frames[0].shape[0] * frames[0].shape[1]
    min_area = max(20, int(frame_area * 0.00005))
    max_area = int(frame_area * 0.20)
    candidates: list[tuple[float, int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not min_area <= area <= max_area:
            continue
        component = labels == label
        contrast = float(difference_gray[component].mean())
        score = contrast * np.sqrt(area)
        candidates.append((score, label, area))

    if not candidates:
        raise ValueError(
            "could not auto-detect a moving object; provide a manual first-frame --seed OBJECT_ID X Y"
        )

    score, selected_label, area = max(candidates, key=lambda item: item[0])
    selected = labels == selected_label
    ys, xs = np.nonzero(selected)
    weights = difference_gray[selected].astype(np.float64) + 1.0
    x = float(np.average(xs, weights=weights))
    y = float(np.average(ys, weights=weights))
    confidence = float(score / (score + 500.0))

    if preview_path is not None:
        preview = frames[0].copy()
        contours, _ = cv2.findContours(
            selected.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(preview, contours, -1, (0, 255, 0), 2)
        cv2.circle(preview, (int(round(x)), int(round(y))), 7, (0, 0, 255), -1)
        cv2.putText(
            preview,
            f"auto seed ({x:.1f}, {y:.1f})",
            (max(0, int(x) + 10), max(25, int(y))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
        destination = Path(preview_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(destination.suffix or ".jpg", preview)
        if not ok:
            raise RuntimeError(f"failed to encode auto-seed preview: {destination}")
        encoded.tofile(destination)

    return AutoSeedResult(x=x, y=y, area=area, confidence=confidence)
