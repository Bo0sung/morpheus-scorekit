from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .freefall_diagnostics import load_freefall_state, scan_freefall_acceleration


def dose_tag(dose: float) -> str:
    sign = "m" if dose < 0 else ""
    return f"{sign}{abs(int(round(dose * 1000))):04d}"


def build_sweep_manifest(
    kubric_repo: str | Path,
    *,
    count: int,
    start_seed: int,
    doses: Iterable[float],
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be positive")
    if start_frame < 1 or end_frame < start_frame:
        raise ValueError("invalid intervention frame range")
    repo = Path(kubric_repo).resolve()
    dose_values = [float(value) for value in doses]
    if not dose_values:
        raise ValueError("at least one dose is required")
    pairs = []
    for seed in range(start_seed, start_seed + count):
        for dose in dose_values:
            scene_id = f"gravity_s{seed:06d}_g{dose_tag(dose)}"
            pairs.append(
                {
                    "scene_id": scene_id,
                    "seed": seed,
                    "dose": dose,
                    "pair_dir": str((repo / "outputs" / scene_id).resolve()),
                }
            )
    return {
        "schema_version": 1,
        "kubric_repo": str(repo),
        "intervention": "gravity_scale",
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "doses": dose_values,
        "seed_count": int(count),
        "start_seed": int(start_seed),
        "pairs": pairs,
    }


def calibrate_threshold(
    normal_decision_values: Iterable[float],
    *,
    minimum_threshold: float = 0.02,
) -> dict[str, float]:
    values = np.asarray(list(normal_decision_values), dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("finite normal decision values are required")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    quantile_99 = float(np.quantile(values, 0.99))
    threshold = max(float(minimum_threshold), quantile_99 + 6.0 * robust_sigma)
    return {
        "threshold": threshold,
        "normal_median": median,
        "normal_mad": mad,
        "normal_quantile_99": quantile_99,
        "minimum_threshold": float(minimum_threshold),
    }


def _interval_iou(detected: dict[str, int] | None, expected: tuple[int, int]) -> float:
    if detected is None:
        return 0.0
    left = max(int(detected["start_frame"]), expected[0])
    right = min(int(detected["end_frame_inclusive"]), expected[1])
    intersection = max(0, right - left + 1)
    union_left = min(int(detected["start_frame"]), expected[0])
    union_right = max(int(detected["end_frame_inclusive"]), expected[1])
    return float(intersection / (union_right - union_left + 1))


def _compact_row(
    pair: dict[str, Any],
    variant: str,
    result: dict[str, Any],
    *,
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    dose = float(pair["dose"])
    expected_label = "normal" if variant == "normal" or np.isclose(dose, 1.0) else "violation"
    expected_residual = 0.0 if variant == "normal" else abs(1.0 - dose)
    detected = result["detected_range"]
    localization_iou = None
    if variant == "violation" and expected_label == "violation":
        localization_iou = _interval_iou(detected, (start_frame, end_frame))
    peak = result["peak_window"]
    return {
        "scene_id": pair["scene_id"],
        "seed": int(pair["seed"]),
        "dose": dose,
        "variant": variant,
        "expected_label": expected_label,
        "predicted_label": result["classification"],
        "correct": result["classification"] == expected_label,
        "decision_value": float(result["decision_value"]),
        "freefall_physics_score": float(result["freefall_physics_score"]),
        "freefall_violation_score": float(result["freefall_violation_score"]),
        "expected_normalized_residual": expected_residual,
        "residual_absolute_error": abs(float(result["decision_value"]) - expected_residual),
        "max_normalized_gravity_residual": float(result["max_normalized_gravity_residual"]),
        "peak_acceleration_m_s2": float(peak["estimated_acceleration_m_s2"]),
        "peak_start_frame": int(peak["start_frame"]),
        "peak_end_frame_inclusive": int(peak["end_frame_inclusive"]),
        "detected_start_frame": None if detected is None else int(detected["start_frame"]),
        "detected_end_frame_inclusive": (
            None if detected is None else int(detected["end_frame_inclusive"])
        ),
        "localization_iou": localization_iou,
        "state_path": str((Path(pair["pair_dir"]) / variant / "state.npz").resolve()),
    }


def evaluate_sweep(
    manifest: dict[str, Any],
    *,
    expected_gravity: float = -9.81,
    window_size: int = 5,
    threshold: float | None = None,
    minimum_auto_threshold: float = 0.02,
) -> dict[str, Any]:
    pairs = list(manifest["pairs"])
    if not pairs:
        raise ValueError("manifest contains no pairs")

    preliminary_normal = []
    for pair in pairs:
        state_path = Path(pair["pair_dir"]) / "normal" / "state.npz"
        result = scan_freefall_acceleration(
            load_freefall_state(state_path),
            expected_gravity=expected_gravity,
            window_size=window_size,
            residual_threshold=1.0,
        )
        preliminary_normal.append(float(result["decision_value"]))

    calibration = calibrate_threshold(
        preliminary_normal, minimum_threshold=minimum_auto_threshold
    )
    if threshold is not None:
        calibration["threshold"] = float(threshold)
        calibration["mode"] = "fixed"
    else:
        calibration["mode"] = "normal-data robust calibration"
    selected_threshold = float(calibration["threshold"])

    rows = []
    start_frame = int(manifest["start_frame"])
    end_frame = int(manifest["end_frame"])
    for pair in pairs:
        for variant in ("normal", "violation"):
            state_path = Path(pair["pair_dir"]) / variant / "state.npz"
            result = scan_freefall_acceleration(
                load_freefall_state(state_path),
                expected_gravity=expected_gravity,
                window_size=window_size,
                residual_threshold=selected_threshold,
            )
            rows.append(
                _compact_row(
                    pair,
                    variant,
                    result,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )

    tp = sum(row["expected_label"] == "violation" and row["predicted_label"] == "violation" for row in rows)
    tn = sum(row["expected_label"] == "normal" and row["predicted_label"] == "normal" for row in rows)
    fp = sum(row["expected_label"] == "normal" and row["predicted_label"] == "violation" for row in rows)
    fn = sum(row["expected_label"] == "violation" and row["predicted_label"] == "normal" for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rows)
    normal_count = tn + fp

    violation_rows = [row for row in rows if row["variant"] == "violation"]
    by_dose: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in violation_rows:
        by_dose[float(row["dose"])].append(row)
    dose_summary = []
    for dose in sorted(by_dose, reverse=True):
        group = by_dose[dose]
        decisions = np.asarray([row["decision_value"] for row in group])
        scores = np.asarray([row["freefall_physics_score"] for row in group])
        accelerations = np.asarray([row["peak_acceleration_m_s2"] for row in group])
        ious = [row["localization_iou"] for row in group if row["localization_iou"] is not None]
        dose_summary.append(
            {
                "dose": dose,
                "expected_residual": abs(1.0 - dose),
                "count": len(group),
                "decision_mean": float(np.mean(decisions)),
                "decision_std": float(np.std(decisions)),
                "physics_score_mean": float(np.mean(scores)),
                "peak_acceleration_mean_m_s2": float(np.mean(accelerations)),
                "detection_rate": float(np.mean([row["predicted_label"] == "violation" for row in group])),
                "localization_iou_mean": float(np.mean(ious)) if ious else None,
            }
        )

    expected = np.asarray([row["expected_normalized_residual"] for row in violation_rows])
    observed = np.asarray([row["decision_value"] for row in violation_rows])
    correlation = float(np.corrcoef(expected, observed)[0, 1]) if np.std(expected) and np.std(observed) else None
    groups_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in violation_rows:
        groups_by_seed[int(row["seed"])].append(row)
    monotonic = []
    for group in groups_by_seed.values():
        ordered = sorted(group, key=lambda row: row["expected_normalized_residual"])
        values = [row["decision_value"] for row in ordered]
        monotonic.append(all(right + 1e-8 >= left for left, right in zip(values, values[1:])))

    return {
        "schema_version": 1,
        "method": {
            "expected_gravity_m_s2": expected_gravity,
            "window_size_frames": window_size,
            "classification_input": "world-space position and time only",
            "known_intervention_window_used_for_classification": False,
        },
        "calibration": calibration,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fp / normal_count if normal_count else 0.0,
            "dose_response_correlation": correlation,
            "monotonic_seed_fraction": float(np.mean(monotonic)) if monotonic else None,
            "mean_residual_absolute_error": float(
                np.mean([row["residual_absolute_error"] for row in rows])
            ),
        },
        "dose_summary": dose_summary,
        "rows": rows,
    }


def render_sweep_markdown(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    calibration = result["calibration"]
    confusion = result["confusion"]
    correlation = metrics["dose_response_correlation"]
    monotonic = metrics["monotonic_seed_fraction"]
    lines = [
        "# Free-fall gravity dose sweep report",
        "",
        "개입 구간이나 저장된 가속도를 입력하지 않고 world-space 위치와 실제 시간만으로 평가했습니다.",
        "각 sliding window에 2차 궤적을 fitting하여 가속도와 중력 residual을 추정했습니다.",
        "",
        "## 분류 결과",
        "",
        f"- Threshold: `{calibration['threshold']:.6f}` ({calibration['mode']})",
        f"- Accuracy: `{metrics['accuracy']:.4f}`",
        f"- Precision / Recall / F1: `{metrics['precision']:.4f}` / `{metrics['recall']:.4f}` / `{metrics['f1']:.4f}`",
        f"- False positive rate: `{metrics['false_positive_rate']:.4f}`",
        f"- Confusion matrix: TP={confusion['tp']}, TN={confusion['tn']}, FP={confusion['fp']}, FN={confusion['fn']}",
        f"- Dose-response correlation: `{correlation:.4f}`" if correlation is not None else "- Dose-response correlation: unavailable",
        f"- Seed별 단조성 만족 비율: `{monotonic:.4f}`" if monotonic is not None else "- Seed별 단조성: unavailable",
        "",
        "## 중력 배율별 결과",
        "",
        "| gravity scale | expected residual | decision mean ± std | physics score | peak acceleration (m/s²) | detection rate | localization IoU |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["dose_summary"]:
        iou = "-" if row["localization_iou_mean"] is None else f"{row['localization_iou_mean']:.4f}"
        lines.append(
            f"| {row['dose']:.3f} | {row['expected_residual']:.3f} | "
            f"{row['decision_mean']:.6f} ± {row['decision_std']:.6f} | "
            f"{row['physics_score_mean']:.6f} | "
            f"{row['peak_acceleration_mean_m_s2']:.6f} | "
            f"{row['detection_rate']:.4f} | {iou} |"
        )
    lines.extend(
        [
            "",
            "## 해석 주의사항",
            "",
            "- `decision_value`는 높을수록 위반이며 확률이 아닙니다.",
            "- `physics_score = exp(-decision_value)`이며 높을수록 정상 중력과 일치합니다.",
            "- 자동 threshold는 정상 GT 데이터의 residual 분포로 보정한 값입니다.",
            "- 현재 결과는 Kubric world-space GT 기준입니다. RGB-only 성능으로 해석하면 안 됩니다.",
            "- 논문용 최종 평가에서는 calibration seed와 test seed를 분리해야 합니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
