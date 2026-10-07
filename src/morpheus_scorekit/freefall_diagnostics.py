from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


EARTH_GRAVITY = -9.81


def _first_contact_exclusive(contact: np.ndarray) -> int:
    indices = np.flatnonzero(np.asarray(contact, dtype=bool))
    return int(indices[0]) if len(indices) else len(contact)


def load_freefall_state(path: str | Path) -> dict[str, np.ndarray]:
    """Load the world-space arrays written by the Kubric free-fall worker."""

    required = {"frame", "time", "position", "contact_ball_floor"}
    with np.load(path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(sorted(missing))}")
        state = {key: data[key].copy() for key in data.files}

    frame_count = len(state["frame"])
    if frame_count < 3:
        raise ValueError("free-fall state needs at least three frames")
    for key, value in state.items():
        if len(value) != frame_count:
            raise ValueError(f"{key} has {len(value)} frames; expected {frame_count}")
    if state["position"].shape != (frame_count, 3):
        raise ValueError(f"position must have shape ({frame_count}, 3)")
    if np.any(np.diff(np.asarray(state["time"], dtype=float)) <= 0):
        raise ValueError("time must be strictly increasing")
    return state


def _vertical_acceleration(state: dict[str, np.ndarray]) -> np.ndarray:
    if "acceleration" in state:
        acceleration = np.asarray(state["acceleration"], dtype=float)
        if acceleration.shape != (len(state["frame"]), 3):
            raise ValueError("acceleration must have shape (frames, 3)")
        return acceleration[:, 2]

    time = np.asarray(state["time"], dtype=float)
    if "velocity" in state:
        velocity = np.asarray(state["velocity"], dtype=float)
        if velocity.shape != (len(time), 3):
            raise ValueError("velocity must have shape (frames, 3)")
        return np.gradient(velocity[:, 2], time)
    position = np.asarray(state["position"], dtype=float)
    return np.gradient(np.gradient(position[:, 2], time), time)


def _quadratic_gravity(time: np.ndarray, height: np.ndarray) -> tuple[float, float]:
    relative_time = time - time[0]
    design = np.column_stack(
        [np.ones(len(relative_time)), relative_time, 0.5 * relative_time**2]
    )
    coefficients, *_ = np.linalg.lstsq(design, height, rcond=None)
    predicted = design @ coefficients
    rmse = float(np.sqrt(np.mean((height - predicted) ** 2)))
    travel = max(float(np.ptp(height)), 1e-12)
    return float(coefficients[2]), rmse / travel


def scan_freefall_acceleration(
    state: dict[str, np.ndarray],
    *,
    expected_gravity: float = EARTH_GRAVITY,
    window_size: int = 5,
    residual_threshold: float = 0.15,
    top_fraction: float = 0.2,
    end_frame_exclusive: int | None = None,
) -> dict[str, Any]:
    """Automatically find anomalous acceleration without a known event window.

    Acceleration is estimated from position, not read from the state's acceleration
    array.  A quadratic ``z(t)`` is fitted in every overlapping window and its second
    derivative is compared with the expected gravity in metric units.
    """

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer of at least 3")
    if residual_threshold <= 0:
        raise ValueError("residual_threshold must be positive")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if expected_gravity == 0:
        raise ValueError("expected_gravity must be non-zero")

    frames = np.asarray(state["frame"], dtype=int)
    time = np.asarray(state["time"], dtype=float)
    height = np.asarray(state["position"], dtype=float)[:, 2]
    contact_end = _first_contact_exclusive(state["contact_ball_floor"])
    end = contact_end if end_frame_exclusive is None else min(contact_end, end_frame_exclusive)
    if end < window_size:
        raise ValueError(f"fewer than {window_size} pre-contact frames remain")

    windows: list[dict[str, Any]] = []
    for start in range(0, end - window_size + 1):
        stop = start + window_size
        segment_time = time[start:stop]
        segment_height = height[start:stop]
        if not np.isfinite(segment_time).all() or not np.isfinite(segment_height).all():
            continue
        fitted_acceleration, fit_nrmse = _quadratic_gravity(segment_time, segment_height)
        residual = abs(fitted_acceleration - expected_gravity) / abs(expected_gravity)
        windows.append(
            {
                "start_frame": int(frames[start]),
                "end_frame_inclusive": int(frames[stop - 1]),
                "center_frame": int(frames[start + window_size // 2]),
                "estimated_acceleration_m_s2": fitted_acceleration,
                "normalized_gravity_residual": float(residual),
                "quadratic_fit_nrmse": fit_nrmse,
            }
        )
    if not windows:
        raise ValueError("no finite sliding windows are available")

    residuals = np.asarray([window["normalized_gravity_residual"] for window in windows])
    top_count = max(1, int(np.ceil(len(residuals) * top_fraction)))
    top_indices = np.argsort(residuals)[-top_count:]
    top_mean = float(np.mean(residuals[top_indices]))
    peak_index = int(np.argmax(residuals))
    abnormal_indices = np.flatnonzero(residuals >= residual_threshold)

    detected_range = None
    if len(abnormal_indices):
        detected_range = {
            "start_frame": int(windows[int(abnormal_indices[0])]["start_frame"]),
            "end_frame_inclusive": int(
                windows[int(abnormal_indices[-1])]["end_frame_inclusive"]
            ),
        }

    return {
        "classification": "violation" if top_mean >= residual_threshold else "normal",
        "decision_value": top_mean,
        "decision_rule": "top-window mean normalized gravity residual >= threshold",
        "residual_threshold": float(residual_threshold),
        "expected_gravity_m_s2": float(expected_gravity),
        "window_size_frames": int(window_size),
        "top_fraction": float(top_fraction),
        "window_count": len(windows),
        "max_normalized_gravity_residual": float(residuals[peak_index]),
        "mean_normalized_gravity_residual": float(np.mean(residuals)),
        "abnormal_window_fraction": float(len(abnormal_indices) / len(windows)),
        "peak_window": windows[peak_index],
        "detected_range": detected_range,
        "windows": windows,
        "notes": [
            "No intervention frames or acceleration array were used.",
            "The default threshold is provisional and must be calibrated on normal validation data.",
        ],
    }


def analyze_freefall_state(
    state: dict[str, np.ndarray],
    *,
    expected_gravity: float = EARTH_GRAVITY,
    end_frame_exclusive: int | None = None,
    event_window: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Measure gravity in actual seconds and world-space metres.

    The score is deliberately labelled as a diagnostic rather than a replacement for
    the Morpheus score.  It is one for exact agreement with the expected gravity and
    decays exponentially with normalized mean absolute acceleration error.
    """

    frames = np.asarray(state["frame"], dtype=int)
    time = np.asarray(state["time"], dtype=float)
    height = np.asarray(state["position"], dtype=float)[:, 2]
    acceleration = _vertical_acceleration(state)
    contact_end = _first_contact_exclusive(state["contact_ball_floor"])
    end = contact_end if end_frame_exclusive is None else min(contact_end, end_frame_exclusive)
    if end < 3:
        raise ValueError("fewer than three pre-contact frames remain")

    base_mask = np.arange(len(frames)) < end
    base_mask &= np.isfinite(time) & np.isfinite(height) & np.isfinite(acceleration)
    if "acceleration_valid" in state:
        acceleration_mask = base_mask & np.asarray(state["acceleration_valid"], dtype=bool)
    else:
        acceleration_mask = base_mask

    if base_mask.sum() < 3 or acceleration_mask.sum() < 1:
        raise ValueError("not enough valid pre-contact observations")

    fitted_g, fit_nrmse = _quadratic_gravity(time[base_mask], height[base_mask])
    accel_error = np.abs(acceleration[acceleration_mask] - expected_gravity)
    normalized_mae = float(np.mean(accel_error) / abs(expected_gravity))
    result: dict[str, Any] = {
        "frame_count": int(base_mask.sum()),
        "start_frame": int(frames[base_mask][0]),
        "end_frame_exclusive": int(end),
        "duration_seconds": float(time[base_mask][-1] - time[base_mask][0]),
        "vertical_travel_metres": float(height[base_mask][0] - height[base_mask][-1]),
        "fitted_gravity_m_s2": fitted_g,
        "quadratic_fit_nrmse": fit_nrmse,
        "mean_vertical_acceleration_m_s2": float(np.mean(acceleration[acceleration_mask])),
        "gravity_mae_m_s2": float(np.mean(accel_error)),
        "gravity_agreement_diagnostic": float(np.exp(-normalized_mae)),
        "valid_acceleration_frames": int(acceleration_mask.sum()),
    }

    if event_window is not None:
        start_frame, end_frame = event_window
        event_mask = acceleration_mask & (frames >= start_frame) & (frames <= end_frame)
        if event_mask.any():
            event_error = np.abs(acceleration[event_mask] - expected_gravity)
            event_normalized_mae = float(np.mean(event_error) / abs(expected_gravity))
            result["event_window"] = {
                "start_frame": int(start_frame),
                "end_frame_inclusive": int(end_frame),
                "valid_acceleration_frames": int(event_mask.sum()),
                "mean_vertical_acceleration_m_s2": float(np.mean(acceleration[event_mask])),
                "gravity_mae_m_s2": float(np.mean(event_error)),
                "gravity_agreement_diagnostic": float(np.exp(-event_normalized_mae)),
            }
        else:
            result["event_window"] = {
                "start_frame": int(start_frame),
                "end_frame_inclusive": int(end_frame),
                "valid_acceleration_frames": 0,
                "available": False,
            }
    return result


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _score_summary(score: dict[str, Any] | None) -> dict[str, float] | None:
    if score is None:
        return None
    summary = {
        key: float(score[key])
        for key in ("physical_score", "statistical_score", "total_score")
        if key in score
    }
    individual = score.get("individual_physical_scores", {})
    for key in (
        "energy_conservation",
        "acceleration_conservation",
        "horizontal_momentum_conservation",
    ):
        if key in individual:
            summary[key] = float(individual[key])
    return summary


def _numeric_delta(normal: dict[str, Any], violation: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(violation[key]) - float(normal[key])
        for key in normal.keys() & violation.keys()
        if isinstance(normal[key], (int, float)) and isinstance(violation[key], (int, float))
    }


def compare_freefall_pair(
    pair_dir: str | Path,
    *,
    score_root: str | Path | None = None,
    expected_gravity: float = EARTH_GRAVITY,
    classifier_window_size: int = 5,
    classifier_threshold: float = 0.15,
) -> dict[str, Any]:
    pair_dir = Path(pair_dir)
    normal_state = load_freefall_state(pair_dir / "normal" / "state.npz")
    violation_state = load_freefall_state(pair_dir / "violation" / "state.npz")
    violation_metadata = _read_json(pair_dir / "violation" / "metadata.json") or {}
    intervention = violation_metadata.get("intervention", {})
    event_window = None
    if intervention.get("start_frame") is not None and intervention.get("end_frame") is not None:
        event_window = (int(intervention["start_frame"]), int(intervention["end_frame"]))

    normal_contact = _first_contact_exclusive(normal_state["contact_ball_floor"])
    violation_contact = _first_contact_exclusive(violation_state["contact_ball_floor"])
    common_end = min(normal_contact, violation_contact)
    normal_full = analyze_freefall_state(
        normal_state, expected_gravity=expected_gravity, event_window=event_window
    )
    violation_full = analyze_freefall_state(
        violation_state, expected_gravity=expected_gravity, event_window=event_window
    )
    normal_common = analyze_freefall_state(
        normal_state,
        expected_gravity=expected_gravity,
        end_frame_exclusive=common_end,
        event_window=event_window,
    )
    violation_common = analyze_freefall_state(
        violation_state,
        expected_gravity=expected_gravity,
        end_frame_exclusive=common_end,
        event_window=event_window,
    )
    normal_classifier = scan_freefall_acceleration(
        normal_state,
        expected_gravity=expected_gravity,
        window_size=classifier_window_size,
        residual_threshold=classifier_threshold,
        end_frame_exclusive=common_end,
    )
    violation_classifier = scan_freefall_acceleration(
        violation_state,
        expected_gravity=expected_gravity,
        window_size=classifier_window_size,
        residual_threshold=classifier_threshold,
        end_frame_exclusive=common_end,
    )

    score_root = Path(score_root) if score_root is not None else pair_dir / "morpheus"
    normal_score = _score_summary(_read_json(score_root / "normal_gt" / "combined_scores.json"))
    violation_score = _score_summary(
        _read_json(score_root / "violation_gt" / "combined_scores.json")
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "pair_dir": str(pair_dir.resolve()),
        "method": {
            "time_axis": "state.npz time in seconds",
            "position_axis": "state.npz world-space z in metres",
            "comparison_horizon": "same pre-contact frame range for both variants",
            "expected_gravity_m_s2": expected_gravity,
            "diagnostic_formula": "exp(-mean(abs(a_z - g)) / abs(g))",
        },
        "intervention": intervention,
        "contact_frame_exclusive": {
            "normal": normal_contact,
            "violation": violation_contact,
            "common": common_end,
        },
        "world_space_full_pre_contact": {
            "normal": normal_full,
            "violation": violation_full,
            "violation_minus_normal": _numeric_delta(normal_full, violation_full),
        },
        "world_space_common_horizon": {
            "normal": normal_common,
            "violation": violation_common,
            "violation_minus_normal": _numeric_delta(normal_common, violation_common),
        },
        "automatic_acceleration_classifier": {
            "uses_known_intervention_window": False,
            "uses_saved_acceleration_array": False,
            "normal": normal_classifier,
            "violation": violation_classifier,
        },
        "original_morpheus": {
            "available": normal_score is not None and violation_score is not None,
            "normal": normal_score,
            "violation": violation_score,
            "violation_minus_normal": (
                _numeric_delta(normal_score, violation_score)
                if normal_score is not None and violation_score is not None
                else None
            ),
        },
    }
    return result


def render_pair_markdown(result: dict[str, Any]) -> str:
    common = result["world_space_common_horizon"]
    normal = common["normal"]
    violation = common["violation"]
    lines = [
        "# Free-fall pair diagnostic",
        "",
        "이 보고서는 원본 Morpheus 점수를 대체하지 않고, 점수 차이가 희석되는 원인을 분리합니다.",
        "실제 시간(초), 월드 z 좌표(미터), 두 영상의 공통 pre-contact 프레임을 사용했습니다.",
        "",
        "## 같은 시간 구간의 world-space 결과",
        "",
        "| 지표 | normal | violation | violation - normal |",
        "|---|---:|---:|---:|",
    ]
    metric_labels = (
        ("gravity_agreement_diagnostic", "중력 일치 진단값"),
        ("mean_vertical_acceleration_m_s2", "평균 수직가속도 (m/s²)"),
        ("fitted_gravity_m_s2", "궤적 적합 중력 (m/s²)"),
        ("quadratic_fit_nrmse", "포물선 적합 NRMSE"),
        ("duration_seconds", "비교 구간 (s)"),
    )
    for key, label in metric_labels:
        delta = float(violation[key]) - float(normal[key])
        lines.append(f"| {label} | {normal[key]:.6f} | {violation[key]:.6f} | {delta:+.6f} |")

    classifier = result["automatic_acceleration_classifier"]
    normal_classifier = classifier["normal"]
    violation_classifier = classifier["violation"]
    lines.extend(
        [
            "",
            "## 자동 가속도 classifier",
            "",
            "개입 프레임과 저장된 가속도를 사용하지 않고, 각 구간의 world-space 위치를 "
            "포물선 fitting하여 가속도를 추정했습니다.",
            "",
            "| 지표 | normal | violation |",
            "|---|---:|---:|",
            f"| 판정 | {normal_classifier['classification']} | "
            f"{violation_classifier['classification']} |",
            f"| 판정값 | {normal_classifier['decision_value']:.6f} | "
            f"{violation_classifier['decision_value']:.6f} |",
            f"| 최대 중력 residual | "
            f"{normal_classifier['max_normalized_gravity_residual']:.6f} | "
            f"{violation_classifier['max_normalized_gravity_residual']:.6f} |",
            f"| peak 추정 가속도 (m/s²) | "
            f"{normal_classifier['peak_window']['estimated_acceleration_m_s2']:.6f} | "
            f"{violation_classifier['peak_window']['estimated_acceleration_m_s2']:.6f} |",
            f"| peak window | "
            f"{normal_classifier['peak_window']['start_frame']}~"
            f"{normal_classifier['peak_window']['end_frame_inclusive']} | "
            f"{violation_classifier['peak_window']['start_frame']}~"
            f"{violation_classifier['peak_window']['end_frame_inclusive']} |",
        ]
    )

    if "event_window" in normal and "event_window" in violation:
        normal_event = normal["event_window"]
        violation_event = violation["event_window"]
        if normal_event.get("valid_acceleration_frames", 0) and violation_event.get(
            "valid_acceleration_frames", 0
        ):
            lines.extend(
                [
                    "",
                    "## 개입 프레임만 본 결과",
                    "",
                    "| 지표 | normal | violation | violation - normal |",
                    "|---|---:|---:|---:|",
                ]
            )
            for key, label in (
                ("gravity_agreement_diagnostic", "중력 일치 진단값"),
                ("mean_vertical_acceleration_m_s2", "평균 수직가속도 (m/s²)"),
                ("gravity_mae_m_s2", "중력 MAE (m/s²)"),
            ):
                delta = float(violation_event[key]) - float(normal_event[key])
                lines.append(
                    f"| {label} | {normal_event[key]:.6f} | "
                    f"{violation_event[key]:.6f} | {delta:+.6f} |"
                )

    original = result["original_morpheus"]
    if original["available"]:
        lines.extend(
            [
                "",
                "## 원본 Morpheus 결과",
                "",
                "| 지표 | normal | violation | violation - normal |",
                "|---|---:|---:|---:|",
            ]
        )
        for key in (
            "energy_conservation",
            "acceleration_conservation",
            "horizontal_momentum_conservation",
            "physical_score",
            "statistical_score",
            "total_score",
        ):
            if key in original["normal"] and key in original["violation"]:
                lines.append(
                    f"| {key} | {original['normal'][key]:.6f} | "
                    f"{original['violation'][key]:.6f} | "
                    f"{original['violation_minus_normal'][key]:+.6f} |"
                )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- `gravity_agreement_diagnostic`은 `exp(-MAE/|g|)`이며 이 실험용 보조 지표입니다.",
            "- 자동 classifier는 개입 구간을 입력받지 않고 위치 궤적에서 가속도를 직접 추정합니다.",
            "- classifier의 기본 임계값은 임시값이며 정상 validation 데이터로 보정해야 합니다.",
            "- 공통 구간은 낙하시간 차이를 지우지 않도록 초 단위 시간을 그대로 유지합니다.",
            "- 개입 구간 결과가 전체 구간보다 크게 벌어지면, 전체 평균이 위반 신호를 희석했다는 증거입니다.",
            "- 최종 보고에서는 원본 Morpheus와 이 진단값을 함께 제시해야 합니다.",
        ]
    )
    return "\n".join(lines) + "\n"
