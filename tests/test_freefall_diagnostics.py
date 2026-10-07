import json

import numpy as np

from morpheus_scorekit.freefall_diagnostics import (
    analyze_freefall_state,
    compare_freefall_pair,
    render_pair_markdown,
    scan_freefall_acceleration,
)


def _state(gravity_by_frame, fps=24):
    gravity_by_frame = np.asarray(gravity_by_frame, dtype=float)
    count = len(gravity_by_frame)
    dt = 1.0 / fps
    velocity = np.zeros((count, 3), dtype=float)
    position = np.zeros((count, 3), dtype=float)
    position[0, 2] = 5.0
    for index in range(1, count):
        velocity[index, 2] = velocity[index - 1, 2] + gravity_by_frame[index - 1] * dt
        position[index, 2] = (
            position[index - 1, 2]
            + velocity[index - 1, 2] * dt
            + 0.5 * gravity_by_frame[index - 1] * dt**2
        )
    acceleration = np.zeros((count, 3), dtype=float)
    acceleration[:, 2] = gravity_by_frame
    return {
        "frame": np.arange(count),
        "time": np.arange(count) / fps,
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "acceleration_valid": np.ones(count, dtype=bool),
        "contact_ball_floor": np.zeros(count, dtype=bool),
    }


def _write_state(path, state):
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, **state)


def test_event_window_exposes_temporary_gravity_change():
    normal = _state([-9.81] * 30)
    violation_g = np.full(30, -9.81)
    violation_g[12:19] = -9.81 * 0.5
    violation = _state(violation_g)

    normal_result = analyze_freefall_state(normal, event_window=(12, 18))
    violation_result = analyze_freefall_state(violation, event_window=(12, 18))

    assert normal_result["event_window"]["gravity_agreement_diagnostic"] == 1.0
    assert violation_result["event_window"]["mean_vertical_acceleration_m_s2"] == -4.905
    assert violation_result["event_window"]["gravity_agreement_diagnostic"] < 0.61
    assert violation_result["gravity_agreement_diagnostic"] > violation_result["event_window"][
        "gravity_agreement_diagnostic"
    ]


def test_sliding_window_classifier_estimates_acceleration_without_event_metadata():
    normal = _state([-9.81] * 30)
    violation_g = np.full(30, -9.81)
    violation_g[12:19] = -4.905
    violation = _state(violation_g)

    normal_result = scan_freefall_acceleration(normal, window_size=5)
    violation_result = scan_freefall_acceleration(violation, window_size=5)

    assert normal_result["classification"] == "normal"
    assert normal_result["max_normalized_gravity_residual"] < 1e-8
    assert violation_result["classification"] == "violation"
    assert violation_result["max_normalized_gravity_residual"] > 0.49
    expected_score = np.exp(-violation_result["decision_value"])
    assert abs(violation_result["freefall_physics_score"] - expected_score) < 1e-12
    assert abs(
        violation_result["freefall_violation_score"] - (1.0 - expected_score)
    ) < 1e-12
    peak = violation_result["peak_window"]
    assert 12 <= peak["start_frame"] <= 18
    assert 12 <= peak["end_frame_inclusive"] <= 19
    assert abs(peak["estimated_acceleration_m_s2"] + 4.905) < 1e-8


def test_pair_report_uses_common_horizon_and_loads_original_scores(tmp_path):
    normal = _state([-9.81] * 30)
    violation_g = np.full(30, -9.81)
    violation_g[12:19] = -4.905
    violation = _state(violation_g)
    normal["contact_ball_floor"][23] = True
    violation["contact_ball_floor"][24] = True
    _write_state(tmp_path / "normal" / "state.npz", normal)
    _write_state(tmp_path / "violation" / "state.npz", violation)
    (tmp_path / "violation" / "metadata.json").write_text(
        json.dumps(
            {
                "intervention": {
                    "type": "gravity_scale",
                    "start_frame": 12,
                    "end_frame": 18,
                    "requested_dose": [0.5],
                }
            }
        ),
        encoding="utf-8",
    )
    for variant, value in (("normal", 0.99), ("violation", 0.98)):
        score_dir = tmp_path / "morpheus" / f"{variant}_gt"
        score_dir.mkdir(parents=True)
        (score_dir / "combined_scores.json").write_text(
            json.dumps(
                {
                    "physical_score": value,
                    "statistical_score": value - 0.1,
                    "total_score": value - 0.05,
                    "individual_physical_scores": {
                        "acceleration_conservation": value - 0.02
                    },
                }
            ),
            encoding="utf-8",
        )

    result = compare_freefall_pair(tmp_path)

    assert result["contact_frame_exclusive"] == {"normal": 23, "violation": 24, "common": 23}
    assert result["world_space_common_horizon"]["normal"]["frame_count"] == 23
    assert result["world_space_common_horizon"]["violation"]["frame_count"] == 23
    assert result["original_morpheus"]["available"]
    assert result["automatic_acceleration_classifier"]["normal"]["classification"] == "normal"
    assert (
        result["automatic_acceleration_classifier"]["violation"]["classification"]
        == "violation"
    )
    assert "자동 가속도 classifier" in render_pair_markdown(result)
    assert "개입 프레임만 본 결과" in render_pair_markdown(result)
