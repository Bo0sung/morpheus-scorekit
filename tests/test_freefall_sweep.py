import json

import numpy as np

from morpheus_scorekit.freefall_sweep import (
    build_sweep_manifest,
    evaluate_sweep,
    render_sweep_markdown,
)


def _write_state(path, gravity, fps=24):
    gravity = np.asarray(gravity, dtype=float)
    count = len(gravity)
    dt = 1.0 / fps
    velocity = np.zeros((count, 3), dtype=float)
    position = np.zeros((count, 3), dtype=float)
    position[0, 2] = 8.0
    for index in range(1, count):
        velocity[index, 2] = velocity[index - 1, 2] + gravity[index - 1] * dt
        position[index, 2] = (
            position[index - 1, 2]
            + velocity[index - 1, 2] * dt
            + 0.5 * gravity[index - 1] * dt**2
        )
    acceleration = np.zeros((count, 3), dtype=float)
    acceleration[:, 2] = gravity
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        frame=np.arange(count),
        time=np.arange(count) / fps,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        acceleration_valid=np.ones(count, dtype=bool),
        contact_ball_floor=np.zeros(count, dtype=bool),
    )


def test_manifest_builds_unique_seed_and_dose_pairs(tmp_path):
    (tmp_path / "scripts").mkdir()
    manifest = build_sweep_manifest(
        tmp_path,
        count=2,
        start_seed=100,
        doses=[0.9, 0.5],
        start_frame=8,
        end_frame=20,
    )

    assert len(manifest["pairs"]) == 4
    assert len({pair["scene_id"] for pair in manifest["pairs"]}) == 4
    assert {pair["seed"] for pair in manifest["pairs"]} == {100, 101}


def test_sweep_evaluation_recovers_dose_response_and_writes_report(tmp_path):
    manifest = build_sweep_manifest(
        tmp_path,
        count=2,
        start_seed=100,
        doses=[0.9, 0.5, 0.0],
        start_frame=8,
        end_frame=20,
    )
    for pair in manifest["pairs"]:
        normal_g = np.full(32, -9.81)
        violation_g = normal_g.copy()
        violation_g[8:21] = -9.81 * pair["dose"]
        pair_dir = tmp_path / "outputs" / pair["scene_id"]
        _write_state(pair_dir / "normal" / "state.npz", normal_g)
        _write_state(pair_dir / "violation" / "state.npz", violation_g)

    result = evaluate_sweep(manifest)

    assert result["calibration"]["threshold"] == 0.02
    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["f1"] == 1.0
    assert result["metrics"]["false_positive_rate"] == 0.0
    assert result["metrics"]["dose_response_correlation"] > 0.999
    assert result["metrics"]["monotonic_seed_fraction"] == 1.0
    assert [row["dose"] for row in result["dose_summary"]] == [0.9, 0.5, 0.0]
    assert "중력 배율별 결과" in render_sweep_markdown(result)
    json.dumps(result)
