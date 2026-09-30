import numpy as np

from morpheus_scorekit.models import Trajectory


def test_trajectory_round_trip(tmp_path):
    positions = np.column_stack([
        np.linspace(10, 100, 20),
        np.linspace(50, 52, 20),
        np.zeros(20),
    ]).astype(np.float32)
    original = Trajectory(fps=30, objects={1: positions}, experiment="falling_ball")
    path = original.save(tmp_path / "trajectory.npz")
    restored = Trajectory.load(path)
    assert restored.experiment == "falling_ball"
    assert restored.fps == 30
    np.testing.assert_allclose(restored.objects[1], positions)
    assert restored.validate()["valid"]


def test_invalid_short_trajectory():
    trajectory = Trajectory(fps=30, objects={1: np.zeros((3, 3))})
    assert not trajectory.validate()["valid"]

