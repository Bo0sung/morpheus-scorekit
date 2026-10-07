import numpy as np

from morpheus_scorekit.freefall_segments import estimate_first_contact_frame, trim_precontact
from morpheus_scorekit.models import Trajectory


def test_trim_precontact_at_first_low_point():
    falling = np.linspace(20, 100, 16)
    bouncing = np.linspace(96, 60, 8)
    rows = np.concatenate([falling, bouncing])
    positions = np.column_stack([rows, np.full(len(rows), 50.0), np.zeros(len(rows))])
    trajectory = Trajectory(fps=24, objects={1: positions}, experiment="falling_ball")

    contact = estimate_first_contact_frame(trajectory, smoothing_window=1)
    trimmed, info = trim_precontact(trajectory)

    assert contact == 15
    assert trimmed.frame_count in range(14, 17)
    assert trimmed.frame_count == info["contact_frame_exclusive"]
    assert trimmed.validate()["valid"]
