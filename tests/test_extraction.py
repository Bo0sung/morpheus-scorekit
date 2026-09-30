import cv2
import numpy as np

from morpheus_scorekit.extraction import SeededColorTracker


def test_seeded_color_tracker(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(16):
        image = np.zeros((160, 200, 3), dtype=np.uint8)
        cv2.circle(image, (80, 20 + index * 6), 8, (0, 140, 255), -1)
        encoded = cv2.imencode(".jpg", image)[1]
        encoded.tofile(frames / f"{index:05d}.jpg")
    trajectory = SeededColorTracker().extract(
        frames, {1: (80, 20)}, fps=30, experiment="falling_ball"
    )
    y = trajectory.objects[1][:, 0]
    assert np.isfinite(y).mean() > 0.9
    assert y[-1] - y[0] > 70
