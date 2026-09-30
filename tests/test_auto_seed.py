import cv2
import numpy as np

from morpheus_scorekit.auto_seed import suggest_moving_object_seed


def test_auto_seed_finds_ball_in_first_frame(tmp_path):
    video = tmp_path / "falling_ball.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (160, 120)
    )
    for frame_index in range(20):
        frame = np.full((120, 160, 3), 35, dtype=np.uint8)
        cv2.rectangle(frame, (10, 100), (150, 108), (90, 90, 90), -1)
        cv2.circle(frame, (78, 18 + frame_index * 3), 8, (40, 70, 240), -1)
        writer.write(frame)
    writer.release()

    preview = tmp_path / "seed.jpg"
    result = suggest_moving_object_seed(video, preview_path=preview)

    assert abs(result.x - 78) < 3
    assert abs(result.y - 18) < 3
    assert result.area > 50
    assert preview.is_file()
