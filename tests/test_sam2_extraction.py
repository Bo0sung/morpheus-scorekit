import cv2
import numpy as np

from morpheus_scorekit.sam2_extraction import (
    SAM2DepthConfig,
    SAM2DepthTrajectoryExtractor,
)


class FakeSAM2Tracker:
    def __init__(self, frame_count: int):
        self.frame_count = frame_count
        self.prompts = None

    def initialize_tracking(self, frames_dir, prompts):
        self.prompts = prompts
        initial = np.zeros((64, 80), dtype=bool)
        initial[8:14, 17:23] = True
        return object(), {1: initial}

    def propagate(self, state):
        segments = {}
        for index in range(self.frame_count):
            mask = np.zeros((64, 80), dtype=bool)
            mask[8 + index:14 + index, 17:23] = True
            segments[index] = {1: mask}
        return segments


class FakeDepthEstimator:
    def predict(self, rgb_image):
        y = np.arange(rgb_image.shape[0], dtype=np.float32)
        return np.repeat(y[:, None], rgb_image.shape[1], axis=1)


def test_sam2_extractor_with_connected_models(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(10):
        image = np.zeros((64, 80, 3), dtype=np.uint8)
        cv2.rectangle(image, (17, 8 + index), (22, 13 + index), (255, 255, 255), -1)
        cv2.imencode(".jpg", image)[1].tofile(frames / f"{index:05d}.jpg")

    tracker = FakeSAM2Tracker(frame_count=10)
    extractor = SAM2DepthTrajectoryExtractor(
        SAM2DepthConfig(reference_repo=tmp_path, use_depth=True),
        tracker=tracker,
        depth_estimator=FakeDepthEstimator(),
    )
    trajectory = extractor.extract(
        frames,
        {1: [(20.0, 10.0)]},
        negative_points_xy={1: [(2.0, 2.0)]},
        fps=24,
        experiment="falling_ball",
        overlay_dir=tmp_path / "overlay",
        mask_dir=tmp_path / "masks",
    )

    assert tracker.prompts[1]["labels"].tolist() == [1, 0]
    assert trajectory.objects[1].shape == (10, 3)
    assert np.allclose(trajectory.objects[1][:, 1], 19.5)
    assert trajectory.objects[1][-1, 0] > trajectory.objects[1][0, 0]
    assert np.allclose(trajectory.objects[1][:, 2], trajectory.objects[1][:, 0])
    assert trajectory.metadata["paper_faithful_tracker"] is True
    assert len(list((tmp_path / "overlay").glob("*.jpg"))) == 10
    assert len(list((tmp_path / "masks").glob("*.png"))) == 10


def test_sam2_environment_report_identifies_missing_checkpoint(tmp_path):
    report = SAM2DepthTrajectoryExtractor(
        SAM2DepthConfig(reference_repo=tmp_path, use_depth=False)
    ).environment_report()
    assert report["checkpoint_exists"] is False
    assert report["ready"] is False

