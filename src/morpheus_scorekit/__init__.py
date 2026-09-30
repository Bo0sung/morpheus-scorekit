"""Small, explicit interfaces around the Morpheus evaluation pipeline."""

from .models import Trajectory
from .extraction import MorpheusSAM2Adapter, SeededColorTracker, TrackingConfig
from .sam2_extraction import SAM2DepthConfig, SAM2DepthTrajectoryExtractor
from .scoring import MorpheusTrajectoryScorer, ScoringConfig

__all__ = [
    "Trajectory",
    "SeededColorTracker",
    "TrackingConfig",
    "MorpheusSAM2Adapter",
    "SAM2DepthConfig",
    "SAM2DepthTrajectoryExtractor",
    "MorpheusTrajectoryScorer",
    "ScoringConfig",
]
