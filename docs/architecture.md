# Architecture

## Boundary 1: video to trajectory

`SeededColorTracker.extract()` accepts a video or frame directory and returns a
`Trajectory`.  The output is frame-aligned and uses the paper implementation's
coordinate convention `[Y, X, depth]`.  It never computes a physics score.

The included colour backend is intentionally small and deterministic so the public
falling-ball clip can be validated without downloading the approximately 900 MB SAM2
checkpoint.  `MorpheusSAM2Adapter` documents the paper-faithful production backend.

## Boundary 2: valid trajectory to score

`MorpheusTrajectoryScorer.score()` accepts only a `Trajectory`, an experiment name,
and an output directory.  It serializes the trajectory into the official scorer's
input format and calls the pinned reference implementation at commit
`ee3b6aec1474a9b905eba9a52efb5ea6cfd9f0bc`.

The output names intentionally follow the official repository:

- `physical_score`: paper Physical Invariance Score
- `statistical_score`: paper Dynamical Score
- `total_score`: arithmetic mean of both

## Profiles

- `real_world_trajectories`: official real-video scoring windows; used for validation.
- `VGM_trajectories`: generated/test-video 25% sliding-window profile; CLI `--generated`.

The real-world score is not used to renormalize another video's score.  It is an
empirical upper reference and a regression target for the evaluation pipeline.

