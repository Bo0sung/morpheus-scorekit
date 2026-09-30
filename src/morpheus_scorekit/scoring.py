from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .models import Trajectory


@dataclass(frozen=True)
class ScoringConfig:
    reference_repo: str | Path = "vendor/Morpheus"
    epochs: int = 200_000
    learning_rate: float = 1e-3
    source_category: str = "real_world_trajectories"
    compute_physical: bool = True
    compute_dynamical: bool = True
    random_seed: int = 0


class MorpheusTrajectoryScorer:
    """Score an already-extracted valid trajectory with the official Morpheus laws.

    Tracking is intentionally absent from this class.  The implementation imports a
    pinned checkout of the authors' public scorer and adapts our explicit NPZ schema to
    the pickle inputs expected by that scorer.
    """

    def __init__(self, config: ScoringConfig | None = None):
        self.config = config or ScoringConfig()
        self.reference_repo = Path(self.config.reference_repo).resolve()
        if not (self.reference_repo / "morpheus" / "scoring").exists():
            raise FileNotFoundError(f"official Morpheus checkout not found: {self.reference_repo}")

    def score(
        self,
        trajectory: Trajectory,
        experiment: str | None = None,
        *,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        experiment = experiment or trajectory.experiment
        if not experiment:
            raise ValueError("experiment must be provided because each phenomenon uses different laws")
        validation = trajectory.validate()
        if not validation["valid"]:
            raise ValueError("trajectory is invalid: " + "; ".join(validation["errors"]))

        self._ensure_reference_importable()
        # The reference PINNs initialise weights randomly.  Pin the seed at the wrapper
        # boundary so repeated validation runs are comparable.
        import torch
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)
        from morpheus.scoring.dynamical_score import run_pin_framework
        from morpheus.scoring.physical_score import calculate_one_video_score
        from morpheus.taxonomy import EXP_NAME_MAP, phenomenon_for

        if experiment not in EXP_NAME_MAP:
            raise ValueError(f"unsupported Morpheus experiment: {experiment}")

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = self._write_reference_inputs(trajectory, output)

        result: dict[str, Any] = {
            "experiment": experiment,
            "valid": True,
            "trajectory_quality": trajectory.quality_summary(),
            "scoring_profile": {
                "reference_repo": str(self.reference_repo),
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "source_category": self.config.source_category,
                "random_seed": self.config.random_seed,
            },
        }

        canonical = EXP_NAME_MAP[experiment]
        if self.config.compute_physical:
            physical = calculate_one_video_score(
                str(paths["object_1"]),
                canonical,
                self.config.source_category,
                figure_name=experiment,
                obj_2_centers_pkl_path=str(paths["object_2"]) if "object_2" in paths else None,
                distance_pkl_path=str(paths["distances"]) if "distances" in paths else None,
                angle_pkl_path=str(paths["angles"]) if "angles" in paths else None,
                videos_dir=str(output),
            )
            result.update(physical)

        if self.config.compute_dynamical:
            dynamic = run_pin_framework(
                str(paths["object_1"]),
                phenomenon=phenomenon_for(experiment),
                verbose=False,
                n_epochs=self.config.epochs,
                lr=self.config.learning_rate,
                obj_2_centers_pkl_path=str(paths["object_2"]) if "object_2" in paths else None,
                angles_pkl_path=str(paths["angles"]) if "angles" in paths else None,
            )
            nmse = float(dynamic["nmse"])
            result["individual_statistical_scores"] = {
                "MSE": float(dynamic["mse"]),
                "NMSE": nmse,
            }
            result["statistical_score"] = max(1.0 - min(nmse, 1.0), 0.0)

        physical_score = result.get("physical_score")
        dynamical_score = result.get("statistical_score")
        if physical_score is not None and dynamical_score is not None:
            result["total_score"] = (float(physical_score) + float(dynamical_score)) / 2.0

        result_path = output / "combined_scores.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    def _ensure_reference_importable(self) -> None:
        path = str(self.reference_repo)
        if path not in sys.path:
            sys.path.insert(0, path)

    @staticmethod
    def _write_reference_inputs(trajectory: Trajectory, output: Path) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for object_id, values in trajectory.objects.items():
            path = output / f"centres3d_obj_{object_id}.pkl"
            with path.open("wb") as stream:
                pickle.dump(np.asarray(values, dtype=np.float32), stream)
            paths[f"object_{object_id}"] = path

        if trajectory.distances is not None:
            path = output / "max_distance.pkl"
            with path.open("wb") as stream:
                pickle.dump({int(k): np.asarray(v) for k, v in trajectory.distances.items()}, stream)
            paths["distances"] = path
        if trajectory.angles is not None:
            path = output / "angles.pkl"
            with path.open("wb") as stream:
                pickle.dump(trajectory.angles, stream)
            paths["angles"] = path
        return paths
