#!/usr/bin/env python3
"""
Generate and save a batch of multidrone trajectories only.

Program 1 of the new two-stage pipeline:
  - Creates per run index n:
      <trajectories_root>/<n>/trajectories.json
      <trajectories_root>/<n>/landmarks.json   (if sim_config n_landmarks > 0)

Landmark count and placement use ``n_landmarks`` from sim_config.json (root).
Does NOT run any simulation / sensor logging.

Primary use (API, not CLI args):
  from multirobot_simulator.generate_batch_multidrone_trajectories import (
      generate_batch_multidrone_trajectories
  )
  generate_batch_multidrone_trajectories(X)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from multirobot_simulator.trajectory_generator import trajectory_generation
    from multirobot_simulator.landmark_generator import landmark_generation
except ModuleNotFoundError:
    # Allow direct execution: python .../generate_batch_multidrone_trajectories.py
    from trajectory_generator import trajectory_generation
    from landmark_generator import landmark_generation


DEFAULT_CONFIG_PATH = "python/multirobot_simulator/config/sim_config_batch.json"
DEFAULT_TRAJ_ROOT = "test_data/multidrone2/scenes/30_20lm_2midpoint"
N_TRAJECTORIES = 30
# Trajectory placement can fail for unlucky seeds; retry with a perturbed seed before giving up.
_MAX_TRAJ_PLACEMENT_ATTEMPTS = 64


def _get_bot_ids(config_path: str) -> list[str]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return list(cfg["bots"].keys())


def _get_n_landmarks(config_path: str) -> int:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return int(cfg.get("n_landmarks", 0))


def _get_n_trajectory_midpoints(config_path: str) -> int:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return int(cfg.get("n_trajectory_midpoints", 0))


def generate_batch_multidrone_trajectories(
    traj_count: int,
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
    trajectories_root: str | Path = DEFAULT_TRAJ_ROOT,
) -> None:
    """
    Create `traj_count` trajectories (one JSON per n).

    Output:
      <trajectories_root>/<n>/trajectories.json
      <trajectories_root>/<n>/landmarks.json  (when n_landmarks > 0 in sim_config)
    """
    if traj_count < 0:
        raise ValueError("traj_count must be non-negative")

    bot_ids = _get_bot_ids(config_path)
    n_landmarks = _get_n_landmarks(config_path)
    n_mid = _get_n_trajectory_midpoints(config_path)

    root = Path(trajectories_root)
    root.mkdir(parents=True, exist_ok=True)

    for n in range(traj_count):
        run_dir = root / str(n)
        run_dir.mkdir(parents=True, exist_ok=True)
        traj_path = str(run_dir / "trajectories.json")
        print(f"[traj {n}] output -> {Path(traj_path).resolve()}")

        seed_used = n
        for attempt in range(_MAX_TRAJ_PLACEMENT_ATTEMPTS):
            try:
                # Vary seed on retries so shuffle / sampling gets a fresh draw.
                seed_try = n + 100_000 * attempt
                trajectory_generation(
                    bot_ids,
                    traj_path,
                    visualize=False,
                    seed=seed_try,
                    n_trajectory_midpoints=n_mid,
                )
                seed_used = seed_try
                break
            except ValueError as e:
                print(
                    f"[traj {n}] trajectory placement failed "
                    f"(attempt {attempt + 1}/{_MAX_TRAJ_PLACEMENT_ATTEMPTS}): {e}"
                )
                print(f"[traj {n}] retrying with a different seed…")
        else:
            raise RuntimeError(
                f"[traj {n}] gave up after {_MAX_TRAJ_PLACEMENT_ATTEMPTS} placement attempts."
            )

        if n_landmarks > 0:
            lm_path = str(run_dir / "landmarks.json")
            landmark_generation(
                lm_path,
                n_landmarks,
                seed=seed_used,
            )
            print(f"[traj {n}] landmarks -> {Path(lm_path).resolve()}")


if __name__ == "__main__":
    generate_batch_multidrone_trajectories(N_TRAJECTORIES)

