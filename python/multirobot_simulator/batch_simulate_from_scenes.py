#!/usr/bin/env python3
"""
Simulate a batch from pre-generated scenes under trajectories_root.

For every immediate subdirectory that contains trajectories.json:
  - Reads trajectories.json and optional landmarks.json
  - Runs the WorldSim step loop
  - Writes logs to batch_root/<subdir_name>/

Scene folder names can be numeric (0,1,...) or arbitrary; numeric names are
processed in numeric order.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

try:
    from multirobot_simulator.simulator import WorldSim
    from multirobot_simulator.trajectory_generator import trajectory_generation  # noqa: F401
except ModuleNotFoundError:
    # Allow direct execution: python .../simulate_batch_multidrone_from_trajectories.py
    from simulator import WorldSim  # type: ignore
    from trajectory_generator import trajectory_generation  # noqa: F401  # type: ignore


DEFAULT_CONFIG_PATH = "python/multirobot_simulator/config/sim_config_batch.json"
DEFAULT_TRAJ_ROOT = "test_data/multidrone2/scenes/30_20lm_2midpoint"
DEFAULT_BATCH_ROOT = "test_data/multidrone2/long_traj/30_20lm_2midpoint_lmf2_of2_bounded"
DEFAULT_MAX_STEPS = 10000


def _get_bot_ids(config_path: str) -> list[str]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return list(cfg["bots"].keys())


def _write_bot_ids_txt(log_dir: str, bot_ids: list[str]) -> None:
    os.makedirs(log_dir, exist_ok=True)
    bot_ids_path = os.path.join(log_dir, "bot_ids.txt")
    with open(bot_ids_path, "w", encoding="utf-8") as f:
        for bid in bot_ids:
            f.write(str(bid) + "\n")
    print(f"Wrote {bot_ids_path}")


def _discover_scene_run_dirs(traj_root: Path) -> list[str]:
    """
    Names of immediate subdirectories of traj_root that contain trajectories.json.
    Sorted: pure integer names in numeric order, then other names lexicographically.
    """
    if not traj_root.is_dir():
        return []
    names: list[str] = []
    for p in traj_root.iterdir():
        if p.is_dir() and (p / "trajectories.json").is_file():
            names.append(p.name)

    def sort_key(name: str) -> tuple:
        try:
            return (0, int(name), "")
        except ValueError:
            return (1, 0, name)

    return sorted(names, key=sort_key)


def run_batch_multidrone_simulations_from_trajectories(
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
    trajectories_root: str | Path = DEFAULT_TRAJ_ROOT,
    batch_root: str | Path = DEFAULT_BATCH_ROOT,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> None:
    """
    For each scene subdirectory under trajectories_root that contains
    trajectories.json, run simulation and write logs to batch_root/<same_name>/.
    """
    traj_root = Path(trajectories_root)
    batch_root = Path(batch_root)
    batch_root.mkdir(parents=True, exist_ok=True)

    run_names = _discover_scene_run_dirs(traj_root)
    if not run_names:
        raise FileNotFoundError(
            f"No scene runs found under {traj_root.resolve()} "
            "(expected subdirs each with trajectories.json)."
        )

    bot_ids = _get_bot_ids(config_path)

    for run_name in run_names:
        src_traj_path = traj_root / run_name / "trajectories.json"

        run_dir = batch_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Copy trajectories.json into the run_dir for compatibility
        # with code that assumes it's present next to logs.
        dst_traj_path = run_dir / "trajectories.json"
        shutil.copyfile(src_traj_path, dst_traj_path)

        src_lm_path = traj_root / run_name / "landmarks.json"
        dst_lm_path = run_dir / "landmarks.json"
        landmark_arg = None
        if src_lm_path.exists():
            shutil.copyfile(src_lm_path, dst_lm_path)
            landmark_arg = str(dst_lm_path.resolve())

        print(f"[batch {run_name}] output -> {run_dir.resolve()}")
        world_sim, _, _ = WorldSim.create(
            config_path=config_path,
            trajectory_path=str(dst_traj_path),
            log_path=str(run_dir),
            landmark_path=landmark_arg,
        )

        _write_bot_ids_txt(str(run_dir), bot_ids)

        for k in range(max_steps):
            world_sim.step()
            if world_sim.reached_dest_all():
                break


if __name__ == "__main__":
    run_batch_multidrone_simulations_from_trajectories()

