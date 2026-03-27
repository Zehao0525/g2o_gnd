#!/usr/bin/env python3
"""
Generate multiple multidrone log bundles under test_data/multidrone/batch/<n>/.

Each run n writes:
  - trajectories.json   (fresh random trajectories, seed=n by default)
  - bot_ids.txt
  - msg_log_<id>.txt, gt_log_<id>.txt  (same outputs as generate_multirobot_data.py)

Intended use: call generate_batch_multidrone_data(X) from Python (not CLI args).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from multirobot_simulator.simulator import WorldSim
    from multirobot_simulator.trajectory_generator import trajectory_generation
except ModuleNotFoundError:
    # Allow direct execution: python .../generate_batch_multidrone_data.py
    from simulator import WorldSim
    from trajectory_generator import trajectory_generation

DEFAULT_CONFIG_PATH = "python/multirobot_simulator/config/sim_config.json"
DEFAULT_BATCH_ROOT = "test_data/multidrone/batch_f1"
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


def _run_single_simulation(
    *,
    out_dir: str,
    config_path: str,
    trajectories_path: str,
    trajectory_seed: int | None,
    max_steps: int,
) -> None:
    """Mirror generate_multirobot_data.py: new trajectories + WorldSim + step loop."""
    bot_ids = _get_bot_ids(config_path)
    trajectory_generation(
        bot_ids,
        trajectories_path,
        visualize=False,
        seed=trajectory_seed,
    )
    world_sim, _, _ = WorldSim.create(
        config_path=config_path,
        trajectory_path=trajectories_path,
        log_path=out_dir,
    )
    _write_bot_ids_txt(out_dir, bot_ids)

    for k in range(max_steps):
        world_sim.step()
        if world_sim.reached_dest_all():
            break


def generate_batch_multidrone_data(
    batch_count: int,
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
    batch_root: str | Path = DEFAULT_BATCH_ROOT,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> None:
    """
    Create directory `batch_root` and, for each n in range(batch_count), create
    `batch_root / str(n) /` with a new trajectories.json and all simulator logs.

    Parameters
    ----------
    batch_count
        Number of datasets (non-negative integer).
    config_path
        Same sim_config.json used by generate_multirobot_data.py.
    batch_root
        Defaults to test_data/multidrone/batch (relative to process CWD).
    max_steps
        Upper bound on simulation steps per dataset (same as generate_multirobot_data.py loop).
    """
    if batch_count < 0:
        raise ValueError("batch_count must be non-negative")

    root = Path(batch_root)
    root.mkdir(parents=True, exist_ok=True)

    for n in range(batch_count):
        run_dir = root / str(n)
        run_dir.mkdir(parents=True, exist_ok=True)
        traj_path = str(run_dir / "trajectories.json")
        print(f"[batch {n}] output -> {run_dir.resolve()}")
        _run_single_simulation(
            out_dir=str(run_dir),
            config_path=config_path,
            trajectories_path=traj_path,
            trajectory_seed=n,
            max_steps=max_steps,
        )


if __name__ == "__main__":
    # Edit the argument when running as a script; primary API is generate_batch_multidrone_data(X).
    X = 30
    generate_batch_multidrone_data(X)
