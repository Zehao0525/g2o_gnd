#!/usr/bin/env python3
"""
Visualize and measure APE for UTISA `utisa_slam_unit_experiment` (or any) run
that only exports a subset of robots under `RESULTS_DIR/trajectories/`.

Uses MR.CLAM `Robot*_Groundtruth.dat` for GT and TUM `trajectory_{id}.txt` for post-opt.
Time window: `SIM_DURATION_SEC`, or derived from trajectories, or `Duration` from
`EXPERIMENT_JSON` (e.g. merged base config).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from mrclam_eval_common import (
    align_gt_to_first_sample,
    canonical_robot_id,
    compute_ape,
    crop_time_interval,
    derive_simulated_duration_from_results,
    discover_robot_gt_paths,
    discover_result_robot_ids,
    duration_from_experiment_json,
    read_mrclam_groundtruth,
    read_tum_xy,
)

# -------------------- Edit these --------------------
DATASET_DIR = Path("test_data/utisa/MRCLAM7/MRCLAM_Dataset7")
RESULTS_DIR = Path("test_results/utisa_unit_experiment")
DISPLAY_DURATION_SEC: float | None = None
SIM_DURATION_SEC: float | None = None
DERIVE_SIM_DURATION = True
EXPERIMENT_JSON: Path | None = Path("Source/Examples/UTISA_slam/config/unit_experiment_config.json")
ROBOT_KEYS = ["Robot1"]
# ----------------------------------------------------


def _robot_key_from_canon(canon: str) -> str:
    return f"Robot{int(canon)}"


def _resolve_sim_duration(robot_canon: List[str]) -> float | None:
    if SIM_DURATION_SEC is not None:
        return float(SIM_DURATION_SEC)
    if DERIVE_SIM_DURATION:
        d = derive_simulated_duration_from_results(RESULTS_DIR, robot_canon)
        print(f"Inferred simulated duration from trajectories: {d:.6f} s")
        return d
    if EXPERIMENT_JSON is not None:
        d = duration_from_experiment_json(EXPERIMENT_JSON)
        if d is not None:
            print(f"Duration from {EXPERIMENT_JSON}: {d:.6f} s")
            return float(d)
    print("SIM_DURATION_SEC is None and no duration inferred; metrics use full trajectories.")
    return None


def main() -> None:
    gt_map = discover_robot_gt_paths(DATASET_DIR)
    if not gt_map:
        raise RuntimeError(f"No Robot*_Groundtruth.dat under {DATASET_DIR}")

    post_dir = RESULTS_DIR / "trajectories"
    if not post_dir.is_dir():
        raise FileNotFoundError(post_dir)

    post_ids = discover_result_robot_ids(post_dir)
    if not post_ids:
        raise RuntimeError(f"No trajectory_*.txt under {post_dir}")

    robot_keys = [_robot_key_from_canon(rid) for rid in post_ids]
    if ROBOT_KEYS:
        robot_keys = [rk for rk in robot_keys if rk in ROBOT_KEYS]
    for rk in robot_keys:
        if rk not in gt_map:
            raise ValueError(f"No ground truth for {rk}; available: {sorted(gt_map.keys())}")

    robot_canon = [canonical_robot_id(rk) for rk in robot_keys]
    sim_dur = _resolve_sim_duration(robot_canon)

    gt_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    post_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for rk, cid in zip(robot_keys, robot_canon):
        gt_t, gt_xy, _ = read_mrclam_groundtruth(gt_map[rk])
        gt_t, gt_xy = align_gt_to_first_sample(gt_t, gt_xy)
        gt_t_m, gt_xy_m = crop_time_interval(gt_t, gt_xy, sim_dur)

        post_path = post_dir / f"trajectory_{cid}.txt"
        if not post_path.is_file():
            raise FileNotFoundError(post_path)
        ot, oxy = read_tum_xy(post_path)
        ot_m, oxy_m = crop_time_interval(ot, oxy, sim_dur)

        gt_data[cid] = (gt_t_m, gt_xy_m)
        post_data[cid] = (ot_m, oxy_m)

    disp_dur = DISPLAY_DURATION_SEC if DISPLAY_DURATION_SEC is not None else sim_dur

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(robot_canon), 1)))

    print("APE (GT vs post-optimization), planar |e|:")
    for i, cid in enumerate(robot_canon):
        gt_t, gt_xy = gt_data[cid]
        post_t, post_xy = post_data[cid]
        gt_td, gt_xyd = crop_time_interval(gt_t, gt_xy, disp_dur)
        post_td, post_xyd = crop_time_interval(post_t, post_xy, disp_dur)

        mean_e, rmse_e, n = compute_ape(gt_td, gt_xyd, post_td, post_xyd)
        print(f"  Robot {cid}: mean={mean_e:.6f} m  RMSE={rmse_e:.6f} m  (n={n})")

        ax.plot(gt_xyd[:, 0], gt_xyd[:, 1], "-", color=colors[i], linewidth=1.8, alpha=0.85, label=f"GT R{cid}")
        ax.plot(
            post_xyd[:, 0],
            post_xyd[:, 1],
            ":",
            color=colors[i],
            linewidth=1.4,
            alpha=0.95,
            label=f"Post R{cid}",
        )

    ax.set_title("UTISA unit experiment: GT vs post-opt")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
