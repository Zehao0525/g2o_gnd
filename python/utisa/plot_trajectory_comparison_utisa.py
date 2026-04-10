#!/usr/bin/env python3
"""
Compare MR.CLAM ground truth vs SLAM pre-opt vs post-opt trajectories (2D).

Adapted from multirobot_simulator/evaluator/plot_trajectory_comparison.py for UTIAS
`Robot*_Groundtruth.dat` + TUM `trajectory_{id}.txt`.

Parameters (edit globals below):
  DISPLAY_DURATION_SEC — plot only points with relative time <= this (None = full length).
  SIM_DURATION_SEC     — ATE/APE metrics use samples with relative time <= this
                         (None + DERIVE_SIM_DURATION uses inferred span from post trajectories).
  DERIVE_SIM_DURATION  — if True and SIM_DURATION_SEC is None, infer sim window from outputs.
  EXPERIMENT_JSON      — optional path to read `Duration` when SIM_DURATION_SEC is None
                         and DERIVE_SIM_DURATION is False.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from mrclam_eval_common import (
    align_gt_to_first_sample,
    canonical_robot_id,
    crop_time_interval,
    derive_simulated_duration_from_results,
    discover_robot_gt_paths,
    discover_result_robot_ids,
    duration_from_experiment_json,
    read_mrclam_groundtruth,
    read_landmark_groundtruth,
    read_tum_xy,
    resolve_pre_opt_trajectories_dir,
)

# -------------------- Edit these --------------------
DATASET_DIR = Path("test_data/utisa/MRCLAM7/MRCLAM_Dataset7")
RESULTS_DIR = Path("test_results/utisa_mrclam7_batch")
# Pre-opt lives under RESULTS_DIR/pre_opt_trajectories/<n>/ (needs debug_outputs in experiment JSON)
PRE_OPT_SUBDIR = None  # None = pick newest numeric subfolder that contains trajectory_*.txt
ROBOTS: List[str] = []  # e.g. ["Robot1", "2"] or [] for all

# Plot: show trajectories only for relative time in [0, DISPLAY_DURATION_SEC].
DISPLAY_DURATION_SEC: float | None = None  # e.g. 120.0; None = no display crop

# Metrics: evaluate ATE using times in [0, SIM_DURATION_SEC] in the same relative frame.
SIM_DURATION_SEC: float | None = None  # e.g. 12.0 to match experiment `Duration`
DERIVE_SIM_DURATION = True
EXPERIMENT_JSON: Path | None = Path(
    "test_results/utisa_mrclam7_batch/.batch_merged_configs/experiment_run_single.json"
)
# ----------------------------------------------------


def _nearest_indices(ref_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ref_t, query_t, side="left")
    idx = np.clip(idx, 0, len(ref_t) - 1)
    left = np.clip(idx - 1, 0, len(ref_t) - 1)
    choose_left = np.abs(query_t - ref_t[left]) < np.abs(ref_t[idx] - query_t)
    idx[choose_left] = left[choose_left]
    return idx


def compute_ate(gt_t: np.ndarray, gt_xy: np.ndarray, est_t: np.ndarray, est_xy: np.ndarray) -> Tuple[float, float, int]:
    if len(gt_t) == 0 or len(est_t) == 0:
        return float("nan"), float("nan"), 0
    gt_idx = _nearest_indices(gt_t, est_t)
    d = est_xy - gt_xy[gt_idx]
    dist = np.linalg.norm(d, axis=1)
    return float(np.mean(dist)), float(np.sqrt(np.mean(dist * dist))), int(len(dist))


def compute_ape(
    ref_t: np.ndarray, ref_xy: np.ndarray, est_t: np.ndarray, est_xy: np.ndarray
) -> Tuple[float, float, int]:
    if len(ref_t) == 0 or len(est_t) == 0:
        return float("nan"), float("nan"), 0
    ref_idx = _nearest_indices(ref_t, est_t)
    d = est_xy - ref_xy[ref_idx]
    dist = np.linalg.norm(d, axis=1)
    return float(np.mean(dist)), float(np.sqrt(np.mean(dist * dist))), int(len(dist))


def resolve_sim_duration(results_dir: Path, robot_ids: List[str]) -> float | None:
    if SIM_DURATION_SEC is not None:
        return float(SIM_DURATION_SEC)
    if DERIVE_SIM_DURATION:
        d = derive_simulated_duration_from_results(results_dir, robot_ids)
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
    lm_path = DATASET_DIR / "Landmark_Groundtruth.dat"
    lm_gt = read_landmark_groundtruth(lm_path) if lm_path.is_file() else {}

    post_dir = RESULTS_DIR / "trajectories"
    pre_root = RESULTS_DIR / "pre_opt_trajectories"

    if not post_dir.is_dir():
        raise FileNotFoundError(post_dir)

    available_robots = sorted(gt_map.keys(), key=lambda s: int(s.replace("Robot", "")))
    if ROBOTS:
        robot_keys = []
        for r in ROBOTS:
            r = r.strip()
            if r.startswith("Robot"):
                robot_keys.append(r)
            else:
                robot_keys.append(f"Robot{int(r)}")
        for k in robot_keys:
            if k not in gt_map:
                raise ValueError(f"Unknown robot {k}. Available: {available_robots}")
    else:
        # Intersect GT files with trajectory outputs
        post_ids = set(discover_result_robot_ids(post_dir))
        robot_keys = [rk for rk in available_robots if canonical_robot_id(rk) in post_ids]
        if not robot_keys:
            raise RuntimeError("No overlapping robots between GT and post trajectories.")

    robot_canon = [canonical_robot_id(rk) for rk in robot_keys]
    sim_dur = resolve_sim_duration(RESULTS_DIR, robot_canon)

    subdir = str(PRE_OPT_SUBDIR) if PRE_OPT_SUBDIR is not None else None
    pre_dir = resolve_pre_opt_trajectories_dir(pre_root, robot_canon, explicit_subdir=subdir)
    if pre_dir is None:
        print(
            f"No pre-opt trajectories under {pre_root} (enable `debug_outputs` in the experiment "
            "config if you need pre-optimization paths). Plotting GT vs post-opt only.\n"
        )

    gt_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    pre_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
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

        if pre_dir is not None:
            pre_path = pre_dir / f"trajectory_{cid}.txt"
            if pre_path.is_file():
                pt, pxy = read_tum_xy(pre_path)
                pre_data[cid] = crop_time_interval(pt, pxy, sim_dur)

    # --- Metrics (sim window) ---
    colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple"]
    print("\nATE / APE (2D, meters) — evaluation window [0, sim_duration]")
    if sim_dur is not None:
        print(f"  sim_duration used: {sim_dur:.6f} s")
    pre_all: List[np.ndarray] = []
    post_all: List[np.ndarray] = []
    ape_all: List[np.ndarray] = []

    for i, cid in enumerate(sorted(gt_data.keys(), key=lambda x: int(x) if x.isdigit() else x)):
        gt_t, gt_xy = gt_data[cid]
        post_t, post_xy = post_data[cid]
        post_mean, post_rmse, post_n = compute_ate(gt_t, gt_xy, post_t, post_xy)
        if cid in pre_data:
            pre_t, pre_xy = pre_data[cid]
            pre_mean, pre_rmse, pre_n = compute_ate(gt_t, gt_xy, pre_t, pre_xy)
            ape_mean, ape_rmse, ape_n = compute_ape(pre_t, pre_xy, post_t, post_xy)
            print(
                f"  Robot {cid}: pre  mean={pre_mean:.6f} rmse={pre_rmse:.6f} (n={pre_n}) | "
                f"post mean={post_mean:.6f} rmse={post_rmse:.6f} (n={post_n}) | "
                f"APE(pre,post) mean={ape_mean:.6f} rmse={ape_rmse:.6f} (n={ape_n})"
            )
            if pre_n > 0:
                pre_idx = _nearest_indices(gt_t, pre_t)
                pre_all.append(np.linalg.norm(pre_xy - gt_xy[pre_idx], axis=1))
            if ape_n > 0:
                ape_idx = _nearest_indices(pre_t, post_t)
                ape_all.append(np.linalg.norm(post_xy - pre_xy[ape_idx], axis=1))
        else:
            print(
                f"  Robot {cid}: post mean={post_mean:.6f} rmse={post_rmse:.6f} (n={post_n}) | "
                "pre/APE skipped (no pre-opt file)"
            )
        if post_n > 0:
            post_idx = _nearest_indices(gt_t, post_t)
            post_all.append(np.linalg.norm(post_xy - gt_xy[post_idx], axis=1))

    if pre_all and post_all and ape_all:
        pre_c = np.concatenate(pre_all)
        post_c = np.concatenate(post_all)
        ape_c = np.concatenate(ape_all)
        print(
            f"  Overall: pre mean={np.mean(pre_c):.6f} | post mean={np.mean(post_c):.6f} | "
            f"APE(pre,post) mean={np.mean(ape_c):.6f}"
        )
    elif post_all:
        post_c = np.concatenate(post_all)
        print(f"  Overall: post mean={np.mean(post_c):.6f} (pre-opt not available for aggregate)")

    # --- Plot (optional display crop, independent of sim_dur) ---
    disp_cut = DISPLAY_DURATION_SEC
    fig, ax = plt.subplots(figsize=(10, 8))
    drone0 = sorted(gt_data.keys(), key=lambda x: int(x) if x.isdigit() else x)[0]

    for i, cid in enumerate(sorted(gt_data.keys(), key=lambda x: int(x) if x.isdigit() else x)):
        c = colors[i % len(colors)]
        gt_t, gt_xy = gt_data[cid]
        post_t, post_xy = post_data[cid]

        if disp_cut is not None:
            gt_xy_p = gt_xy[gt_t <= disp_cut + 1e-9]
            post_xy_p = post_xy[post_t <= disp_cut + 1e-9]
            pre_xy_p: np.ndarray | None = None
            if cid in pre_data:
                pre_t, pre_xy = pre_data[cid]
                pre_xy_p = pre_xy[pre_t <= disp_cut + 1e-9]
        else:
            gt_xy_p = gt_xy
            post_xy_p = post_xy
            pre_xy_p = pre_data[cid][1] if cid in pre_data else None

        ax.plot(gt_xy_p[:, 0], gt_xy_p[:, 1], "-", color=c, linewidth=2.0, label=f"Robot {cid} GT")
        is0 = cid == drone0
        if pre_xy_p is not None and len(pre_xy_p) > 0:
            ax.plot(
                pre_xy_p[:, 0],
                pre_xy_p[:, 1],
                "-",
                color=c,
                alpha=0.35,
                linewidth=2.0,
                label="pre (ref)" if is0 else None,
            )
        ax.plot(post_xy_p[:, 0], post_xy_p[:, 1], ":", color=c, linewidth=2.2, label="post (ref)" if is0 else None)

    if lm_gt:
        lm_ids = sorted(lm_gt.keys())
        lm_xy = np.vstack([lm_gt[i] for i in lm_ids])
        ax.scatter(lm_xy[:, 0], lm_xy[:, 1], marker="x", s=36, c="k", alpha=0.8, label="landmarks GT")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = f"UTISA MR.CLAM — {DATASET_DIR.name} vs {RESULTS_DIR.name}"
    if disp_cut is not None:
        title += f" | plot ≤ {disp_cut:.1f} s"
    if sim_dur is not None:
        title += f" | metrics ≤ {sim_dur:.1f} s"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
