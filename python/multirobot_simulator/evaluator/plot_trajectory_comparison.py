#!/usr/bin/env python3
"""
Plot ground-truth vs pre-optimization vs post-optimization trajectories.

Expected TUM format per line:
  timestamp x y z qx qy qz qw

Line styles:
  - Ground truth: solid
  - Pre-optimization: semi-transparent solid
  - Post-optimization: dotted
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# This script intentionally does NOT depend on EVO.

def read_tum_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"File has fewer than 4 columns: {path}")
    return data[:, 0], data[:, 1:4]  # t, xyz


def read_gt_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[Tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # gt format used in this project: t pose x y z ...
            if parts[1] != "pose":
                continue
            rows.append((float(parts[0]), float(parts[2]), float(parts[3]), float(parts[4])))
    if not rows:
        raise ValueError(f"No GT pose rows found in: {path}")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1:4]


def _nearest_indices(ref_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """
    Map each query timestamp to the nearest timestamp index in `ref_t`.
    Assumes `ref_t` is sorted.
    """
    idx = np.searchsorted(ref_t, query_t, side="left")
    idx = np.clip(idx, 0, len(ref_t) - 1)
    left = np.clip(idx - 1, 0, len(ref_t) - 1)
    choose_left = np.abs(query_t - ref_t[left]) < np.abs(ref_t[idx] - query_t)
    idx[choose_left] = left[choose_left]
    return idx


def compute_ate(gt_t: np.ndarray, gt_xyz: np.ndarray, est_t: np.ndarray, est_xyz: np.ndarray) -> Tuple[float, float, int]:
    """Return (mean_abs_error, rmse, num_pairs) using nearest timestamp matching."""
    if len(gt_t) == 0 or len(est_t) == 0:
        return float("nan"), float("nan"), 0
    gt_idx = _nearest_indices(gt_t, est_t)
    d = est_xyz - gt_xyz[gt_idx]
    dist = np.linalg.norm(d, axis=1)
    return float(np.mean(dist)), float(np.sqrt(np.mean(dist * dist))), int(len(dist))


def compute_ape(ref_t: np.ndarray, ref_xyz: np.ndarray, est_t: np.ndarray, est_xyz: np.ndarray) -> Tuple[float, float, int]:
    """Return APE(mean, rmse, num_pairs) using nearest timestamp matching."""
    if len(ref_t) == 0 or len(est_t) == 0:
        return float("nan"), float("nan"), 0
    ref_idx = _nearest_indices(ref_t, est_t)
    d = est_xyz - ref_xyz[ref_idx]
    dist = np.linalg.norm(d, axis=1)
    return float(np.mean(dist)), float(np.sqrt(np.mean(dist * dist))), int(len(dist))


def latest_pre_opt_run_dir(pre_root: Path) -> Path:
    """
    If `pre_root` contains subdirectories named with non-negative integers (0, 1, 2, …),
    return the one with the largest index (most recent dump batch). Otherwise return
    `pre_root` for a flat legacy layout (trajectory_*.txt directly under pre_root).
    """
    if not pre_root.is_dir():
        return pre_root
    numeric_runs: List[Tuple[int, Path]] = []
    for p in pre_root.iterdir():
        if p.is_dir() and p.name.isdigit():
            numeric_runs.append((int(p.name), p))
    if not numeric_runs:
        return pre_root
    return min(numeric_runs, key=lambda t: t[0])[1]


def detect_drone_ids(gt_dir: Path, pre_dir: Path, post_dir: Path) -> List[str]:
    gt_ids = {p.stem.replace("gt_log_", "") for p in gt_dir.glob("gt_log_*.txt")}
    pre_ids = {p.stem.replace("trajectory_", "") for p in pre_dir.glob("trajectory_*.txt")}
    post_ids = {p.stem.replace("trajectory_", "") for p in post_dir.glob("trajectory_*.txt")}
    return sorted(gt_ids & pre_ids & post_ids, key=lambda x: int(x) if x.isdigit() else x)


def main(
    gt_dir: Path = Path("test_data/multidrone"),
    pre_dir: Path = Path("test_results/multidrone/pre_opt_trajectories"),
    post_dir: Path = Path("test_results/multidrone/trajectories"),
    drone_ids: List[str] | None = None,
    xy_only: bool = False,
) -> None:
    pre_root = pre_dir
    pre_dir = pre_dir
    if pre_dir != pre_root:
        print(f"pre-opt: using latest run directory {pre_dir} (under {pre_root})")

    if drone_ids is None:
        drone_ids = detect_drone_ids(gt_dir, pre_dir, post_dir)

    if not drone_ids:
        raise RuntimeError("No common drone IDs found across gt/pre/post directories.")

    # 5 distinct colors (for 5 drones as requested). If more IDs are passed, colors repeat.
    colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple"]

    gt_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    pre_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    post_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for drone_id in drone_ids:
        gt_path = gt_dir / f"gt_log_{drone_id}.txt"
        pre_path = pre_dir / f"trajectory_{drone_id}.txt"
        post_path = post_dir / f"trajectory_{drone_id}.txt"

        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT file: {gt_path}")
        if not pre_path.exists():
            raise FileNotFoundError(f"Missing pre-opt file: {pre_path}")
        if not post_path.exists():
            raise FileNotFoundError(f"Missing post-opt file: {post_path}")

        gt_data[drone_id] = read_gt_trajectory(gt_path)
        pre_data[drone_id] = read_tum_trajectory(pre_path)
        post_data[drone_id] = read_tum_trajectory(post_path)

    # ATE report (per drone + overall)
    pre_all_dist: List[np.ndarray] = []
    post_all_dist: List[np.ndarray] = []
    ape_all_dist: List[np.ndarray] = []
    print("\nATE report (meters):")
    for drone_id in drone_ids:
        gt_t, gt_xyz = gt_data[drone_id]
        pre_t, pre_xyz = pre_data[drone_id]
        post_t, post_xyz = post_data[drone_id]

        pre_mean, pre_rmse, pre_n = compute_ate(gt_t, gt_xyz, pre_t, pre_xyz)
        post_mean, post_rmse, post_n = compute_ate(gt_t, gt_xyz, post_t, post_xyz)
        ape_mean, ape_rmse, ape_n = compute_ape(pre_t, pre_xyz, post_t, post_xyz)
        print(
            f"  Drone {drone_id}: "
            f"pre-opt mean={pre_mean:.6f}, rmse={pre_rmse:.6f} (n={pre_n}) | "
            f"post-opt mean={post_mean:.6f}, rmse={post_rmse:.6f} (n={post_n}) | "
            f"APE(pre,post) mean={ape_mean:.6f}, rmse={ape_rmse:.6f} (n={ape_n})"
        )

        # Store distances for global aggregate
        # (only meaningful if timestamps are synchronized; relies on strict indexing)
        if pre_n > 0:
            pre_idx = _nearest_indices(gt_t, pre_t)
            pre_all_dist.append(np.linalg.norm(pre_xyz - gt_xyz[pre_idx], axis=1))
        if post_n > 0:
            post_idx = _nearest_indices(gt_t, post_t)
            post_all_dist.append(np.linalg.norm(post_xyz - gt_xyz[post_idx], axis=1))
        if ape_n > 0:
            ape_idx = _nearest_indices(pre_t, post_t)
            ape_all_dist.append(np.linalg.norm(post_xyz - pre_xyz[ape_idx], axis=1))

    if pre_all_dist and post_all_dist and ape_all_dist:
        pre_concat = np.concatenate(pre_all_dist)
        post_concat = np.concatenate(post_all_dist)
        ape_concat = np.concatenate(ape_all_dist)
        print(
            "  Overall: "
            f"pre-opt mean={np.mean(pre_concat):.6f}, rmse={np.sqrt(np.mean(pre_concat**2)):.6f} | "
            f"post-opt mean={np.mean(post_concat):.6f}, rmse={np.sqrt(np.mean(post_concat**2)):.6f} | "
            f"APE(pre,post) mean={np.mean(ape_concat):.6f}, rmse={np.sqrt(np.mean(ape_concat**2)):.6f}"
        )

    if xy_only:
        fig, ax = plt.subplots(figsize=(10, 8))
        drone0_id = "0" if "0" in drone_ids else drone_ids[0]
        for i, drone_id in enumerate(drone_ids):
            c = colors[i % len(colors)]
            _, gt = gt_data[drone_id]
            _, pre = pre_data[drone_id]
            _, post = post_data[drone_id]

            ax.plot(gt[:, 0], gt[:, 1], "-", color=c, linewidth=2.0, label=f"Drone {drone_id} GT")
            is_drone0 = (drone_id == drone0_id)

            # GT start / finish markers for each drone (legend only for drone0)
            start_idx = 0
            end_idx = len(gt) - 1
            ax.scatter(
                gt[start_idx, 0],
                gt[start_idx, 1],
                s=40,
                marker="o",
                color=c,
                label="GT start (drone0)" if is_drone0 else None,
            )
            ax.scatter(
                gt[end_idx, 0],
                gt[end_idx, 1],
                s=40,
                marker="X",
                color=c,
                label="GT finish (drone0)" if is_drone0 else None,
            )

            # pre-opt / post-opt lines (legend only for drone0)
            ax.plot(
                pre[:, 0],
                pre[:, 1],
                "-",
                color=c,
                alpha=0.35,
                linewidth=2.0,
                label="pre-opt (drone0)" if is_drone0 else None,
            )
            ax.plot(
                post[:, 0],
                post[:, 1],
                ":",
                color=c,
                linewidth=2.2,
                label="post-opt (drone0)" if is_drone0 else None,
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Trajectory Comparison (XY)")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
    else:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        drone0_id = "0" if "0" in drone_ids else drone_ids[0]
        for i, drone_id in enumerate(drone_ids):
            c = colors[i % len(colors)]
            _, gt = gt_data[drone_id]
            _, pre = pre_data[drone_id]
            _, post = post_data[drone_id]

            ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], "-", color=c, linewidth=2.0, label=f"Drone {drone_id} GT")
            is_drone0 = (drone_id == drone0_id)

            start_idx = 0
            end_idx = len(gt) - 1
            ax.scatter(
                gt[start_idx, 0],
                gt[start_idx, 1],
                gt[start_idx, 2],
                s=40,
                marker="o",
                color=c,
                label="GT start (drone0)" if is_drone0 else None,
            )
            ax.scatter(
                gt[end_idx, 0],
                gt[end_idx, 1],
                gt[end_idx, 2],
                s=40,
                marker="X",
                color=c,
                label="GT finish (drone0)" if is_drone0 else None,
            )
            ax.plot(
                pre[:, 0],
                pre[:, 1],
                pre[:, 2],
                "-",
                color=c,
                alpha=0.35,
                linewidth=2.0,
                label="pre-opt (drone0)" if is_drone0 else None,
            )
            ax.plot(
                post[:, 0],
                post[:, 1],
                post[:, 2],
                ":",
                color=c,
                linewidth=2.2,
                label="post-opt (drone0)" if is_drone0 else None,
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Trajectory Comparison (3D)")

    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Edit these directly when you want to change paths quickly.
    # `pre_dir` is the pre_opt_trajectories root; the latest numeric subfolder (e.g. …/2/) is picked automatically.
    main(
        gt_dir=Path("test_data/multidrone2/long_traj/30_20lm_2midpoint/1"),
        pre_dir=Path("test_results/multidrone2/long_traj/30_20lm_2midpoint_lmf1_of1/1/pre_opt_trajectories/0"),
        post_dir=Path("test_results/multidrone2/long_traj/30_20lm_2midpoint_lmf1_of1/1/trajectories"),
        drone_ids=["0", "1", "2", "3", "4"],
        xy_only=False,
    )

