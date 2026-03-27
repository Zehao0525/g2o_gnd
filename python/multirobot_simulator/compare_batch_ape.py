#!/usr/bin/env python3
"""
Compare batch APE (GT vs estimate) side-by-side for:
  - test_results/multidrone/batch
  - test_results/multidrone/batch_noGnd

Uses compute_ape() from plot_trajectory_comparison.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from plot_trajectory_comparison import compute_ape, read_gt_trajectory, read_tum_trajectory

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


def _numeric_subdirs(root: Path) -> List[str]:
    if not root.exists():
        return []
    out: List[str] = []
    for p in root.iterdir():
        if p.is_dir() and p.name.isdigit():
            out.append(p.name)
    return sorted(out, key=lambda s: int(s))


def _detect_drone_ids(gt_run_dir: Path, est_run_dir: Path) -> List[str]:
    gt_ids = {p.stem.replace("gt_log_", "") for p in gt_run_dir.glob("gt_log_*.txt")}
    est_ids = {p.stem.replace("trajectory_", "") for p in est_run_dir.glob("trajectory_*.txt")}
    common = gt_ids & est_ids
    return sorted(common, key=lambda x: int(x) if x.isdigit() else x)


def _run_ape_mean(
    gt_root: Path,
    est_root: Path,
    run_id: str,
    drone_ids: Iterable[str] | None = None,
) -> float:
    gt_run = gt_root / run_id
    est_run = est_root / run_id / "trajectories"
    if not gt_run.exists() or not est_run.exists():
        return float("nan")

    ids = list(drone_ids) if drone_ids is not None else _detect_drone_ids(gt_run, est_run)
    if not ids:
        return float("nan")

    weighted_sum = 0.0
    total_n = 0
    for did in ids:
        gt_t, gt_xyz = read_gt_trajectory(gt_run / f"gt_log_{did}.txt")
        est_t, est_xyz = read_tum_trajectory(est_run / f"trajectory_{did}.txt")
        mean_err, _rmse, n = compute_ape(gt_t, gt_xyz, est_t, est_xyz)
        if n <= 0 or not np.isfinite(mean_err):
            continue
        weighted_sum += mean_err * n
        total_n += n
    if total_n == 0:
        return float("nan")
    return weighted_sum / total_n


def main(
    gt_batch_root: Path = Path("test_data/multidrone/batch_f1"),
    batch_root: Path = Path("test_results/multidrone/batch_f1"),
    batch_no_gnd_root: Path = Path("test_results/multidrone/batch_f1_noGnd"),
    drone_ids: List[str] | None = None,
) -> None:
    runs = sorted(
        set(_numeric_subdirs(gt_batch_root))
        & set(_numeric_subdirs(batch_root))
        & set(_numeric_subdirs(batch_no_gnd_root)),
        key=lambda s: int(s),
    )
    if not runs:
        raise RuntimeError("No common numeric run directories found across gt/batch/batch_noGnd roots.")

    batch_vals: List[float] = []
    no_gnd_vals: List[float] = []

    print("\nAPE mean (GT vs estimate), side-by-side")
    print("run | batch             | batch_noGnd")
    print("----+-------------------+-------------------")
    for r in runs:
        v_batch = _run_ape_mean(gt_batch_root, batch_root, r, drone_ids)
        v_nognd = _run_ape_mean(gt_batch_root, batch_no_gnd_root, r, drone_ids)
        batch_vals.append(v_batch)
        no_gnd_vals.append(v_nognd)
        b_str = f"{v_batch:.6f}" if np.isfinite(v_batch) else "nan"
        n_str = f"{v_nognd:.6f}" if np.isfinite(v_nognd) else "nan"
        print(f"{int(r):>3d} | {b_str:>17} | {n_str:>17}")

    batch_arr = np.asarray(batch_vals, dtype=float)
    no_gnd_arr = np.asarray(no_gnd_vals, dtype=float)
    print("\nOverall mean across runs:")
    print(f"  batch mean APE      = {np.nanmean(batch_arr):.6f}")
    print(f"  batch_noGnd mean APE= {np.nanmean(no_gnd_arr):.6f}")

    valid_mask = np.isfinite(batch_arr) & np.isfinite(no_gnd_arr)
    valid_mask = valid_mask.astype(bool)
    paired_batch = batch_arr[valid_mask]
    paired_no_gnd = no_gnd_arr[valid_mask]

    print("\nPaired significance tests (batch vs batch_noGnd):")
    print(f"  valid paired runs   = {paired_batch.size}")
    if paired_batch.size < 2:
        print("  Not enough paired samples for significance testing.")
        return

    if scipy_stats is None:
        print("  SciPy not installed. Install scipy to run paired t-test and Wilcoxon.")
        return

    # H0: mean(batch - no_gnd) == 0
    # H1 (two-sided): mean(batch - no_gnd) != 0
    t_stat, t_p = scipy_stats.ttest_rel(paired_batch, paired_no_gnd, nan_policy="omit")
    print(f"  Paired t-test       : t={t_stat:.6f}, p={t_p:.6e} (two-sided)")

    # Wilcoxon signed-rank test for paired samples.
    # zero_method='wilcox' discards exact ties.
    try:
        w_res = scipy_stats.wilcoxon(
            paired_batch,
            paired_no_gnd,
            zero_method="wilcox",
            alternative="two-sided",
        )
        print(f"  Wilcoxon signed-rank: W={w_res.statistic:.6f}, p={w_res.pvalue:.6e} (two-sided)")
    except ValueError as e:
        print(f"  Wilcoxon signed-rank: unavailable ({e})")


if __name__ == "__main__":
    main()

