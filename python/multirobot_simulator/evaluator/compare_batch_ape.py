#!/usr/bin/env python3
"""
Compare batch APE (GT vs estimate) side-by-side for two result roots (e.g. with / without
GND) plus a shared GT batch root.

Only the first N runs are compared, with N = min(# numeric run subdirs in batch root,
# numeric run subdirs in batch_noGnd root). Run ids are those in the intersection of both
result roots and GT, sorted numerically (0, 1, 2, …); then the list is truncated to N.

Uses compute_ape() from plot_trajectory_comparison.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:
    from multirobot_simulator.evaluator.plot_trajectory_comparison import (
        compute_ape,
        read_gt_trajectory,
        read_tum_trajectory,
    )
except ModuleNotFoundError:
    # Allow direct execution: python .../evaluator/compare_batch_ape.py
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


def _format_batch_roots_debug(
    gt_root: Path,
    batch_root: Path,
    no_gnd_root: Path,
) -> str:
    """Human-readable report: absolute paths, existence, and numeric run id sets."""
    lines: List[str] = []

    def describe(label: str, p: Path) -> set[str]:
        abs_p = p.expanduser().resolve()
        if not p.exists():
            lines.append(f"  [{label}] MISSING (path does not exist):")
            lines.append(f"           {abs_p}")
            return set()
        if not p.is_dir():
            lines.append(f"  [{label}] EXISTS BUT IS NOT A DIRECTORY:")
            lines.append(f"           {abs_p}")
            return set()
        nums = set(_numeric_subdirs(p))
        non_numeric_dirs = sorted(
            x.name for x in p.iterdir() if x.is_dir() and not x.name.isdigit()
        )
        lines.append(f"  [{label}] ok — directory:")
        lines.append(f"           {abs_p}")
        lines.append(
            f"           numeric run subdirs: {len(nums)} → {sorted(nums, key=int)}"
        )
        if non_numeric_dirs:
            sample = non_numeric_dirs[:15]
            more = " ..." if len(non_numeric_dirs) > 15 else ""
            lines.append(
                f"           (non-numeric subdirs, ignored: {sample}{more})"
            )
        return nums

    lines.append("Batch APE directory diagnostics:")
    s_gt = describe("gt_batch_root", gt_root)
    s_b = describe("batch_root", batch_root)
    s_ng = describe("batch_noGnd_root", no_gnd_root)

    common = s_gt & s_b & s_ng
    lines.append(f"  Intersection of numeric run ids (all three): {sorted(common, key=int)}")

    if not s_gt and gt_root.exists() and gt_root.is_dir():
        lines.append("  Hint: gt root has no numeric subdirs (0,1,2,…); check layout.")
    if not s_b:
        lines.append(
            "  Hint: batch_root has no numeric runs — often missing directory or wrong "
            "path (e.g. typo batch_f2_nGnd vs batch_f2_noGnd)."
        )
    if not s_ng:
        lines.append(
            "  Hint: batch_noGnd_root has no numeric runs — path may be wrong or folder missing."
        )

    if s_gt and s_b and not (s_gt & s_b):
        lines.append(
            f"  gt ∩ batch is empty. Only in gt: {sorted(s_gt - s_b, key=int)[:15]}…"
            f" only in batch: {sorted(s_b - s_gt, key=int)[:15]}…"
        )
    if s_gt and s_ng and not (s_gt & s_ng):
        lines.append(
            f"  gt ∩ batch_noGnd is empty. Only in gt: {sorted(s_gt - s_ng, key=int)[:15]}…"
        )

    return "\n".join(lines)


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
    gt_batch_root: Path = Path("test_data/multidrone2/repeats/batch_f2_bounded"),
    batch_root: Path = Path("test_results/multidrone2/repeats/batch_f2_bounded"),
    batch_no_gnd_root: Path = Path("test_results/multidrone2/repeats/batch_f2_nGnd_bounded"),
    drone_ids: List[str] | None = None,
) -> None:
    ids_b = set(_numeric_subdirs(batch_root))
    ids_ng = set(_numeric_subdirs(batch_no_gnd_root))
    ids_gt = set(_numeric_subdirs(gt_batch_root))
    n_cap = min(len(ids_b), len(ids_ng))
    common = sorted(ids_b & ids_ng & ids_gt, key=lambda s: int(s))
    runs = common[:n_cap] if n_cap > 0 else []
    if not runs:
        raise RuntimeError(
            "No runs to compare: need non-empty intersection of numeric subdirs across "
            "gt / batch / batch_noGnd, after capping at N=min(#batch, #batch_noGnd).\n"
            + _format_batch_roots_debug(gt_batch_root, batch_root, batch_no_gnd_root)
        )
    print(
        f"Comparing first N={len(runs)} trajectories "
        f"(N_cap=min(batch={len(ids_b)}, batch_noGnd={len(ids_ng)})={n_cap}; "
        f"{len(common)} common with GT, numeric order {runs[0]}…{runs[-1]})."
    )

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
    print(f"  batch mean APE      = {np.nanmean(batch_arr):.3f} \pm {np.nanstd(batch_arr):.3f}")
    print(f"  batch_noGnd mean APE= {np.nanmean(no_gnd_arr):.3f} \pm {np.nanstd(no_gnd_arr):.3f}")

    valid_mask = np.isfinite(batch_arr) & np.isfinite(no_gnd_arr)
    valid_mask = valid_mask.astype(bool)
    paired_batch = batch_arr[valid_mask]
    paired_no_gnd = no_gnd_arr[valid_mask]
    
    print(f"\n GND results are better in {np.sum(paired_no_gnd > paired_batch)} out of {paired_batch.size} tests.")

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
    # Note: on disk the no-GND folder is typically `batch_f2_noGnd`, not `batch_f2_nGnd`.
    main(
        gt_batch_root=Path("test_data/multidrone2/long_traj/30_20lm_2midpoint"),
        batch_root=Path("test_results/multidrone2/long_traj/30_20lm_2midpoint_lmf2_of2_bounded"),
        batch_no_gnd_root=Path("test_results/multidrone2/long_traj/30_20lm_2midpoint_lmf2_of2_bounded_nGnd"),
    )

