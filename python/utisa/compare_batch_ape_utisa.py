#!/usr/bin/env python3
"""
Batch-style APE (2D) for UTIAS MR.CLAM: ground truth from `Robot*_Groundtruth.dat` vs TUM
estimates under each result root's `trajectories/` (optionally `run_id/trajectories/`).

Adapted from multirobot_simulator/evaluator/compare_batch_ape.py.

Parameters (globals):
  SIM_DURATION_SEC / DERIVE_SIM_DURATION / EXPERIMENT_JSON — same meaning as
  plot_trajectory_comparison_utisa.py (evaluation time window in simulation-relative seconds).

DISPLAY_DURATION_SEC is not used here (metrics only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    from mrclam_eval_common import (
        align_gt_to_first_sample,
        canonical_robot_id,
        crop_time_interval,
        derive_simulated_duration_sec,
        discover_robot_gt_paths,
        discover_result_robot_ids,
        duration_from_experiment_json,
        read_mrclam_groundtruth,
        read_tum_xy,
    )
except ModuleNotFoundError:
    from python.utisa.mrclam_eval_common import (  # type: ignore
        align_gt_to_first_sample,
        canonical_robot_id,
        crop_time_interval,
        derive_simulated_duration_sec,
        discover_robot_gt_paths,
        discover_result_robot_ids,
        duration_from_experiment_json,
        read_mrclam_groundtruth,
        read_tum_xy,
    )

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

try:
    from python.utisa.plot_trajectory_comparison_utisa import _nearest_indices
except ModuleNotFoundError:
    from plot_trajectory_comparison_utisa import _nearest_indices  # type: ignore


def compute_ape(ref_t: np.ndarray, ref_xy: np.ndarray, est_t: np.ndarray, est_xy: np.ndarray) -> Tuple[float, float, int]:
    if len(ref_t) == 0 or len(est_t) == 0:
        return float("nan"), float("nan"), 0
    ref_idx = _nearest_indices(ref_t, est_t)
    d = est_xy - ref_xy[ref_idx]
    dist = np.linalg.norm(d, axis=1)
    return float(np.mean(dist)), float(np.sqrt(np.mean(dist * dist))), int(len(dist))


# -------------------- Edit these --------------------
GT_DATASET_DIR = Path("test_data/utisa/MRCLAM7/MRCLAM_Dataset7")

BATCH_ROOT_A = Path("test_results/utisa_mrclam7_batch")
BATCH_ROOT_B = Path("test_results/utisa_mrclam7_batch_alt")  # set to same as A if no B

LABEL_A = "condition_A"
LABEL_B = "condition_B"

ROBOTS: List[str] = []  # [] = intersect GT and trajectories

SIM_DURATION_SEC: float | None = None
DERIVE_SIM_DURATION = True
EXPERIMENT_JSON: Path | None = Path(
    "test_results/utisa_mrclam7_batch/.batch_merged_configs/experiment_run_single.json"
)
# ----------------------------------------------------


def _numeric_subdirs_with_trajectories(root: Path) -> List[str]:
    if not root.is_dir():
        return []
    out: List[str] = []
    for p in root.iterdir():
        if p.is_dir() and p.name.isdigit() and (p / "trajectories").is_dir():
            out.append(p.name)
    return sorted(out, key=lambda s: int(s))


def resolve_trajectory_dir(batch_root: Path, run_id: str) -> Path:
    if run_id in ("", ".", "single"):
        d = batch_root / "trajectories"
        if d.is_dir():
            return d
        raise FileNotFoundError(f"No trajectories under {batch_root}")
    d = batch_root / run_id / "trajectories"
    if not d.is_dir():
        raise FileNotFoundError(d)
    return d


def list_runs_for_root(batch_root: Path) -> List[str]:
    nums = _numeric_subdirs_with_trajectories(batch_root)
    if nums:
        return nums
    if (batch_root / "trajectories").is_dir():
        return ["single"]
    return []


def _detect_robot_ids(gt_dataset: Path, traj_dir: Path) -> List[str]:
    gt_map = discover_robot_gt_paths(gt_dataset)
    post_ids = discover_result_robot_ids(traj_dir)
    canon_gt = {canonical_robot_id(k) for k in gt_map}
    common = sorted(canon_gt & set(post_ids), key=lambda x: int(x) if x.isdigit() else x)
    return common


def resolve_sim_duration(traj_dir: Path, robot_ids: List[str]) -> float | None:
    if SIM_DURATION_SEC is not None:
        return float(SIM_DURATION_SEC)
    if DERIVE_SIM_DURATION:
        paths = [traj_dir / f"trajectory_{rid}.txt" for rid in robot_ids]
        return derive_simulated_duration_sec(paths)
    if EXPERIMENT_JSON is not None:
        v = duration_from_experiment_json(EXPERIMENT_JSON)
        if v is not None:
            return float(v)
    return None


def run_ape_mean(
    gt_dataset: Path,
    traj_dir: Path,
    drone_ids: Optional[Iterable[str]] = None,
    sim_dur: Optional[float] = None,
) -> float:
    if not traj_dir.is_dir():
        return float("nan")
    ids = list(drone_ids) if drone_ids is not None else _detect_robot_ids(gt_dataset, traj_dir)
    if not ids:
        return float("nan")

    gt_map = discover_robot_gt_paths(gt_dataset)
    weighted_sum = 0.0
    total_n = 0
    for cid in ids:
        robot_key = next((k for k in gt_map if canonical_robot_id(k) == cid), None)
        if robot_key is None:
            continue
        gt_t, gt_xy, _ = read_mrclam_groundtruth(gt_map[robot_key])
        gt_t, gt_xy = align_gt_to_first_sample(gt_t, gt_xy)
        gt_t, gt_xy = crop_time_interval(gt_t, gt_xy, sim_dur)

        est_path = traj_dir / f"trajectory_{cid}.txt"
        if not est_path.is_file():
            continue
        est_t, est_xy = read_tum_xy(est_path)
        est_t, est_xy = crop_time_interval(est_t, est_xy, sim_dur)

        mean_err, _rmse, n = compute_ape(gt_t, gt_xy, est_t, est_xy)
        if n <= 0 or not np.isfinite(mean_err):
            continue
        weighted_sum += mean_err * n
        total_n += n
    if total_n == 0:
        return float("nan")
    return weighted_sum / total_n


def main() -> None:
    runs_a = list_runs_for_root(BATCH_ROOT_A)
    runs_b = list_runs_for_root(BATCH_ROOT_B)
    common = sorted(set(runs_a) & set(runs_b), key=lambda s: int(s) if s.isdigit() else s)
    if not common:
        raise RuntimeError(
            f"No common runs. A={runs_a} B={runs_b}\n"
            f"Expected numeric subdirs with trajectories/, or a flat trajectories/ folder (single)."
        )

    ids_arg: Optional[List[str]] = None
    if ROBOTS:
        ids_arg = [canonical_robot_id(r) for r in ROBOTS]

    print(f"Common runs: {common[:10]}{'…' if len(common) > 10 else ''} ({len(common)} total)")

    vals_a: List[float] = []
    vals_b: List[float] = []

    print("\nAPE mean (2D, meters) — GT vs estimate")
    print(f"run | {LABEL_A:^21} | {LABEL_B:^21}")
    print("----+-----------------------+-----------------------")

    for r in common:
        ta = resolve_trajectory_dir(BATCH_ROOT_A, r if r != "single" else "single")
        tb = resolve_trajectory_dir(BATCH_ROOT_B, r if r != "single" else "single")
        ids = ids_arg if ids_arg else _detect_robot_ids(GT_DATASET_DIR, ta)
        sim_dur_a = resolve_sim_duration(ta, ids)
        sim_dur_b = resolve_sim_duration(tb, ids)
        sim_dur = sim_dur_a if sim_dur_a is not None else sim_dur_b

        va = run_ape_mean(GT_DATASET_DIR, ta, ids, sim_dur)
        vb = run_ape_mean(GT_DATASET_DIR, tb, ids, sim_dur)
        vals_a.append(va)
        vals_b.append(vb)
        sa = f"{va:.6f}" if np.isfinite(va) else "nan"
        sb = f"{vb:.6f}" if np.isfinite(vb) else "nan"
        print(f"{r:>5} | {sa:>21} | {sb:>21}")

    a_arr = np.asarray(vals_a, dtype=float)
    b_arr = np.asarray(vals_b, dtype=float)
    print("\nOverall mean across runs:")
    print(f"  {LABEL_A}: {np.nanmean(a_arr):.6f} ± {np.nanstd(a_arr):.6f}")
    print(f"  {LABEL_B}: {np.nanmean(b_arr):.6f} ± {np.nanstd(b_arr):.6f}")

    ok = np.isfinite(a_arr) & np.isfinite(b_arr)
    if int(ok.sum()) >= 2 and scipy_stats is not None:
        t_stat, t_p = scipy_stats.ttest_rel(a_arr[ok], b_arr[ok], nan_policy="omit")
        print(f"  Paired t-test: t={t_stat:.6f}, p={t_p:.6e}")
    elif scipy_stats is None:
        print("  Install scipy for paired t-test.")


if __name__ == "__main__":
    main()
