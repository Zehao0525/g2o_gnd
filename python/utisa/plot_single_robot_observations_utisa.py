#!/usr/bin/env python3
"""
Visualize one robot's pre/post trajectory and its MR.CLAM observations.

- Shows robot GT, pre-opt, and post-opt trajectories.
- Plots landmark GT locations.
- Draws measurement rays from source robot pose to observed target:
  - landmark observations: orange
  - robot observations: cyan
- Uses the same simulation-window logic as other UTISA eval scripts:
  SIM_DURATION_SEC or (if None) derived from post trajectories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from mrclam_eval_common import (
    align_gt_to_first_sample,
    crop_time_interval,
    derive_simulated_duration_from_results,
    duration_from_experiment_json,
    read_barcodes,
    read_landmark_groundtruth,
    read_mrclam_groundtruth,
    read_tum_xy,
    resolve_pre_opt_trajectories_dir,
)

# -------------------- Edit these --------------------
DATASET_DIR = Path("test_data/utisa/MRCLAM7/MRCLAM_Dataset7")
RESULTS_DIR = Path("test_results/utisa_mrclam7_batch")
ROBOT_ID = 2  # 1..5
PRE_OPT_SUBDIR = None

SIM_DURATION_SEC: float | None = None
DERIVE_SIM_DURATION = True
EXPERIMENT_JSON: Path | None = Path(
    "test_results/utisa_mrclam7_batch/.batch_merged_configs/experiment_run_single.json"
)

# Visual clutter control
MAX_RAYS = 1500  # sample rays if there are more
# ----------------------------------------------------


def _nearest_index(ref_t: np.ndarray, q: float) -> int:
    i = int(np.searchsorted(ref_t, q, side="left"))
    if i <= 0:
        return 0
    if i >= len(ref_t):
        return len(ref_t) - 1
    return i if abs(ref_t[i] - q) < abs(q - ref_t[i - 1]) else i - 1


def _read_measurements(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=float, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"Measurement file needs 4 columns: {path}")
    return data[:, :4]  # t, barcode, range, bearing


def _read_tum_xy_yaw(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read TUM trajectory and return (t, xy, yaw)."""
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 8:
        raise ValueError(f"TUM file needs 8 columns: {path}")
    t = data[:, 0]
    xy = data[:, 1:3]
    qx = data[:, 4]
    qy = data[:, 5]
    qz = data[:, 6]
    qw = data[:, 7]
    # Standard quaternion->yaw (Z-axis) conversion.
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return t, xy, yaw


def _read_estimated_landmarks(path: Path) -> np.ndarray:
    """Read exported landmarks_{robot}.txt -> array Nx2."""
    if not path.is_file():
        return np.zeros((0, 2), dtype=float)
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        return np.zeros((0, 2), dtype=float)
    return data[:, 1:3]


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
    print("SIM_DURATION_SEC is None and no duration inferred; using full trajectories.")
    return None


def main() -> None:
    rid = int(ROBOT_ID)
    rid_str = str(rid)

    # --- Ground truth ---
    src_gt_path = DATASET_DIR / f"Robot{rid}_Groundtruth.dat"
    src_meas_path = DATASET_DIR / f"Robot{rid}_Measurement.dat"
    barcodes_path = DATASET_DIR / "Barcodes.dat"
    lm_path = DATASET_DIR / "Landmark_Groundtruth.dat"
    if not src_gt_path.is_file() or not src_meas_path.is_file():
        raise FileNotFoundError(f"Missing Robot{rid}_Groundtruth.dat or Robot{rid}_Measurement.dat")
    if not barcodes_path.is_file() or not lm_path.is_file():
        raise FileNotFoundError("Missing Barcodes.dat or Landmark_Groundtruth.dat")

    src_t_abs, src_xy_abs, src_th_abs = read_mrclam_groundtruth(src_gt_path)
    src_t, src_xy = align_gt_to_first_sample(src_t_abs, src_xy_abs)
    src_th = src_th_abs

    lm_gt = read_landmark_groundtruth(lm_path)
    barcode_to_subject = read_barcodes(barcodes_path)
    meas = _read_measurements(src_meas_path)
    meas_t_rel = meas[:, 0] - src_t_abs[0]

    # --- Trajectories ---
    post_path = RESULTS_DIR / "trajectories" / f"trajectory_{rid_str}.txt"
    if not post_path.is_file():
        raise FileNotFoundError(post_path)
    post_t, post_xy, post_yaw = _read_tum_xy_yaw(post_path)

    pre_root = RESULTS_DIR / "pre_opt_trajectories"
    pre_dir = resolve_pre_opt_trajectories_dir(pre_root, [rid_str], explicit_subdir=str(PRE_OPT_SUBDIR) if PRE_OPT_SUBDIR is not None else None)
    pre_t = np.array([], dtype=float)
    pre_xy = np.zeros((0, 2), dtype=float)
    if pre_dir is not None:
        pp = pre_dir / f"trajectory_{rid_str}.txt"
        if pp.is_file():
            pre_t, pre_xy = read_tum_xy(pp)

    sim_dur = resolve_sim_duration(RESULTS_DIR, [rid_str])
    src_t_p, src_xy_p = crop_time_interval(src_t, src_xy, sim_dur)
    post_t_p, post_xy_p = crop_time_interval(post_t, post_xy, sim_dur)
    pre_t_p, pre_xy_p = crop_time_interval(pre_t, pre_xy, sim_dur)
    est_lm_xy = _read_estimated_landmarks(RESULTS_DIR / "landmarks" / f"landmarks_{rid_str}.txt")

    # --- Build observation rays ---
    m_mask = np.ones(len(meas_t_rel), dtype=bool)
    if sim_dur is not None:
        m_mask &= meas_t_rel <= sim_dur + 1e-9
    idxs = np.where(m_mask)[0]
    if len(idxs) > MAX_RAYS:
        keep = np.linspace(0, len(idxs) - 1, MAX_RAYS, dtype=int)
        idxs = idxs[keep]

    lm_rays_by_id: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    rb_rays: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in idxs:
        t = float(meas_t_rel[k])
        barcode = int(round(meas[k, 1]))
        rng = float(meas[k, 2])
        brg = float(meas[k, 3])
        subject = barcode_to_subject.get(barcode)
        if subject is None:
            continue
        si = _nearest_index(post_t, t)
        p_src = post_xy[si]
        th_src = float(post_yaw[si])

        # Draw measured ray directly from source GT pose using logged (range, bearing).
        th_world = th_src + brg
        p_tip = p_src + np.asarray([rng * np.cos(th_world), rng * np.sin(th_world)], dtype=float)

        if 6 <= subject <= 20:
            lm_rays_by_id.setdefault(subject, []).append((p_src, p_tip))
        elif 1 <= subject <= 5 and subject != rid:
            rb_rays.append((p_src, p_tip))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Landmarks
    lm_ids = sorted(lm_gt.keys())
    lm_xy = np.vstack([lm_gt[i] for i in lm_ids])
    ax.scatter(lm_xy[:, 0], lm_xy[:, 1], marker="x", c="k", s=35, alpha=0.8, label="Landmarks GT")
    if len(est_lm_xy) > 0:
        ax.scatter(
            est_lm_xy[:, 0],
            est_lm_xy[:, 1],
            marker="o",
            facecolors="none",
            edgecolors="tab:purple",
            s=28,
            linewidths=1.0,
            alpha=0.9,
            label="Landmarks est (post)",
        )

    # Rays first (behind trajectories)
    if lm_rays_by_id:
        cmap = plt.cm.get_cmap("tab20", max(len(lm_rays_by_id), 1))
        for ci, lm_id in enumerate(sorted(lm_rays_by_id.keys())):
            c = cmap(ci)
            segs = lm_rays_by_id[lm_id]
            for j, (a, b) in enumerate(segs):
                ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    color=c,
                    alpha=0.22,
                    linewidth=0.9,
                    label=f"LM obs id={lm_id}" if j == 0 else None,
                )
    for i, (a, b) in enumerate(rb_rays):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:cyan", alpha=0.16, linewidth=0.8,
                label="Robot observations" if i == 0 else None)

    # Trajectories (different shades of green)
    gt_green = (0.05, 0.45, 0.10)
    pre_green = (0.20, 0.65, 0.25)
    post_green = (0.45, 0.80, 0.45)
    ax.plot(src_xy_p[:, 0], src_xy_p[:, 1], "-", color=gt_green, linewidth=2.2, label=f"Robot {rid} GT")
    if len(pre_xy_p) > 0:
        ax.plot(pre_xy_p[:, 0], pre_xy_p[:, 1], "-", color=pre_green, alpha=0.75, linewidth=2.0, label="Pre-opt")
    ax.plot(post_xy_p[:, 0], post_xy_p[:, 1], ":", color=post_green, linewidth=2.4, label="Post-opt")

    title = f"Robot {rid}: pre/post + observations"
    if sim_dur is not None:
        title += f" | <= {sim_dur:.1f}s"
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
