#!/usr/bin/env python3
"""
Generate a GT-derived MR.CLAM-like dataset.

Output layout (default):
  test_data/utisa/MRCLAM7_gt/MARCLAM_Dataset7/
    - Barcodes.dat                      (copied)
    - Landmark_Groundtruth.dat          (copied)
    - Robot{N}_Groundtruth.dat          (copied)
    - Robot{N}_Odometry.dat             (derived from Robot{N}_Groundtruth.dat)
    - Robot{N}_Measurement.dat          (derived from robot+landmark ground truth)

Rules:
  - Odometry timestamps exactly match unique Robot{N}_Groundtruth timestamps.
  - Robot/landmark measurements are synthesized at each robot GT timestamp.
  - For robot-to-robot measurements, target pose uses nearest GT timestamp.
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_txt(path: Path, cols: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float, comments="#")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < cols:
        raise ValueError(f"{path} has {arr.shape[1]} cols, expected >= {cols}")
    return arr


def _wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class RobotGT:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    th: np.ndarray


def read_robot_gt(path: Path) -> RobotGT:
    arr = _load_txt(path, cols=4)
    # Ensure strictly nondecreasing timestamps and unique timestamp rows.
    t = arr[:, 0]
    order = np.argsort(t, kind="stable")
    arr = arr[order]
    t_sorted = arr[:, 0]
    unique_mask = np.ones_like(t_sorted, dtype=bool)
    unique_mask[1:] = t_sorted[1:] > t_sorted[:-1]
    arr = arr[unique_mask]
    return RobotGT(t=arr[:, 0], x=arr[:, 1], y=arr[:, 2], th=arr[:, 3])


def read_barcodes(path: Path) -> Dict[int, int]:
    arr = _load_txt(path, cols=2)
    # File format: subject barcode
    subject_to_barcode: Dict[int, int] = {}
    for r in arr:
        subject_to_barcode[int(r[0])] = int(r[1])
    return subject_to_barcode


def read_landmarks(path: Path) -> Dict[int, Tuple[float, float]]:
    arr = _load_txt(path, cols=3)
    out: Dict[int, Tuple[float, float]] = {}
    for r in arr:
        out[int(r[0])] = (float(r[1]), float(r[2]))
    return out


def nearest_index(times: np.ndarray, q: float) -> int:
    i = int(np.searchsorted(times, q, side="left"))
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    if abs(times[i] - q) < abs(q - times[i - 1]):
        return i
    return i - 1


def derive_odometry(gt: RobotGT) -> np.ndarray:
    n = len(gt.t)
    out = np.zeros((n, 3), dtype=float)
    out[:, 0] = gt.t
    if n <= 1:
        return out
    dt = np.diff(gt.t)
    dx_w = np.diff(gt.x)
    dy_w = np.diff(gt.y)
    dth = _wrap_angle(np.diff(gt.th))
    th_prev = gt.th[:-1]

    # Express translational velocity in robot body frame at t_{k-1}
    c = np.cos(th_prev)
    s = np.sin(th_prev)
    vx = (c * dx_w + s * dy_w) / dt
    wz = dth / dt

    # Keep first sample at zero; fill derived values on subsequent timestamps.
    out[1:, 1] = vx
    out[1:, 2] = wz
    return out


def synth_measurements_for_robot(
    rid: int,
    robots: Dict[int, RobotGT],
    landmarks: Dict[int, Tuple[float, float]],
    subject_to_barcode: Dict[int, int],
) -> np.ndarray:
    src = robots[rid]
    rows: List[Tuple[float, int, float, float]] = []

    robot_subjects = sorted(robots.keys())
    landmark_subjects = sorted(landmarks.keys())

    for k, t in enumerate(src.t):
        xi = src.x[k]
        yi = src.y[k]
        thi = src.th[k]
        c = math.cos(thi)
        s = math.sin(thi)

        def emit_for_point(subject_id: int, xw: float, yw: float) -> None:
            dx = xw - xi
            dy = yw - yi
            # World -> body transform: R(theta)^T * d
            xb = c * dx + s * dy
            yb = -s * dx + c * dy
            r = math.hypot(xb, yb)
            b = math.atan2(yb, xb)
            barcode = subject_to_barcode.get(subject_id)
            if barcode is None:
                raise KeyError(f"No barcode for subject {subject_id}")
            rows.append((float(t), int(barcode), float(r), float(b)))

        for subj in robot_subjects:
            if subj == rid:
                continue
            tgt = robots[subj]
            j = nearest_index(tgt.t, t)
            emit_for_point(subj, float(tgt.x[j]), float(tgt.y[j]))

        for subj in landmark_subjects:
            lx, ly = landmarks[subj]
            emit_for_point(subj, lx, ly)

    return np.asarray(rows, dtype=float)


def write_odometry(path: Path, odom: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for t, v, w in odom:
            f.write(f"{t:.6f} {v:.9f} {w:.9f}\n")


def write_measurements(path: Path, meas: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for t, barcode, r, b in meas:
            f.write(f"{t:.6f} {int(round(barcode))} {r:.9f} {b:.9f}\n")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("test_data/utisa/MRCLAM7/MRCLAM_Dataset7"),
        help="Source MR.CLAM dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_data/utisa/MRCLAM7_gt/MARCLAM_Dataset7"),
        help="Output generated dataset directory.",
    )
    parser.add_argument(
        "--robots",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated robot subject ids to generate.",
    )
    args = parser.parse_args()

    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir
    robot_ids = [int(x.strip()) for x in args.robots.split(",") if x.strip()]
    if not robot_ids:
        raise ValueError("No robots selected.")

    out_dir.mkdir(parents=True, exist_ok=True)

    barcodes_src = in_dir / "Barcodes.dat"
    lms_src = in_dir / "Landmark_Groundtruth.dat"
    if not barcodes_src.is_file() or not lms_src.is_file():
        raise FileNotFoundError("Missing Barcodes.dat or Landmark_Groundtruth.dat in input dir.")

    subject_to_barcode = read_barcodes(barcodes_src)
    landmarks = read_landmarks(lms_src)

    copy_if_exists(barcodes_src, out_dir / "Barcodes.dat")
    copy_if_exists(lms_src, out_dir / "Landmark_Groundtruth.dat")

    robots: Dict[int, RobotGT] = {}
    for rid in robot_ids:
        gt_src = in_dir / f"Robot{rid}_Groundtruth.dat"
        if not gt_src.is_file():
            raise FileNotFoundError(gt_src)
        gt = read_robot_gt(gt_src)
        robots[rid] = gt
        shutil.copy2(gt_src, out_dir / gt_src.name)

    for rid in robot_ids:
        gt = robots[rid]
        odom = derive_odometry(gt)
        meas = synth_measurements_for_robot(rid, robots, landmarks, subject_to_barcode)
        write_odometry(out_dir / f"Robot{rid}_Odometry.dat", odom)
        write_measurements(out_dir / f"Robot{rid}_Measurement.dat", meas)
        print(
            f"Robot{rid}: GT={len(gt.t)} samples | "
            f"Odom={len(odom)} rows | Meas={len(meas)} rows"
        )

    print(f"Generated dataset at: {out_dir}")


if __name__ == "__main__":
    main()
