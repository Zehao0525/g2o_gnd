#!/usr/bin/env python3
"""Visualize UTIAS MR.CLAM trajectories from a dataset folder."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# -------------------- Hardcoded config --------------------
# Update these globals directly, then run the script.
DATASET_DIR = Path("test_data/utisa/MRCLAM7/MRCLAM_Dataset7")
# Show first X seconds from global start time. Use None for full trajectory.
DURATION_SEC = 120.0
# Use [] for all robots, or e.g. ["Robot1", "Robot3"].
ROBOTS: List[str] = []
SHOW_ORIENTATION = False
# ---------------------------------------------------------


def _read_gt_file(path: Path) -> np.ndarray:
    rows: List[Tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            t, x, y, theta = map(float, parts[:4])
            rows.append((t, x, y, theta))
    if not rows:
        return np.zeros((0, 4), dtype=float)
    return np.asarray(rows, dtype=float)


def _discover_robot_gt(dataset_dir: Path) -> Dict[str, Path]:
    files = sorted(dataset_dir.glob("Robot*_Groundtruth.dat"))
    out: Dict[str, Path] = {}
    for p in files:
        name = p.name
        robot_name = name.split("_", 1)[0]  # Robot1, Robot2, ...
        out[robot_name] = p
    return out


def _parse_robot_selection(selected: List[str], available: List[str]) -> List[str]:
    if not selected:
        return available
    req = [s.strip() for s in selected if s.strip()]
    missing = [r for r in req if r not in available]
    if missing:
        raise ValueError(f"Unknown robots: {missing}. Available: {available}")
    return req


def main() -> None:
    dataset_dir = DATASET_DIR
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_dir}")

    robot_gt = _discover_robot_gt(dataset_dir)
    if not robot_gt:
        raise RuntimeError(f"No Robot*_Groundtruth.dat found in: {dataset_dir}")

    available = sorted(robot_gt.keys(), key=lambda s: int(s.replace("Robot", "")))
    selected = _parse_robot_selection(ROBOTS, available)

    data: Dict[str, np.ndarray] = {rid: _read_gt_file(robot_gt[rid]) for rid in selected}
    if any(arr.shape[0] == 0 for arr in data.values()):
        empty = [rid for rid, arr in data.items() if arr.shape[0] == 0]
        raise RuntimeError(f"Groundtruth file had no valid rows for: {empty}")

    # Global start time across selected robots.
    t0 = min(arr[0, 0] for arr in data.values())
    t_cut = None if DURATION_SEC is None else (t0 + max(DURATION_SEC, 0.0))

    plt.figure(figsize=(8, 8))
    for rid in selected:
        arr = data[rid]
        if t_cut is not None:
            arr = arr[arr[:, 0] <= t_cut]
        if arr.shape[0] == 0:
            continue
        x = arr[:, 1]
        y = arr[:, 2]
        theta = arr[:, 3]
        plt.plot(x, y, linewidth=1.6, label=f"{rid} ({arr.shape[0]} pts)")
        plt.scatter([x[0]], [y[0]], s=20, marker="o")
        plt.scatter([x[-1]], [y[-1]], s=20, marker="x")

        if SHOW_ORIENTATION:
            step = max(1, arr.shape[0] // 50)
            dx = np.cos(theta[::step]) * 0.12
            dy = np.sin(theta[::step]) * 0.12
            plt.quiver(
                x[::step],
                y[::step],
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.0025,
                alpha=0.6,
            )

    title = f"UTIAS trajectories: {dataset_dir.name}"
    if DURATION_SEC is not None:
        title += f" | first {DURATION_SEC:.1f}s"
    plt.title(title)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

