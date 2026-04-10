#!/usr/bin/env python3
"""Shared helpers for MR.CLAM (UTIAS) 2D trajectory evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

Point2 = Tuple[np.ndarray, np.ndarray]


def read_mrclam_groundtruth(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read `RobotN_Groundtruth.dat`: columns t [s], x, y, theta [rad]."""
    rows: List[Tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 4:
                continue
            t, x, y, th = map(float, parts[:4])
            rows.append((t, x, y, th))
    if not rows:
        raise ValueError(f"No ground truth rows in {path}")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1:3], arr[:, 3]


def read_tum_xy(path: Path) -> Point2:
    """TUM: t x y z qx qy qz qw -> use x,y; ignore z."""
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"TUM file needs >=4 columns: {path}")
    return data[:, 0], data[:, 1:3]


def read_landmark_groundtruth(path: Path) -> Dict[int, np.ndarray]:
    """Read `Landmark_Groundtruth.dat`: subject, x, y, ... -> {subject: [x, y]}."""
    out: Dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 3:
                continue
            sid = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            out[sid] = np.asarray([x, y], dtype=float)
    return out


def read_barcodes(path: Path) -> Dict[int, int]:
    """Read `Barcodes.dat`: subject, barcode -> {barcode: subject}."""
    out: Dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                continue
            subject = int(float(parts[0]))
            barcode = int(float(parts[1]))
            out[barcode] = subject
    return out


def align_gt_to_first_sample(t: np.ndarray, xy: np.ndarray) -> Point2:
    """MR.CLAM GT uses absolute epoch time; align to t_rel = t - t[0] (per robot, like the simulator)."""
    if len(t) == 0:
        return t, xy
    t0 = t[0]
    return t - t0, xy


def crop_time_interval(t: np.ndarray, xy: np.ndarray, t_max: Optional[float]) -> Point2:
    """Keep samples with t <= t_max (relative time). If t_max is None, no crop."""
    if t_max is None or not np.isfinite(t_max) or t_max <= 0:
        return t, xy
    m = t <= t_max + 1e-9
    return t[m], xy[m]


def discover_robot_gt_paths(dataset_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted(dataset_dir.glob("Robot*_Groundtruth.dat")):
        stem = p.stem  # Robot1_Groundtruth
        robot = stem.split("_", 1)[0]  # Robot1
        out[robot] = p
    return out


def canonical_robot_id(label: str) -> str:
    """Robot1 / robot1 -> 1 (matches C++ trajectory_1.txt)."""
    s = label.strip()
    if s.lower().startswith("robot") and len(s) > 5:
        return str(int(s[5:]))
    return str(int(s))


def discover_result_robot_ids(trajectories_dir: Path) -> List[str]:
    ids = []
    for p in trajectories_dir.glob("trajectory_*.txt"):
        stem = p.stem.replace("trajectory_", "")
        if stem:
            ids.append(stem)
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def latest_pre_opt_subdir(pre_opt_root: Path) -> Path:
    """Pick numeric subdir with largest index (e.g. pre_opt_trajectories/2)."""
    if not pre_opt_root.is_dir():
        return pre_opt_root
    best: Tuple[int, Path] | None = None
    for p in pre_opt_root.iterdir():
        if p.is_dir() and p.name.isdigit():
            k = int(p.name)
            if best is None or k > best[0]:
                best = (k, p)
    return best[1] if best else pre_opt_root


def resolve_pre_opt_trajectories_dir(
    pre_opt_root: Path,
    robot_canonical_ids: List[str],
    explicit_subdir: Optional[str] = None,
) -> Optional[Path]:
    """
    Directory that holds pre-opt `trajectory_{id}.txt` files.

    Layout matches UTISA/Multidrone: `pre_opt_trajectories/<batch>/` or a flat
    `pre_opt_trajectories/`. Returns None if no files exist (e.g. `debug_outputs: false`).
    """
    if not pre_opt_root.is_dir():
        return None

    def has_all(d: Path) -> bool:
        return all((d / f"trajectory_{rid}.txt").is_file() for rid in robot_canonical_ids)

    def has_any(d: Path) -> bool:
        return any((d / f"trajectory_{rid}.txt").is_file() for rid in robot_canonical_ids)

    candidates: List[Path] = []
    if explicit_subdir is not None:
        candidates.append(pre_opt_root / explicit_subdir)
    numeric = sorted(
        (p for p in pre_opt_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
        reverse=True,
    )
    candidates.extend(numeric)
    candidates.append(pre_opt_root)

    seen: set[str] = set()
    ordered: List[Path] = []
    for c in candidates:
        key = str(c.resolve())
        if key not in seen:
            seen.add(key)
            ordered.append(c)

    for c in ordered:
        if c.is_dir() and has_all(c):
            return c
    for c in ordered:
        if c.is_dir() and has_any(c):
            return c
    return None


def duration_from_experiment_json(path: Path) -> Optional[float]:
    """Read `Duration` or `duration` from experiment JSON (seconds)."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        j = json.load(f)
    v = j.get("Duration", j.get("duration"))
    if v is None:
        return None
    return float(v)


def derive_simulated_duration_sec(trajectory_paths: List[Path]) -> float:
    """
    Infer how long the SLAM run covered in simulation time.

    UTISA `saveTrajectoryTUM` writes timestamps in simulation-relative time (same frame as
    `Duration` in the experiment config). For each trajectory file, span = last(t) - first(t);
    we return the minimum span across robots so all compared trajectories share a common window.
    """
    spans: List[float] = []
    for p in trajectory_paths:
        if not p.is_file():
            continue
        t, _ = read_tum_xy(p)
        if len(t) < 2:
            continue
        spans.append(float(t[-1] - t[0]))
    if not spans:
        raise RuntimeError("Could not derive simulated duration: no valid trajectory files.")
    return min(spans)


def derive_simulated_duration_from_results(
    results_dir: Path, robot_ids: Optional[List[str]] = None
) -> float:
    """Use post-optimization `trajectories/trajectory_{id}.txt` under results_dir."""
    traj_dir = results_dir / "trajectories"
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"No trajectories directory: {traj_dir}")
    ids = robot_ids if robot_ids else discover_result_robot_ids(traj_dir)
    paths = [traj_dir / f"trajectory_{rid}.txt" for rid in ids]
    return derive_simulated_duration_sec(paths)
