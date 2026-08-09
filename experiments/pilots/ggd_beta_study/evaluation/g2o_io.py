"""Minimal g2o reader for TUTORIAL slam2d pose graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PoseDict = Dict[int, Tuple[float, float, float]]


def read_tutorial_se2_poses(path: Path) -> PoseDict:
    """Parse TUTORIAL_VERTEX_SE2 lines from a g2o file."""
    poses: PoseDict = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("TUTORIAL_VERTEX_SE2"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            vertex_id = int(parts[1])
            x, y, theta = map(float, parts[2:5])
            poses[vertex_id] = (x, y, theta)
    if not poses:
        raise ValueError(f"No TUTORIAL_VERTEX_SE2 vertices found in {path}")
    return poses


def read_gt_poses_ordered(path: Path) -> List[Tuple[float, float, float]]:
    """
    Ground truth robot poses in simulator trajectory order.

    gt.g2o stores one TUTORIAL_VERTEX_SE2 per pose; vertex ids in the file may
    not match optimizer ids (landmarks share the global id space), so callers
    should align by index using the odometry pose chain from an estimate graph.
    """
    ordered: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("TUTORIAL_VERTEX_SE2"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            ordered.append((float(parts[2]), float(parts[3]), float(parts[4])))
    if not ordered:
        raise ValueError(f"No ground-truth poses found in {path}")
    return ordered


def read_pose_id_chain(path: Path, start_id: int = 0) -> List[int]:
    """
    Recover robot pose vertex ids in trajectory order from odometry edges.

    Landmark vertices use POINT_XY types; odometry connects consecutive poses only.
    """
    successor: Dict[int, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("TUTORIAL_EDGE_SE2 "):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            from_id = int(parts[1])
            to_id = int(parts[2])
            if from_id in successor and successor[from_id] != to_id:
                raise ValueError(
                    f"Non-linear odometry chain at vertex {from_id} in {path}"
                )
            successor[from_id] = to_id

    if start_id not in successor and start_id not in read_tutorial_se2_poses(path):
        raise ValueError(f"Start pose id {start_id} not found in {path}")

    chain = [start_id]
    while chain[-1] in successor:
        chain.append(successor[chain[-1]])
    return chain


def compute_translation_ape_vs_gt(
    gt_ordered: Sequence[Tuple[float, float, float]],
    est_poses: PoseDict,
    pose_id_chain: Sequence[int],
) -> Tuple[float, float, float, int]:
    """
    Absolute Pose Error (translation) vs ordered ground truth.

    Matches gt_ordered[i] to est_poses[pose_id_chain[i]].
    Returns (mean, rmse, max, num_pairs).
    """
    if len(gt_ordered) != len(pose_id_chain):
        raise ValueError(
            f"Ground truth length {len(gt_ordered)} != pose chain length {len(pose_id_chain)}"
        )

    errors: List[float] = []
    for gt_pose, vid in zip(gt_ordered, pose_id_chain):
        if vid not in est_poses:
            raise KeyError(f"Estimate missing pose vertex {vid}")
        gx, gy = gt_pose[0], gt_pose[1]
        ex, ey, _ = est_poses[vid]
        errors.append(float(np.hypot(ex - gx, ey - gy)))

    err = np.asarray(errors, dtype=float)
    return float(np.mean(err)), float(np.sqrt(np.mean(err * err))), float(np.max(err)), len(errors)


def per_pose_translation_errors(
    gt_path: Path, est_path: Path
) -> Tuple[np.ndarray, int]:
    """Per-pose translation errors (m) in trajectory order."""
    gt_ordered = read_gt_poses_ordered(gt_path)
    est_poses = read_tutorial_se2_poses(est_path)
    pose_chain = read_pose_id_chain(est_path, start_id=0)
    if len(gt_ordered) != len(pose_chain):
        raise ValueError(
            f"Ground truth length {len(gt_ordered)} != pose chain length {len(pose_chain)}"
        )
    errors = []
    for gt_pose, vid in zip(gt_ordered, pose_chain):
        if vid not in est_poses:
            raise KeyError(f"Estimate missing pose vertex {vid}")
        gx, gy = gt_pose[0], gt_pose[1]
        ex, ey, _ = est_poses[vid]
        errors.append(float(np.hypot(ex - gx, ey - gy)))
    err = np.asarray(errors, dtype=float)
    return err, len(err)


def compute_translation_ape_from_g2o(
    gt_path: Path, est_path: Path
) -> Tuple[float, float, float, int]:
    """APE vs gt.g2o using odometry ordering from the estimate graph."""
    err, n = per_pose_translation_errors(gt_path, est_path)
    return float(np.mean(err)), float(np.sqrt(np.mean(err * err))), float(np.max(err)), n


def compute_translation_mse_from_g2o(gt_path: Path, est_path: Path) -> Tuple[float, int]:
    """
    Mean squared translation error (m²) vs gt.g2o.

    Same pose pairing as APE; reports mean(err²) as in the legacy batch visualiser.
    """
    err, n = per_pose_translation_errors(gt_path, est_path)
    return float(np.mean(err * err)), n


def beta_output_stem(beta: float) -> str:
    """Match C++ outputTag for ggd runs (beta_6, beta_1000, ...)."""
    if abs(beta - round(beta)) < 1e-9:
        tag = str(int(round(beta)))
    else:
        tag = str(beta).replace(".", "p")
    return f"beta_{tag}"
