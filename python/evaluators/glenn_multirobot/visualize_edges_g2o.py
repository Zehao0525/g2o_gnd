#!/usr/bin/env python3
"""
Visualize odometry edges from a bot edges.g2o file.

Reads EDGE_SE3:QUAT lines, chains relative measurements into a trajectory,
and optionally overlays vertices.g2o / gt*.tum from the same directory.

Run from repo root:
  python3 python/evaluators/glenn_multirobot/visualize_edges_g2o.py
  python3 python/evaluators/glenn_multirobot/visualize_edges_g2o.py test_data/test1_new_data/bot1/edges.g2o
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

Pose = Tuple[float, float, float]  # x, y, theta


def _se3_from_tx_quat(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = [tx, ty, tz]
    return transform


def _se3_from_edge_tokens(tokens: Sequence[str]) -> np.ndarray:
    tx, ty, tz = map(float, tokens[3:6])
    qx, qy, qz, qw = map(float, tokens[6:10])
    return _se3_from_tx_quat(tx, ty, tz, qx, qy, qz, qw)


def _pose_from_matrix(transform: np.ndarray) -> Pose:
    x, y = float(transform[0, 3]), float(transform[1, 3])
    theta = float(np.arctan2(transform[1, 0], transform[0, 0]))
    return x, y, theta


def parse_se3_edges(path: Path) -> List[Tuple[int, int, np.ndarray]]:
    edges: List[Tuple[int, int, np.ndarray]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("EDGE_SE3:QUAT"):
                continue
            tokens = line.split()
            edges.append((int(tokens[1]), int(tokens[2]), _se3_from_edge_tokens(tokens)))
    return edges


def parse_se3_vertices(path: Path) -> Dict[int, Pose]:
    poses: Dict[int, Pose] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("VERTEX_SE3:QUAT"):
                continue
            tokens = line.split()
            vid = int(tokens[1])
            tx, ty, tz = map(float, tokens[2:5])
            qx, qy, qz, qw = map(float, tokens[5:9])
            transform = _se3_from_tx_quat(tx, ty, tz, qx, qy, qz, qw)
            poses[vid] = _pose_from_matrix(transform)
    return poses


def chain_odometry(edges: Sequence[Tuple[int, int, np.ndarray]]) -> Dict[int, Pose]:
    if not edges:
        return {}

    by_from: Dict[int, Tuple[int, np.ndarray]] = {i: (j, t) for i, j, t in edges}
    root = min(by_from)
    world: Dict[int, np.ndarray] = {root: np.eye(4)}

    ordered = [root]
    current = root
    while current in by_from:
        nxt, rel = by_from[current]
        world[nxt] = world[current] @ rel
        ordered.append(nxt)
        current = nxt

    return {vid: _pose_from_matrix(transform) for vid, transform in world.items()}


def parse_tum_xy(path: Path, stride: int = 50) -> List[Pose]:
    poses: List[Pose] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx % stride != 0:
                continue
            tokens = line.split()
            x, y = float(tokens[1]), float(tokens[2])
            qx, qy, qz, qw = map(float, tokens[4:8])
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            theta = float(np.arctan2(rot[1, 0], rot[0, 0]))
            poses.append((x, y, theta))
    if not poses:
        return poses

    x0, y0, t0 = poses[0]
    c, s = np.cos(-t0), np.sin(-t0)
    aligned: List[Pose] = []
    for x, y, theta in poses:
        dx, dy = x - x0, y - y0
        aligned.append((
            dx * c - dy * s,
            dx * s + dy * c,
            float(np.arctan2(np.sin(theta - t0), np.cos(theta - t0))),
        ))
    return aligned


def _plot_trajectory(
    ax: plt.Axes,
    poses: Sequence[Pose],
    *,
    color: str,
    label: str,
    linewidth: float = 1.8,
    alpha: float = 1.0,
    linestyle: str = "-",
) -> None:
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    ax.plot(xs, ys, color=color, label=label, linewidth=linewidth, alpha=alpha, linestyle=linestyle)


def _edge_step_lengths(edges: Sequence[Tuple[int, int, np.ndarray]]) -> np.ndarray:
    return np.array([float(np.hypot(t[0, 3], t[1, 3])) for _, _, t in edges])


def plot_edges(
    edges_path: Path,
    vertices_path: Optional[Path] = None,
    gt_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    edges = parse_se3_edges(edges_path)
    chained = chain_odometry(edges)
    chained_sorted = [chained[vid] for vid in sorted(chained)]

    vertices: Dict[int, Pose] = {}
    if vertices_path and vertices_path.is_file():
        vertices = parse_se3_vertices(vertices_path)
        vertex_sorted = [vertices[vid] for vid in sorted(vertices)]

    gt_poses: List[Pose] = []
    if gt_path and gt_path.is_file():
        gt_poses = parse_tum_xy(gt_path)

    step_lengths = _edge_step_lengths(edges)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    _plot_trajectory(ax, chained_sorted, color="#1565c0", label="chained odometry")
    if vertices:
        _plot_trajectory(ax, vertex_sorted, color="#2e7d32", label="vertices.g2o", linestyle="--", alpha=0.85)
    if gt_poses:
        _plot_trajectory(ax, gt_poses, color="#c62828", label="gt (aligned, subsampled)", linestyle=":", alpha=0.8)
    ax.set_title(f"Trajectory from {edges_path.name}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(step_lengths, color="#6a1b9a", linewidth=1.0)
    ax.set_title("Odometry step length per edge")
    ax.set_xlabel("edge index")
    ax.set_ylabel("|translation| [m]")
    ax.grid(True, alpha=0.3)

    bot_dir = edges_path.parent.name
    fig.suptitle(f"{bot_dir}: {len(edges)} odometry edges, {len(chained)} poses")
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Edges: {len(edges)}")
    print(f"Chained poses: {len(chained)}")
    if vertices:
        print(f"Vertices: {len(vertices)}")
    print(
        "Step length [m]: "
        f"mean={step_lengths.mean():.4f}, max={step_lengths.max():.4f}, min={step_lengths.min():.4f}"
    )


def _default_sibling(edges_path: Path, name: str) -> Optional[Path]:
    candidate = edges_path.parent / name
    return candidate if candidate.is_file() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize bot odometry edges.g2o")
    parser.add_argument(
        "edges",
        nargs="?",
        type=Path,
        default=Path("test_data/test1_new_data/bot1/edges.g2o"),
        help="Path to edges.g2o",
    )
    parser.add_argument(
        "--vertices",
        type=Path,
        default=None,
        help="Optional vertices.g2o overlay (default: sibling of edges file)",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Optional TUM ground-truth overlay (default: sibling gt*.tum)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional PNG output path",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edges_path = args.edges.resolve()
    if not edges_path.is_file():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    vertices_path = args.vertices or _default_sibling(edges_path, "vertices.g2o")
    if args.gt:
        gt_path = args.gt
    else:
        gt_path = _default_sibling(edges_path, "gt1.tum") or _default_sibling(edges_path, "gt0.tum")

    plot_edges(
        edges_path,
        vertices_path=vertices_path,
        gt_path=gt_path,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
