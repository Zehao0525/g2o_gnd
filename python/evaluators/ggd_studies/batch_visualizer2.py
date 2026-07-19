#!/usr/bin/env python3
"""
Adapted from repo-root batch_visualizer2.py for tutorial_w_references outputs.

Default results directory:
  test_results/gnd_beta_study/references_robust_kernels

Files per test_k/:
  twb_gt.g2o, twb_before.g2o, twb_{gauss,gnd,huber,...}.g2o

Run full statistical evaluation (APE + MSE, kernel ranking, seed analysis):
  python3 python/evaluators/gnd_studies/batch_visualizer2.py

Trajectory overlay for one seed (optional):
  python3 python/evaluators/gnd_studies/batch_visualizer2.py --plot-test 0 --plot-kernel gnd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_references import OUTPUT_DIR, run_evaluation
from g2o_io import read_gt_poses_ordered, read_pose_id_chain, read_tutorial_se2_poses

DEFAULT_RESULTS_ROOT = Path("test_results/simulated_gps_like_test/30_p2_sig25_w_robust_kernels")
KERNEL_FILE_ALIASES = {"gaussian": "gauss", "gnd": "gnd", "huber": "huber", "tukey": "tukey"}


def _trajectory_xy(path: Path) -> np.ndarray:
    """Pose (x,y) in trajectory order."""
    gt_ordered = read_gt_poses_ordered(path) if "gt" in path.name else None
    if gt_ordered is not None and "gt" in path.stem:
        return np.array([[p[0], p[1]] for p in gt_ordered], dtype=float)

    poses = read_tutorial_se2_poses(path)
    chain = read_pose_id_chain(path, start_id=0)
    return np.array([[poses[vid][0], poses[vid][1]] for vid in chain], dtype=float)


def plot_kernel_vs_baseline(
    results_root: Path,
    test_idx: int,
    kernel: str,
    baseline: str = "gaussian",
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    test_dir = results_root / f"test_{test_idx}"
    base_stem = KERNEL_FILE_ALIASES.get(baseline, baseline)
    kernel_stem = KERNEL_FILE_ALIASES.get(kernel, kernel)

    gt_path = test_dir / "twb_gt.g2o"
    base_path = test_dir / f"twb_{base_stem}.g2o"
    kernel_path = test_dir / f"twb_{kernel_stem}.g2o"
    for p in (gt_path, base_path, kernel_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    gt_xy = _trajectory_xy(gt_path)
    base_xy = _trajectory_xy(base_path)
    kernel_xy = _trajectory_xy(kernel_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "g-", lw=2, label="Ground truth")
    ax.plot(base_xy[:, 0], base_xy[:, 1], "--", color="#ff7f0e", label=baseline)
    ax.plot(kernel_xy[:, 0], kernel_xy[:, 1], "-", color="#1f77b4", label=kernel)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"test_{test_idx}: {kernel} vs {baseline} vs GT")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="References batch visualiser + evaluation.")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Experiment root (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument("--plot-test", type=int, default=None, help="Optional test index for trajectory plot")
    parser.add_argument("--plot-kernel", default="gnd", help="Kernel to compare against baseline in plot")
    parser.add_argument("--baseline", default="gaussian", help="Baseline kernel for trajectory plot")
    parser.add_argument("--show", action="store_true", help="Show trajectory plot interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results.resolve()

    run_evaluation(results_root, OUTPUT_DIR, baseline=args.baseline)

    if args.plot_test is not None:
        out = SCRIPT_DIR / "output" / "references" / f"trajectory_test{args.plot_test}_{args.plot_kernel}.png"
        plot_kernel_vs_baseline(
            results_root,
            args.plot_test,
            args.plot_kernel,
            baseline=args.baseline,
            save_path=out,
            show=args.show,
        )


if __name__ == "__main__":
    main()
