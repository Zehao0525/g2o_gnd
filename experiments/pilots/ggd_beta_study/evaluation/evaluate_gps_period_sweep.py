#!/usr/bin/env python3
"""
Evaluate GPS period sweep outputs from tutorial_gps_period_sweep.

Expects layout:
  test_results/ggd_beta_study/gps_period_sweep/period_{P}/test_{i}/...

Run from repo root:
  python3 experiments/pilots/ggd_beta_study/evaluation/evaluate_gps_period_sweep.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_references import BASELINE_KERNEL
from make_kernel_comparison_table import process_period

DEFAULT_SWEEP_ROOT = Path("test_results/ggd_beta_study/gps_period_sweep")
DEFAULT_PERIODS = [1, 2, 4, 6, 8, 10, 14, 18, 22, 26, 30]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "gps_period_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPS period sweep experiment.")
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=DEFAULT_SWEEP_ROOT,
        help=f"Root output directory (default: {DEFAULT_SWEEP_ROOT})",
    )
    parser.add_argument(
        "--periods",
        type=int,
        nargs="*",
        default=DEFAULT_PERIODS,
        help="GPS periods to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for evaluation artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--baseline",
        default=BASELINE_KERNEL,
        help=f"Baseline kernel (default: {BASELINE_KERNEL})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sweep_root.is_dir():
        raise FileNotFoundError(f"Missing sweep output directory: {sweep_root}")

    combined: list[str] = []
    for period in args.periods:
        results_root = sweep_root / f"period_{period}"
        if not results_root.is_dir():
            raise FileNotFoundError(f"Missing period directory: {results_root}")
        combined.append(process_period(results_root, output_dir, period, args.baseline))
        combined.append("\n")

    combined_path = output_dir / "kernel_comparison_tables_all_periods.txt"
    combined_path.write_text("".join(combined), encoding="utf-8")
    print(f"\nWrote combined LaTeX: {combined_path}")


if __name__ == "__main__":
    main()
