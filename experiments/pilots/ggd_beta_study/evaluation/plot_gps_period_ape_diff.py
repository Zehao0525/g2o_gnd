#!/usr/bin/env python3
"""
Plot and tabulate Gaussian vs GGD APE *difference* across GPS periods.

Box plot of per-seed APE difference (Gaussian − GGD); positive values mean GGD has
lower APE. Annotations show GGD win count and paired t-test / Wilcoxon p-values.

Run from repo root:
  python3 experiments/pilots/ggd_beta_study/evaluation/plot_gps_period_ape_diff.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_gps_period_gaussian_vs_ggd import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PERIODS,
    DEFAULT_STUDY_ROOT,
    DEFAULT_SWEEP_ROOT,
    LEGACY_PERIOD_MAX_TRIALS,
    SIGFIGS,
    _format_p_value,
    _format_sigfig,
    summarize_period,
)

# Baseline y [m] for wins / p-value text above each box (data coordinates).
# Edit per period; use None to fall back to auto placement below.
DEFAULT_ANNOTATION_Y_OFFSET = 0.22  # × data span above ymax when auto
XLIM_PADDING = 1.9  # × min period gap added on each side
TEXT_BLOCK_HEIGHT = 0.38  # × data span reserved for annotation lines

ANNOTATION_Y_BY_PERIOD: Dict[int, Optional[float]] = {
    1: 2.5,
    2: 4.1,
    4: 2.5,
    6: 4.1,
    8: 2.5,
    10: 4.1,
    14: None,
    18: None,
    22: None,
    26: None,
    30: None,
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.family": "serif",
        }
    )


def _format_p_annotation(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3g}"


def write_csv(path: Path, summary: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gps_period",
        "results_dir",
        "n_trials",
        "mean_ape_diff_m",
        "std_ape_diff_m",
        "sem_ape_diff_m",
        "ggd_wins",
        "ggd_losses",
        "ggd_ties",
        "ttest_stat",
        "ttest_p",
        "wilcoxon_stat",
        "wilcoxon_p",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row.get(key) for key in fieldnames})


def render_latex_table(summary: Sequence[Dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Gaussian vs \gls{ggd} mean APE difference across GPS observation periods. "
        r"Difference is Gaussian minus \gls{ggd} mean translation APE per seed (positive = GGD better). "
        r"Paired $t$-test and Wilcoxon tests are two-sided on per-seed differences.}",
        r"\label{tab:gps_period_ape_diff}",
        r"\small",
        r"\begin{tabularx}{\columnwidth}{@{}r r c c c@{}}",
        r"\toprule",
        r"GPS period & Mean APE diff.\ [m] & GGD wins & paired $t$ $p$ & Wilcoxon $p$ \\",
        r"\midrule",
    ]

    best_diff = max(summary, key=lambda r: r["mean_ape_diff_m"])
    best_wins = max(summary, key=lambda r: r["ggd_wins"])

    for row in summary:
        period = int(row["gps_period"])
        diff_cell = _format_sigfig(row["mean_ape_diff_m"])
        if row is best_diff:
            diff_cell = rf"\mathbf{{{diff_cell}}}"
        wins = f"{row['ggd_wins']}/{row['n_trials']}"
        if row is best_wins:
            wins = rf"\textbf{{{wins}}}"
        t_p = _format_p_value(float(row["ttest_p"]))
        w_p = _format_p_value(float(row["wilcoxon_p"]))
        lines.append(f"{period} & ${diff_cell}$ & {wins} & {t_p} & {w_p} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def print_summary_table(summary: Sequence[Dict[str, Any]]) -> None:
    header = (
        f"{'Period':>6} {'APE diff':>10} {'Wins':>10} "
        f"{'t-test p':>12} {'Wilcoxon p':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['gps_period']:>6} "
            f"{_format_sigfig(row['mean_ape_diff_m']):>10} "
            f"{row['ggd_wins']:>4}/{row['n_trials']:<5} "
            f"{row['ttest_p']:>12.4g} "
            f"{row['wilcoxon_p']:>12.4g}"
        )


def _box_widths(periods: Sequence[int], scale: float = 0.45) -> List[float]:
    if len(periods) < 2:
        return [1.0]
    min_gap = min(periods[i + 1] - periods[i] for i in range(len(periods) - 1))
    return [scale * min_gap] * len(periods)


def _annotation_y_for_period(
    period: int,
    default_y: float,
    overrides: Dict[int, Optional[float]] = ANNOTATION_Y_BY_PERIOD,
) -> float:
    custom = overrides.get(period)
    return default_y if custom is None else custom


def plot_ape_diff(summary: Sequence[Dict[str, Any]], out_path: Path) -> None:
    _style()
    periods = [int(r["gps_period"]) for r in summary]
    data = [np.array(r["ape_diffs_m"], dtype=float) for r in summary]
    positions = [float(p) for p in periods]
    widths = _box_widths(periods)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, diffs in zip(bp["boxes"], data):
        patch.set_facecolor("#cfe2f3" if float(np.median(diffs)) >= 0 else "#f4cccc")
        patch.set_alpha(0.9)

    ax.axhline(0.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("GPS observation period")
    ax.set_ylabel("APE difference [m]\n(Gaussian − GGD)")
    ax.set_title("Per-seed APE difference vs GPS period (positive = lower APE with GGD)")
    ax.grid(True, axis="y", alpha=0.3)

    y_values = np.concatenate(data) if data else np.array([0.0])
    y_top = float(np.max(y_values))
    y_bot = float(np.min(y_values))
    span = max(y_top - y_bot, 0.05)

    min_gap = min(periods[i + 1] - periods[i] for i in range(len(periods) - 1))
    ax.set_xlim(periods[0] - XLIM_PADDING * min_gap, periods[-1] + XLIM_PADDING * min_gap)
    ax.set_xticks(periods)
    ax.set_xticklabels([str(p) for p in periods])

    default_label_y = y_top + DEFAULT_ANNOTATION_Y_OFFSET * span
    label_ys = [_annotation_y_for_period(period, default_label_y) for period in periods]
    text_block_height = TEXT_BLOCK_HEIGHT * span
    ax.set_ylim(y_bot - 0.10 * span, max(label_ys) + text_block_height)

    for pos, row, label_y in zip(positions, summary, label_ys):
        wins = f"{row['ggd_wins']}/{row['n_trials']}"
        t_p = _format_p_annotation(float(row["ttest_p"]))
        w_p = _format_p_annotation(float(row["wilcoxon_p"]))
        ax.text(
            pos,
            label_y,
            f"{wins}\n$t$: {t_p}\n$W$: {w_p}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            linespacing=1.15,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GPS-period APE difference (Gaussian − GGD) with significance."
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_STUDY_ROOT,
        help=f"Study results root (default: {DEFAULT_STUDY_ROOT})",
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=None,
        help="GPS period sweep directory (default: study-root/gps_period_sweep)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        nargs="*",
        default=DEFAULT_PERIODS,
        help="GPS periods to include",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--legacy-max-trials",
        type=int,
        default=LEGACY_PERIOD_MAX_TRIALS,
        help="Max seeds from legacy period 1/2 runs (default: first 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_root = args.study_root.resolve()
    sweep_root = (args.sweep_root or study_root / "gps_period_sweep").resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    missing: List[int] = []

    for period in args.periods:
        try:
            summary.append(
                summarize_period(
                    period,
                    study_root,
                    sweep_root,
                    legacy_max_trials=args.legacy_max_trials,
                )
            )
        except FileNotFoundError:
            missing.append(period)

    if missing:
        print(f"WARNING: skipped missing periods: {missing}")
    if not summary:
        raise RuntimeError("No period results found")

    summary.sort(key=lambda r: int(r["gps_period"]))

    csv_path = output_dir / "gps_period_ape_diff_summary.csv"
    latex_path = output_dir / "gps_period_ape_diff_table.txt"
    plot_path = output_dir / "gps_period_ape_diff.png"

    write_csv(csv_path, summary)
    latex_path.write_text(render_latex_table(summary), encoding="utf-8")
    plot_ape_diff(summary, plot_path)

    print("\nAPE difference (Gaussian − GGD) across GPS periods")
    print(f"Sweep root: {sweep_root}")
    print_summary_table(summary)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {latex_path}")


if __name__ == "__main__":
    main()
