#!/usr/bin/env python3
"""
Build kernel-comparison paper tables from references_robust_kernels_period_* results.

For each GPS period experiment (100 Monte Carlo trials by default), computes:
  - APE mean ± std and median (from trajectory-ordered translation APE)
  - Win rate vs Gaussian baseline (lower APE wins)
  - Two-sided Wilcoxon signed-rank p-value vs Gaussian

Writes:
  - kernel_comparison_stats_period_{X}.csv
  - kernel_comparison_table_ape_period_{X}.txt
  - kernel_comparison_table_tests_period_{X}.txt
  - kernel_comparison_tables_all_periods.txt

Run from repo root:
  python3 experiments/pilots/ggd_beta_study/evaluation/make_kernel_comparison_table.py
  python3 experiments/pilots/ggd_beta_study/evaluation/make_kernel_comparison_table.py --periods 1 2 30
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_references import (
    BASELINE_KERNEL,
    evaluate_batch,
    win_rates_vs_baseline,
)

DEFAULT_RESULTS_PARENT = Path("test_results/ggd_beta_study")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "references" / "paper_tables"

# Publication row order. Existing test_results use on-disk kernel id "gnd";
# new dumps may use "ggd". Both map to the same GGD display name.
KERNEL_ORDER = [
    "gaussian",
    "geman_mcclure",
    "gnd",
    "ggd",
    "huber",
    "tukey",
]

KERNEL_LATEX = {
    "gaussian": "Gaussian",
    "geman_mcclure": "Geman--McClure",
    "gnd": r"\gls{ggd}",
    "ggd": r"\gls{ggd}",
    "huber": "Huber",
    "tukey": "Tukey",
}


def _aggregate_ape(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "n": int(values.size),
    }


def build_kernel_stats(
    rows: Sequence[Dict[str, Any]],
    baseline: str = BASELINE_KERNEL,
) -> Tuple[List[Dict[str, Any]], int]:
    """Per-kernel APE stats, win rates, and Wilcoxon p vs baseline."""
    by_test: Dict[int, Dict[str, float]] = {}
    for row in rows:
        by_test.setdefault(row["test_idx"], {})[row["kernel"]] = row["ape_mean_m"]

    tests = sorted(t for t, d in by_test.items() if baseline in d)
    n_trials = len(tests)
    base_vec = np.array([by_test[t][baseline] for t in tests], dtype=float)

    wins_map = {w["kernel"]: w for w in win_rates_vs_baseline(rows, baseline)}

    stats: List[Dict[str, Any]] = []
    for kernel in KERNEL_ORDER:
        if kernel not in {r["kernel"] for r in rows}:
            continue
        ape_vals = np.array(
            [by_test[t][kernel] for t in tests if kernel in by_test[t]], dtype=float
        )
        agg = _aggregate_ape(ape_vals)
        entry: Dict[str, Any] = {
            "kernel": kernel,
            "kernel_latex": KERNEL_LATEX.get(kernel, kernel.replace("_", " ").title()),
            "n_trials": agg["n"],
            "ape_mean_m": agg["mean"],
            "ape_std_m": agg["std"],
            "ape_median_m": agg["median"],
        }

        if kernel == baseline:
            entry["wins_vs_baseline"] = None
            entry["win_rate"] = None
            entry["wilcoxon_p"] = None
        else:
            win_row = wins_map.get(kernel, {})
            wins = int(win_row.get("wins", 0))
            entry["wins_vs_baseline"] = wins
            entry["win_rate"] = wins / n_trials if n_trials else np.nan
            paired = np.array([by_test[t][kernel] for t in tests], dtype=float)
            diff = base_vec - paired
            try:
                _, wp = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto")
            except ValueError:
                wp = np.nan
            entry["wilcoxon_p"] = float(wp) if np.isfinite(wp) else np.nan

        stats.append(entry)

    return stats, n_trials


SIGFIGS = 3


def _format_sigfig(value: float, sigfigs: int = SIGFIGS) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{sigfigs}g}"


def _format_p_value(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p == 0.0:
        return r"$<10^{-308}$"
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        mantissa = p / (10.0**exp)
        return rf"${_format_sigfig(mantissa)}\times10^{{{exp}}}$"
    return f"${_format_sigfig(p)}$"


def _bold(text: str, use_bold: bool) -> str:
    return rf"\textbf{{{text}}}" if use_bold else text


def _bold_math_dollar(text: str, use_bold: bool) -> str:
    if not use_bold:
        return text
    inner = text.strip("$")
    return rf"$\mathbf{{{inner}}}$"


def _format_ape(mean: float, std: float, bold: bool) -> str:
    body = f"{_format_sigfig(mean)} \\pm {_format_sigfig(std)}"
    if bold:
        return rf"$\mathbf{{{body}}}$"
    return f"${body}$"


def _format_median(median: float, bold: bool) -> str:
    body = _format_sigfig(median)
    if bold:
        return rf"$\mathbf{{{body}}}$"
    return f"${body}$"


def _format_wins(wins: int, n_trials: int, bold: bool) -> str:
    return _bold(f"{wins}/{n_trials}", bold)


def _best_kernels(stats: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    non_baseline = [s for s in stats if s["kernel"] != BASELINE_KERNEL]
    return {
        "ape": min(stats, key=lambda s: s["ape_mean_m"])["kernel"],
        "median": min(stats, key=lambda s: s["ape_median_m"])["kernel"],
        "wins": max(non_baseline, key=lambda s: s["wins_vs_baseline"] or -1)["kernel"],
        "p": min(non_baseline, key=lambda s: s["wilcoxon_p"])["kernel"],
    }


def _table_preamble(
    caption: str,
    label: str,
    col_spec: str,
    header: str,
    *,
    width: Optional[str] = None,
) -> Tuple[List[str], str]:
    end_tag = "tabularx" if width is not None else "tabular"
    if width is None:
        begin_env = rf"\begin{{tabular}}{{{col_spec}}}"
    else:
        begin_env = rf"\begin{{tabularx}}{{{width}}}{{{col_spec}}}"

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        begin_env,
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    return lines, end_tag


def _table_postamble(lines: List[str], end_tag: str) -> str:
    lines.extend(
        [
            r"\bottomrule",
            rf"\end{{{end_tag}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_latex_ape_table(
    stats: Sequence[Dict[str, Any]],
    *,
    gps_period: int,
    n_trials: int,
    label_suffix: str,
) -> str:
    best = _best_kernels(stats)
    caption = (
        f"APE mean $\\pm$ std by robust kernel under correlated bounded GPS-like noise "
        f"(GPS period {gps_period}, {n_trials} Monte Carlo trials). Lower is better."
    )
    lines, end_tag = _table_preamble(
        caption,
        f"tab:kernel_comparison_ape_period_{label_suffix}",
        r"@{}>{\raggedright\arraybackslash}X r r@{}",
        r"Kernel & APE mean $\pm$ std [m] & APE median [m]",
        width=r"\columnwidth",
    )
    for row in stats:
        kernel = row["kernel"]
        ape_cell = _format_ape(
            row["ape_mean_m"],
            row["ape_std_m"],
            bold=(kernel == best["ape"]),
        )
        median_cell = _format_median(
            row["ape_median_m"],
            bold=(kernel == best["median"]),
        )
        lines.append(f"{row['kernel_latex']} & {ape_cell} & {median_cell} \\\\")
    return _table_postamble(lines, end_tag)


def render_latex_tests_table(
    stats: Sequence[Dict[str, Any]],
    *,
    gps_period: int,
    n_trials: int,
    label_suffix: str,
) -> str:
    best = _best_kernels(stats)
    caption = (
        f"Paired comparison against the Gaussian baseline "
        f"(GPS period {gps_period}, {n_trials} Monte Carlo trials). "
        f"Win rate counts seeds with lower APE than Gaussian; "
        f"Wilcoxon $p$ is two-sided on paired APE differences."
    )
    lines, end_tag = _table_preamble(
        caption,
        f"tab:kernel_comparison_tests_period_{label_suffix}",
        r"@{}l c c@{}",
        r"Kernel & Wins vs Gaussian & Wilcoxon $p$ vs Gaussian",
    )
    for row in stats:
        kernel = row["kernel"]
        if kernel == BASELINE_KERNEL:
            wins_cell = "--"
            p_cell = "--"
        else:
            wins_cell = _format_wins(
                int(row["wins_vs_baseline"]),
                n_trials,
                bold=(kernel == best["wins"]),
            )
            p_cell = _bold_math_dollar(
                _format_p_value(float(row["wilcoxon_p"])),
                kernel == best["p"],
            )
        lines.append(f"{row['kernel_latex']} & {wins_cell} & {p_cell} \\\\")
    return _table_postamble(lines, end_tag)


def render_latex_tables(
    stats: Sequence[Dict[str, Any]],
    *,
    gps_period: int,
    n_trials: int,
    label_suffix: str,
) -> Tuple[str, str]:
    ape = render_latex_ape_table(
        stats, gps_period=gps_period, n_trials=n_trials, label_suffix=label_suffix
    )
    tests = render_latex_tests_table(
        stats, gps_period=gps_period, n_trials=n_trials, label_suffix=label_suffix
    )
    return ape, tests


def write_stats_csv(path: Path, stats: Sequence[Dict[str, Any]], n_trials: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kernel",
        "kernel_latex",
        "n_trials",
        "ape_mean_m",
        "ape_std_m",
        "ape_median_m",
        "wins_vs_baseline",
        "win_rate",
        "wilcoxon_p",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            out = dict(row)
            out["n_trials"] = n_trials
            writer.writerow(out)


def print_stats_table(stats: Sequence[Dict[str, Any]], n_trials: int, gps_period: int) -> None:
    print(f"\n=== GPS period {gps_period} ({n_trials} trials) ===")
    header = f"{'Kernel':<16} {'APE mean±std':>22} {'median':>10} {'Wins':>10} {'Wilcoxon p':>14}"
    print(header)
    print("-" * len(header))
    for row in stats:
        wins = "--" if row["wins_vs_baseline"] is None else f"{row['wins_vs_baseline']}/{n_trials}"
        p = "--" if row["wilcoxon_p"] is None else _format_sigfig(row["wilcoxon_p"])
        print(
            f"{row['kernel']:<16} "
            f"{_format_sigfig(row['ape_mean_m'])} ± {_format_sigfig(row['ape_std_m'])} "
            f"{_format_sigfig(row['ape_median_m']):>10} "
            f"{wins:>10} "
            f"{p:>14}"
        )


def process_period(
    results_root: Path,
    output_dir: Path,
    gps_period: int,
    baseline: str,
) -> str:
    if not results_root.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_root}")

    rows = evaluate_batch(results_root)
    stats, n_trials = build_kernel_stats(rows, baseline=baseline)

    suffix = str(gps_period)
    write_stats_csv(output_dir / f"kernel_comparison_stats_period_{suffix}.csv", stats, n_trials)
    ape_latex, tests_latex = render_latex_tables(
        stats, gps_period=gps_period, n_trials=n_trials, label_suffix=suffix
    )
    ape_path = output_dir / f"kernel_comparison_table_ape_period_{suffix}.txt"
    tests_path = output_dir / f"kernel_comparison_table_tests_period_{suffix}.txt"
    ape_path.write_text(ape_latex, encoding="utf-8")
    tests_path.write_text(tests_latex, encoding="utf-8")

    print_stats_table(stats, n_trials, gps_period)
    print(f"Wrote {output_dir / f'kernel_comparison_stats_period_{suffix}.csv'}")
    print(f"Wrote {ape_path}")
    print(f"Wrote {tests_path}")
    return ape_latex + "\n" + tests_latex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate kernel comparison LaTeX tables.")
    parser.add_argument(
        "--results-parent",
        type=Path,
        default=DEFAULT_RESULTS_PARENT,
        help=f"Parent dir containing references_robust_kernels_period_* (default: {DEFAULT_RESULTS_PARENT})",
    )
    parser.add_argument(
        "--periods",
        type=int,
        nargs="+",
        default=[1, 2, 30],
        help="GPS period suffixes to process (default: 1 2 30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--baseline",
        default=BASELINE_KERNEL,
        help=f"Baseline kernel (default: {BASELINE_KERNEL})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_parent = args.results_parent.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building kernel comparison tables from {results_parent}")

    combined_lines: List[str] = []
    for period in args.periods:
        results_root = results_parent / f"references_robust_kernels_period_{period}"
        combined_lines.append(process_period(results_root, output_dir, period, args.baseline))
        combined_lines.append("\n")

    combined_path = output_dir / "kernel_comparison_tables_all_periods.txt"
    combined_path.write_text("".join(combined_lines), encoding="utf-8")
    print(f"\nWrote combined LaTeX: {combined_path}")


if __name__ == "__main__":
    main()
