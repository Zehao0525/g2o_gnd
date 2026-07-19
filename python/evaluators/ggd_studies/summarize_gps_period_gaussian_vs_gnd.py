#!/usr/bin/env python3
"""
Summarize Gaussian vs GND (GGD) APE across GPS period experiments.

Reads:
  - Legacy period 1/2: references_robust_kernels_period_{P}/ (first 30 seeds only)
  - Period sweep 4–30: gps_period_sweep/period_{P}/

For each GPS period, reports mean APE (Gaussian, GND), GND win count vs Gaussian,
and paired t-test / Wilcoxon signed-rank p-values on per-seed APE differences.

Run from repo root:
  python3 python/evaluators/gnd_studies/summarize_gps_period_gaussian_vs_gnd.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_references import evaluate_batch

DEFAULT_STUDY_ROOT = Path("test_results/gnd_beta_study")
DEFAULT_SWEEP_ROOT = DEFAULT_STUDY_ROOT / "gps_period_sweep"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "gps_period_sweep"
DEFAULT_PERIODS = [1, 2, 4, 6, 8, 10, 14, 18, 22, 26, 30]
LEGACY_PERIOD_MAX_TRIALS = 30  # periods 1/2 have 100 seeds; use first 30 for fair comparison

GAUSSIAN_KERNEL = "gaussian"
GND_KERNEL = "gnd"
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


def resolve_period_dir(
    period: int,
    study_root: Path,
    sweep_root: Path,
) -> Optional[Path]:
    legacy = study_root / f"references_robust_kernels_period_{period}"
    sweep = sweep_root / f"period_{period}"

    if period <= 2 and legacy.is_dir():
        return legacy
    if sweep.is_dir():
        return sweep
    if legacy.is_dir():
        return legacy
    return None


def paired_gaussian_vs_gnd(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_test: Dict[int, Dict[str, float]] = {}
    for row in rows:
        if row["kernel"] not in {GAUSSIAN_KERNEL, GND_KERNEL}:
            continue
        by_test.setdefault(int(row["test_idx"]), {})[row["kernel"]] = float(row["ape_mean_m"])

    tests = sorted(t for t, kernels in by_test.items() if GAUSSIAN_KERNEL in kernels and GND_KERNEL in kernels)
    if not tests:
        raise RuntimeError("No paired Gaussian/GND rows found")

    gauss = np.array([by_test[t][GAUSSIAN_KERNEL] for t in tests], dtype=float)
    gnd = np.array([by_test[t][GND_KERNEL] for t in tests], dtype=float)
    diff = gauss - gnd  # positive => GND better (lower APE)

    wins = int(np.sum(gnd < gauss - 1e-12))
    losses = int(np.sum(gnd > gauss + 1e-12))
    ties = len(tests) - wins - losses

    try:
        wstat, wp = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto")
    except ValueError:
        wstat, wp = np.nan, np.nan

    tstat, tp = ttest_rel(gauss, gnd, nan_policy="omit")

    return {
        "n_trials": len(tests),
        "ape_mean_gaussian_m": float(np.mean(gauss)),
        "ape_mean_gnd_m": float(np.mean(gnd)),
        "ape_median_gaussian_m": float(np.median(gauss)),
        "ape_median_gnd_m": float(np.median(gnd)),
        "gnd_wins": wins,
        "gnd_losses": losses,
        "gnd_ties": ties,
        "mean_ape_diff_m": float(np.mean(diff)),
        "std_ape_diff_m": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
        "sem_ape_diff_m": float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0,
        "ape_diffs_m": diff.tolist(),
        "ttest_stat": float(tstat),
        "ttest_p": float(tp),
        "wilcoxon_stat": float(wstat) if np.isfinite(wstat) else np.nan,
        "wilcoxon_p": float(wp) if np.isfinite(wp) else np.nan,
    }


def limit_rows_to_first_n_trials(
    rows: Sequence[Dict[str, Any]], max_trials: int
) -> List[Dict[str, Any]]:
    return [row for row in rows if int(row["test_idx"]) < max_trials]


def summarize_period(
    period: int,
    study_root: Path,
    sweep_root: Path,
    legacy_max_trials: int = LEGACY_PERIOD_MAX_TRIALS,
) -> Dict[str, Any]:
    results_dir = resolve_period_dir(period, study_root, sweep_root)
    if results_dir is None:
        raise FileNotFoundError(f"No results found for GPS period {period}")

    rows = evaluate_batch(results_dir)
    if period <= 2:
        rows = limit_rows_to_first_n_trials(rows, legacy_max_trials)
    stats = paired_gaussian_vs_gnd(rows)
    stats["gps_period"] = period
    stats["results_dir"] = str(results_dir)
    return stats


def write_csv(path: Path, summary: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gps_period",
        "results_dir",
        "n_trials",
        "ape_mean_gaussian_m",
        "ape_mean_gnd_m",
        "ape_median_gaussian_m",
        "ape_median_gnd_m",
        "gnd_wins",
        "gnd_losses",
        "gnd_ties",
        "mean_ape_diff_m",
        "ttest_stat",
        "ttest_p",
        "wilcoxon_stat",
        "wilcoxon_p",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def render_latex_table(summary: Sequence[Dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Gaussian vs \gls{ggd} APE across GPS observation periods. "
        r"Win count is the number of seeds where GND mean translation APE is lower than Gaussian. "
        r"Paired $t$-test and Wilcoxon tests are two-sided on per-seed APE differences.}",
        r"\label{tab:gps_period_gaussian_vs_gnd}",
        r"\small",
        r"\begin{tabularx}{\columnwidth}{@{}r r r c c c@{}}",
        r"\toprule",
        r"GPS period & APE mean Gaussian [m] & APE mean \gls{ggd} [m] & GND wins & "
        r"paired $t$ $p$ & Wilcoxon $p$ \\",
        r"\midrule",
    ]

    best_gnd_mean = min(summary, key=lambda r: r["ape_mean_gnd_m"])
    best_wins = max(summary, key=lambda r: r["gnd_wins"])

    for row in summary:
        period = int(row["gps_period"])
        gauss_cell = _format_sigfig(row["ape_mean_gaussian_m"])
        gnd_cell = _format_sigfig(row["ape_mean_gnd_m"])
        if row is best_gnd_mean:
            gnd_cell = rf"\mathbf{{{gnd_cell}}}"
        wins = f"{row['gnd_wins']}/{row['n_trials']}"
        if row is best_wins:
            wins = rf"\textbf{{{wins}}}"
        t_p = _format_p_value(float(row["ttest_p"]))
        w_p = _format_p_value(float(row["wilcoxon_p"]))
        lines.append(
            f"{period} & ${gauss_cell}$ & ${gnd_cell}$ & {wins} & {t_p} & {w_p} \\\\"
        )

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
        f"{'Period':>6} {'APE Gauss':>10} {'APE GND':>10} {'Wins':>10} "
        f"{'t-test p':>12} {'Wilcoxon p':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['gps_period']:>6} "
            f"{_format_sigfig(row['ape_mean_gaussian_m']):>10} "
            f"{_format_sigfig(row['ape_mean_gnd_m']):>10} "
            f"{row['gnd_wins']:>4}/{row['n_trials']:<5} "
            f"{row['ttest_p']:>12.4g} "
            f"{row['wilcoxon_p']:>12.4g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gaussian vs GND summary table across GPS periods."
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

    csv_path = output_dir / "gps_period_gaussian_vs_gnd_summary.csv"
    latex_path = output_dir / "gps_period_gaussian_vs_gnd_table.txt"
    write_csv(csv_path, summary)
    latex_path.write_text(render_latex_table(summary), encoding="utf-8")

    print(f"\nGaussian vs GND across GPS periods")
    print(f"Sweep root: {sweep_root}")
    print_summary_table(summary)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {latex_path}")


if __name__ == "__main__":
    main()
