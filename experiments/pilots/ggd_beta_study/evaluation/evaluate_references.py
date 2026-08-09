#!/usr/bin/env python3
"""
Evaluate tutorial_w_references multi-kernel batch outputs.

Metrics per (test, kernel):
  - APE mean / RMSE / max  (correct trajectory-ordered translation error)
  - MSE                    (mean squared translation error, legacy-style)

Analyses:
  1. Multi-kernel ranking (median/IQR, win rate vs baseline, paired tests)
  2. Relative APE improvement vs Gaussian baseline
  8. Seed sensitivity (paired scatter, hard-seed identification)

Run from repo root:
  python3 experiments/pilots/ggd_beta_study/evaluation/evaluate_references.py
  python3 experiments/pilots/ggd_beta_study/evaluation/evaluate_references.py \\
      --results test_results/ggd_beta_study/references_robust_kernels
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from g2o_io import compute_translation_ape_from_g2o, compute_translation_mse_from_g2o

DEFAULT_RESULTS_ROOT = Path("test_results/ggd_beta_study/references_robust_kernels")
OUTPUT_DIR = SCRIPT_DIR / "output" / "references"
BASELINE_KERNEL = "gaussian"
SKIP_G2O_STEMS = frozenset({"before", "gt"})


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


def discover_test_dirs(results_root: Path) -> List[Path]:
    dirs = sorted(results_root.glob("test_*"), key=lambda p: int(p.name.split("_")[1]))
    if not dirs:
        raise FileNotFoundError(f"No test_* directories under {results_root}")
    return dirs


def load_kernel_specs(test_dir: Path) -> List[Tuple[str, str]]:
    """
    Return (kernel_name, g2o_stem) pairs.

    Uses kernel_summary.csv kernel names with twb_{stem}.g2o on disk.
    Handles config aliases such as gaussian -> twb_gauss.g2o.
    """
    summary_path = test_dir / "kernel_summary.csv"
    if summary_path.is_file():
        names: List[str] = []
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                names.append(row["kernel"].strip())

        stems_on_disk = {
            p.stem.replace("twb_", "")
            for p in test_dir.glob("twb_*.g2o")
            if p.stem.replace("twb_", "") not in SKIP_G2O_STEMS
        }
        alias_stem = {"gaussian": "gauss"}

        specs: List[Tuple[str, str]] = []
        for name in names:
            candidates = [alias_stem.get(name, name), name]
            stem = next((c for c in candidates if c in stems_on_disk), None)
            if stem is None:
                print(f"[WARNING] no g2o file for kernel {name} in {test_dir}")
                continue
            specs.append((name, stem))
        if specs:
            return specs

    name_map = {"gauss": "gaussian"}
    stems = sorted(
        p.stem.replace("twb_", "")
        for p in test_dir.glob("twb_*.g2o")
        if p.stem.replace("twb_", "") not in SKIP_G2O_STEMS
    )
    return [(name_map.get(s, s), s) for s in stems]


def resolve_kernel_specs(test_dir: Path) -> List[Tuple[str, str]]:
    specs = load_kernel_specs(test_dir)
    if not specs:
        raise FileNotFoundError(f"No kernel g2o files in {test_dir}")
    return specs


def evaluate_single_test(
    test_dir: Path,
    test_idx: int,
    kernel_specs: Sequence[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    gt_path = test_dir / "twb_gt.g2o"
    if not gt_path.is_file():
        gt_path = test_dir / "gt.g2o"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing ground truth in {test_dir}")

    rows: List[Dict[str, Any]] = []
    for kernel_name, stem in kernel_specs:
        est_path = test_dir / f"twb_{stem}.g2o"
        if not est_path.is_file():
            print(f"[WARNING] skip test_{test_idx} kernel {kernel_name}: missing {est_path.name}")
            continue
        ape_mean, ape_rmse, ape_max, n = compute_translation_ape_from_g2o(gt_path, est_path)
        mse, _ = compute_translation_mse_from_g2o(gt_path, est_path)
        rows.append(
            {
                "test_idx": test_idx,
                "kernel": kernel_name,
                "file_stem": stem,
                "ape_mean_m": ape_mean,
                "ape_rmse_m": ape_rmse,
                "ape_max_m": ape_max,
                "mse_m2": mse,
                "num_poses": n,
            }
        )
    return rows


def evaluate_batch(results_root: Path) -> List[Dict[str, Any]]:
    test_dirs = discover_test_dirs(results_root)
    kernel_specs = resolve_kernel_specs(test_dirs[0])
    all_rows: List[Dict[str, Any]] = []
    for test_dir in test_dirs:
        test_idx = int(test_dir.name.split("_")[1])
        try:
            all_rows.extend(evaluate_single_test(test_dir, test_idx, kernel_specs))
        except Exception as exc:
            print(f"[ERROR] test_{test_idx}: {exc}")
    if not all_rows:
        raise RuntimeError("No results evaluated")
    return all_rows


def _aggregate(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def summarize_kernels(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kernels = sorted({r["kernel"] for r in rows})
    summary: List[Dict[str, Any]] = []
    for kernel in kernels:
        group = [r for r in rows if r["kernel"] == kernel]
        entry: Dict[str, Any] = {"kernel": kernel, "num_tests": len(group)}
        for metric in ("ape_mean_m", "ape_rmse_m", "ape_max_m", "mse_m2"):
            stats = _aggregate(np.array([r[metric] for r in group], dtype=float))
            for k, v in stats.items():
                entry[f"{metric}_{k}"] = v
        summary.append(entry)
    return summary


def relative_improvement_vs_baseline(
    rows: Sequence[Dict[str, Any]], baseline: str = BASELINE_KERNEL
) -> List[Dict[str, Any]]:
    by_test: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_test.setdefault(row["test_idx"], {})[row["kernel"]] = row

    out: List[Dict[str, Any]] = []
    for test_idx, kernels in sorted(by_test.items()):
        if baseline not in kernels:
            continue
        base_ape = kernels[baseline]["ape_mean_m"]
        base_mse = kernels[baseline]["mse_m2"]
        for kernel, row in kernels.items():
            if kernel == baseline:
                continue
            rel_ape = (base_ape - row["ape_mean_m"]) / base_ape if base_ape > 0 else np.nan
            rel_mse = (base_mse - row["mse_m2"]) / base_mse if base_mse > 0 else np.nan
            out.append(
                {
                    "test_idx": test_idx,
                    "kernel": kernel,
                    "baseline": baseline,
                    "ape_rel_improvement": rel_ape,
                    "mse_rel_improvement": rel_mse,
                    "ape_abs_reduction_m": base_ape - row["ape_mean_m"],
                }
            )
    return out


def win_rates_vs_baseline(
    rows: Sequence[Dict[str, Any]], baseline: str = BASELINE_KERNEL
) -> List[Dict[str, Any]]:
    by_test: Dict[int, Dict[str, float]] = {}
    for row in rows:
        by_test.setdefault(row["test_idx"], {})[row["kernel"]] = row["ape_mean_m"]

    kernels = sorted({r["kernel"] for r in rows if r["kernel"] != baseline})
    results: List[Dict[str, Any]] = []
    tests_with_base = [t for t, d in by_test.items() if baseline in d]
    n = len(tests_with_base)
    for kernel in kernels:
        wins = ties = losses = 0
        for test_idx in tests_with_base:
            base = by_test[test_idx][baseline]
            if kernel not in by_test[test_idx]:
                continue
            val = by_test[test_idx][kernel]
            if val < base - 1e-12:
                wins += 1
            elif val > base + 1e-12:
                losses += 1
            else:
                ties += 1
        results.append(
            {
                "kernel": kernel,
                "baseline": baseline,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate": wins / n if n else np.nan,
            }
        )
    return results


def paired_tests_vs_baseline(
    rows: Sequence[Dict[str, Any]], baseline: str = BASELINE_KERNEL
) -> List[Dict[str, Any]]:
    by_test: Dict[int, Dict[str, float]] = {}
    for row in rows:
        by_test.setdefault(row["test_idx"], {})[row["kernel"]] = row["ape_mean_m"]

    tests = sorted(t for t, d in by_test.items() if baseline in d)
    base = np.array([by_test[t][baseline] for t in tests], dtype=float)
    kernels = sorted({r["kernel"] for r in rows if r["kernel"] != baseline})

    out: List[Dict[str, Any]] = []
    for kernel in kernels:
        paired = np.array([by_test[t][kernel] for t in tests if kernel in by_test[t]], dtype=float)
        if paired.size != base.size:
            continue
        diff = base - paired  # positive => kernel better (lower APE)
        try:
            wstat, wp = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto")
        except ValueError:
            wstat, wp = np.nan, np.nan
        tstat, tp = ttest_rel(base, paired, nan_policy="omit")
        out.append(
            {
                "kernel": kernel,
                "baseline": baseline,
                "mean_ape_diff_m": float(np.mean(diff)),
                "median_ape_diff_m": float(np.median(diff)),
                "wilcoxon_stat": float(wstat) if np.isfinite(wstat) else np.nan,
                "wilcoxon_p": float(wp) if np.isfinite(wp) else np.nan,
                "ttest_stat": float(tstat),
                "ttest_p": float(tp),
            }
        )
    return out


def identify_hard_seeds(
    rows: Sequence[Dict[str, Any]], baseline: str = BASELINE_KERNEL, quantile: float = 0.75
) -> List[Dict[str, Any]]:
    by_test: Dict[int, Dict[str, float]] = {}
    for row in rows:
        by_test.setdefault(row["test_idx"], {})[row["kernel"]] = row["ape_mean_m"]

    kernels = sorted({r["kernel"] for r in rows})
    tests = sorted(by_test.keys())
    threshold = float(np.quantile([by_test[t][baseline] for t in tests if baseline in by_test[t]], quantile))

    out: List[Dict[str, Any]] = []
    for test_idx in tests:
        if baseline not in by_test[test_idx]:
            continue
        base_ape = by_test[test_idx][baseline]
        kernel_apes = {k: by_test[test_idx][k] for k in kernels if k in by_test[test_idx]}
        best_kernel = min(kernel_apes, key=kernel_apes.get)
        best_ape = kernel_apes[best_kernel]
        out.append(
            {
                "test_idx": test_idx,
                "baseline_ape_m": base_ape,
                "best_kernel": best_kernel,
                "best_ape_m": best_ape,
                "hard_for_baseline": base_ape >= threshold,
                "all_kernels_worse_than_best": {
                    k: v - best_ape for k, v in kernel_apes.items() if k != best_kernel
                },
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_ape_violin(rows: Sequence[Dict[str, Any]], summary: Sequence[Dict[str, Any]]) -> None:
    kernels = [s["kernel"] for s in summary]
    data = [[r["ape_mean_m"] for r in rows if r["kernel"] == k] for k in kernels]
    fig, ax = plt.subplots(figsize=(max(8, len(kernels) * 1.2), 5))
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    ax.set_xticks(np.arange(1, len(kernels) + 1))
    ax.set_xticklabels(kernels, rotation=20, ha="right")
    ax.set_ylabel("APE mean [m]")
    ax.set_title("Pose error distribution over seeds (violin)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "ape_violin_by_kernel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_relative_improvement(rel_rows: Sequence[Dict[str, Any]]) -> None:
    kernels = sorted({r["kernel"] for r in rel_rows})
    fig, ax = plt.subplots(figsize=(max(8, len(kernels) * 1.2), 5))
    data = [[100.0 * r["ape_rel_improvement"] for r in rel_rows if r["kernel"] == k] for k in kernels]
    bp = ax.boxplot(data, labels=kernels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe2f3")
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    ax.set_ylabel("Relative APE improvement vs Gaussian [%]")
    ax.set_title("Positive = lower APE than Gaussian baseline")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out = OUTPUT_DIR / "ape_relative_improvement_vs_gaussian.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_seed_scatter(rows: Sequence[Dict[str, Any]], baseline: str = BASELINE_KERNEL) -> None:
    by_test: Dict[int, Dict[str, float]] = {}
    for row in rows:
        by_test.setdefault(row["test_idx"], {})[row["kernel"]] = row["ape_mean_m"]
    tests = sorted(by_test.keys())
    kernels = sorted(k for k in {r["kernel"] for r in rows} if k != baseline)

    n = len(kernels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows), squeeze=False)
    base = np.array([by_test[t].get(baseline, np.nan) for t in tests])

    for idx, kernel in enumerate(kernels):
        ax = axes[idx // ncols][idx % ncols]
        other = np.array([by_test[t].get(kernel, np.nan) for t in tests])
        ax.scatter(base, other, alpha=0.75, s=40)
        lim_lo = float(np.nanmin([base, other]))
        lim_hi = float(np.nanmax([base, other]))
        pad = 0.05 * (lim_hi - lim_lo + 1e-9)
        ax.plot([lim_lo - pad, lim_hi + pad], [lim_lo - pad, lim_hi + pad], "k--", lw=1, alpha=0.5)
        ax.set_xlabel(f"{baseline} APE [m]")
        ax.set_ylabel(f"{kernel} APE [m]")
        ax.set_title(kernel)
        ax.grid(True, alpha=0.3)
        below = np.sum(other < base - 1e-12)
        ax.text(0.05, 0.95, f"wins {below}/{len(tests)}", transform=ax.transAxes, va="top")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Seed sensitivity: APE per kernel vs Gaussian baseline", y=1.01)
    fig.tight_layout()
    out = OUTPUT_DIR / "seed_scatter_vs_gaussian.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary(
    summary: Sequence[Dict[str, Any]],
    win_rates: Sequence[Dict[str, Any]],
    paired: Sequence[Dict[str, Any]],
) -> None:
    print("\n=== Kernel ranking (APE mean, median [IQR]) ===")
    for row in summary:
        med = row["ape_mean_m_median"]
        q25 = row["ape_mean_m_q25"]
        q75 = row["ape_mean_m_q75"]
        mse_med = row["mse_m2_median"]
        print(f"  {row['kernel']:<16} APE {med:.4f} [{q25:.4f}, {q75:.4f}]   MSE {mse_med:.4f}")

    print(f"\n=== Win rate vs {BASELINE_KERNEL} (lower APE) ===")
    for row in win_rates:
        print(f"  {row['kernel']:<16} {row['wins']}/{row['wins']+row['losses']+row['ties']} wins ({100*row['win_rate']:.1f}%)")

    print(f"\n=== Paired tests vs {BASELINE_KERNEL} (positive diff = kernel better) ===")
    for row in paired:
        print(
            f"  {row['kernel']:<16} ΔAPE median {row['median_ape_diff_m']:+.4f} m   "
            f"Wilcoxon p={row['wilcoxon_p']:.4g}   t-test p={row['ttest_p']:.4g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tutorial_w_references batch outputs.")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"References experiment root (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--baseline",
        default=BASELINE_KERNEL,
        help=f"Baseline kernel name (default: {BASELINE_KERNEL})",
    )
    return parser.parse_args()


def run_evaluation(
    results_root: Path,
    output_dir: Optional[Path] = None,
    baseline: str = BASELINE_KERNEL,
) -> None:
    global OUTPUT_DIR, BASELINE_KERNEL
    if output_dir is not None:
        OUTPUT_DIR = output_dir.resolve()
    BASELINE_KERNEL = baseline
    results_root = results_root.resolve()

    _style()
    print(f"Evaluating references results in {results_root}")

    rows = evaluate_batch(results_root)
    summary = summarize_kernels(rows)
    rel = relative_improvement_vs_baseline(rows, BASELINE_KERNEL)
    wins = win_rates_vs_baseline(rows, BASELINE_KERNEL)
    paired = paired_tests_vs_baseline(rows, BASELINE_KERNEL)
    hard = identify_hard_seeds(rows, BASELINE_KERNEL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "references_enriched.csv", rows)
    write_csv(OUTPUT_DIR / "references_kernel_summary.csv", summary)
    write_csv(OUTPUT_DIR / "references_relative_improvement.csv", rel)
    write_csv(OUTPUT_DIR / "references_win_rates.csv", wins)
    write_csv(OUTPUT_DIR / "references_paired_tests.csv", paired)
    write_csv(
        OUTPUT_DIR / "references_hard_seeds.csv",
        [
            {
                "test_idx": h["test_idx"],
                "baseline_ape_m": h["baseline_ape_m"],
                "best_kernel": h["best_kernel"],
                "best_ape_m": h["best_ape_m"],
                "hard_for_baseline": h["hard_for_baseline"],
            }
            for h in hard
        ],
    )

    print_summary(summary, wins, paired)
    plot_ape_violin(rows, summary)
    plot_relative_improvement(rel)
    plot_seed_scatter(rows, BASELINE_KERNEL)
    print(f"\nOutputs written to {OUTPUT_DIR}")


def main() -> None:
    args = parse_args()
    run_evaluation(args.results, args.output, args.baseline)
    print("Done.")


if __name__ == "__main__":
    main()
