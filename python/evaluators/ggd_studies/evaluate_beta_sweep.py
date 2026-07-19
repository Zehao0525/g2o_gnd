#!/usr/bin/env python3
"""
Evaluate GND beta oscillation / convergence study outputs.

Primary inputs:
  - optimization_trace.csv   per-iteration chi2, rel_gain, lm_inners, chi2_increased
  - beta_sweep_aggregate.csv scalar summaries + final pose graphs for APE

Writes enriched tables, trace summaries, and oscillation-focused figures.

Run from repo root:
  python python/evaluators/gnd_studies/evaluate_beta_sweep.py
  python python/evaluators/gnd_studies/evaluate_beta_sweep.py --results test_results/gnd_beta_study/correlated_gps
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from g2o_io import beta_output_stem, compute_translation_ape_from_g2o

DEFAULT_RESULTS_ROOT = Path("test_results/gnd_beta_study/correlated_gps_tight")
OUTPUT_DIR = SCRIPT_DIR / "output"

AGGREGATE_NUMERIC = [
    "mean_pose_translation_error",
    "chi2_initial",
    "chi2_after_warmup",
    "chi2_final",
    "chi2_gain_warmup",
    "chi2_gain_active",
    "warmup_outer_iters",
    "warmup_lm_inners",
    "warmup_time_s",
    "warmup_non_monotone_steps",
    "warmup_iters_to_tol",
    "warmup_chi2_sign_flips",
    "warmup_max_chi2_spike",
    "warmup_chi2_range",
    "active_outer_iters",
    "active_lm_inners",
    "active_time_s",
    "active_non_monotone_steps",
    "active_iters_to_tol",
    "active_chi2_sign_flips",
    "active_max_chi2_spike",
    "active_chi2_range",
    "total_outer_iters",
    "total_lm_inners",
    "total_time_s",
    "ape_mean_m",
    "ape_rmse_m",
    "ape_max_m",
]

AGGREGATE_BOOL = [
    "warmup_converged",
    "warmup_hit_max_iters",
    "warmup_solver_failed",
    "active_converged",
    "active_hit_max_iters",
    "active_solver_failed",
    "solver_failed",
]

OSCILLATION_SUMMARY_METRICS = [
    "active_iters_to_tol",
    "active_outer_iters",
    "warmup_outer_iters",
    "total_outer_iters",
    "total_lm_inners",
    "active_non_monotone_steps",
    "active_chi2_sign_flips",
    "active_max_chi2_spike",
    "active_chi2_range",
    "warmup_iters_to_tol",
    "warmup_non_monotone_steps",
    "warmup_chi2_sign_flips",
    "chi2_final",
    "ape_mean_m",
    "ape_rmse_m",
    "total_time_s",
]

TRACE_RUN_FIELDS = [
    "final_iter",
    "final_chi2",
    "chi2_range",
    "max_chi2_spike",
    "non_monotone_steps",
    "chi2_sign_flips",
    "num_chi2_increases",
    "mean_rel_gain",
    "min_rel_gain",
]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.family": "serif",
        }
    )


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _parse_float(value: str) -> float:
    value = value.strip()
    if not value:
        return float("nan")
    return float(value)


def _parse_int(value: str) -> int:
    value = value.strip()
    if not value:
        return -1
    return int(float(value))


def condition_label(variant: str, beta: Optional[float]) -> str:
    if variant == "gaussian":
        return "Gaussian"
    if beta is None or np.isnan(beta):
        return "GND"
    if abs(beta - round(beta)) < 1e-9:
        return f"GND β={int(round(beta))}"
    return f"GND β={beta:g}"


def condition_sort_key(variant: str, beta: Optional[float]) -> Tuple[int, float]:
    if variant == "gaussian":
        return (0, 0.0)
    if beta is None or np.isnan(beta):
        return (1, float("inf"))
    return (1, float(beta))


def load_aggregate_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Aggregate CSV not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            variant = raw["variant"].strip()
            beta = _parse_float(raw["beta"]) if raw["beta"].strip() else None
            row: Dict[str, Any] = {
                "test_idx": int(raw["test_idx"]),
                "variant": variant,
                "beta": beta,
                "condition": condition_label(variant, beta),
            }
            for col in AGGREGATE_NUMERIC:
                if col in raw:
                    row[col] = _parse_float(raw[col])
            for col in AGGREGATE_BOOL:
                if col in raw:
                    row[col] = _parse_bool(raw[col])
            rows.append(row)
    return rows


def load_optimization_trace(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            variant = raw["variant"].strip()
            beta = _parse_float(raw["beta"]) if raw["beta"].strip() else None
            rows.append(
                {
                    "test_idx": int(raw["test_idx"]),
                    "variant": variant,
                    "beta": beta,
                    "condition": condition_label(variant, beta),
                    "phase": raw["phase"].strip(),
                    "iter": int(raw["iter"]),
                    "chi2": _parse_float(raw["chi2"]),
                    "rel_gain": _parse_float(raw["rel_gain"]),
                    "lm_inners": _parse_int(raw["lm_inners"]),
                    "chi2_increased": _parse_bool(raw["chi2_increased"]),
                }
            )
    return rows


def estimate_g2o_path(results_root: Path, test_idx: int, row: Dict[str, Any]) -> Path:
    test_dir = results_root / f"test_{test_idx}"
    if row["variant"] == "gaussian":
        return test_dir / "gaussian.g2o"
    beta = row["beta"]
    if beta is None or np.isnan(beta):
        raise ValueError(f"Missing beta for gnd row in test {test_idx}")
    return test_dir / f"{beta_output_stem(beta)}.g2o"


def enrich_with_ape(results_root: Path, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        test_idx = int(row["test_idx"])
        gt_path = results_root / f"test_{test_idx}" / "gt.g2o"
        est_path = estimate_g2o_path(results_root, test_idx, row)

        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing ground truth: {gt_path}")
        if not est_path.is_file():
            raise FileNotFoundError(f"Missing estimate graph: {est_path}")

        ape_mean, ape_rmse, ape_max, n_pairs = compute_translation_ape_from_g2o(gt_path, est_path)
        out["ape_mean_m"] = ape_mean
        out["ape_rmse_m"] = ape_rmse
        out["ape_max_m"] = ape_max
        out["ape_num_pairs"] = n_pairs
        enriched.append(out)
    return enriched


def _aggregate_metric(values: np.ndarray) -> Dict[str, float]:
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


def _metric_values(group: Sequence[Dict[str, Any]], metric: str) -> np.ndarray:
    return np.array([float(r.get(metric, np.nan)) for r in group], dtype=float)


def _rate_values(group: Sequence[Dict[str, Any]], metric: str) -> float:
    present = [r.get(metric) for r in group if metric in r]
    if not present:
        return float("nan")
    return float(np.mean([1.0 if bool(v) else 0.0 for v in present]))


def summarize_by_condition(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    summary: List[Dict[str, Any]] = []
    for condition in sorted(
        by_condition.keys(),
        key=lambda c: condition_sort_key(by_condition[c][0]["variant"], by_condition[c][0]["beta"]),
    ):
        group = by_condition[condition]
        rep = group[0]
        entry: Dict[str, Any] = {
            "condition": condition,
            "variant": rep["variant"],
            "beta": rep["beta"],
            "num_tests": len(group),
            "solver_fail_rate": _rate_values(group, "solver_failed"),
        }
        for metric in OSCILLATION_SUMMARY_METRICS:
            stats = _aggregate_metric(_metric_values(group, metric))
            for stat_name, value in stats.items():
                entry[f"{metric}_{stat_name}"] = value
        summary.append(entry)
    return summary


def _trace_run_key(row: Dict[str, Any]) -> Tuple[int, str, Optional[float], str]:
    return (row["test_idx"], row["variant"], row["beta"], row["phase"])


def group_trace_runs(trace_rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, str, Optional[float], str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[int, str, Optional[float], str], List[Dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        grouped[_trace_run_key(row)].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda r: r["iter"])
    return grouped


def compute_trace_run_metrics(series: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not series:
        return {}

    chi2_vals = [r["chi2"] for r in series if np.isfinite(r["chi2"])]
    rel_gains = [r["rel_gain"] for r in series if np.isfinite(r["rel_gain"])]

    prev_chi2 = chi2_vals[0] if chi2_vals else float("nan")
    prev_delta_sign = 0
    sign_flips = 0
    non_monotone = 0
    max_spike = 0.0

    for i, row in enumerate(series):
        if i == 0 or not np.isfinite(row["chi2"]):
            continue
        chi2 = row["chi2"]
        if chi2 > prev_chi2 * (1.0 + 1e-9):
            non_monotone += 1
            max_spike = max(max_spike, chi2 - prev_chi2)
        delta = chi2 - prev_chi2
        delta_sign = 1 if delta > 1e-9 else (-1 if delta < -1e-9 else 0)
        if delta_sign != 0 and prev_delta_sign != 0 and delta_sign != prev_delta_sign:
            sign_flips += 1
        if delta_sign != 0:
            prev_delta_sign = delta_sign
        prev_chi2 = chi2

    rep = series[0]
    return {
        "test_idx": rep["test_idx"],
        "variant": rep["variant"],
        "beta": rep["beta"],
        "condition": rep["condition"],
        "phase": rep["phase"],
        "final_iter": series[-1]["iter"],
        "final_chi2": chi2_vals[-1] if chi2_vals else float("nan"),
        "chi2_range": (max(chi2_vals) - min(chi2_vals)) if chi2_vals else float("nan"),
        "max_chi2_spike": max_spike,
        "non_monotone_steps": non_monotone,
        "chi2_sign_flips": sign_flips,
        "num_chi2_increases": int(sum(1 for r in series if r.get("chi2_increased"))),
        "mean_rel_gain": float(np.mean(rel_gains)) if rel_gains else float("nan"),
        "min_rel_gain": float(np.min(rel_gains)) if rel_gains else float("nan"),
    }


def summarize_trace_runs(trace_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = group_trace_runs(trace_rows)
    return [compute_trace_run_metrics(series) for series in grouped.values()]


def summarize_trace_by_condition(trace_run_summary: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: DefaultDict[Tuple[str, Optional[float], str], List[Dict[str, Any]]] = defaultdict(list)
    for row in trace_run_summary:
        by_key[(row["condition"], row["beta"], row["phase"])].append(row)

    out: List[Dict[str, Any]] = []
    for (condition, beta, phase), group in sorted(
        by_key.items(),
        key=lambda item: (condition_sort_key(item[0][0], item[0][1]), item[0][2]),
    ):
        entry: Dict[str, Any] = {
            "condition": condition,
            "beta": beta,
            "phase": phase,
            "num_runs": len(group),
        }
        for metric in TRACE_RUN_FIELDS:
            stats = _aggregate_metric(_metric_values(group, metric))
            for stat_name, value in stats.items():
                entry[f"{metric}_{stat_name}"] = value
        out.append(entry)
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            beta = out.get("beta")
            if beta is None or (isinstance(beta, float) and np.isnan(beta)):
                out["beta"] = ""
            writer.writerow(out)


def write_enriched_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "test_idx",
        "variant",
        "beta",
        "condition",
        "ape_mean_m",
        "ape_rmse_m",
        "ape_max_m",
        "ape_num_pairs",
        *AGGREGATE_NUMERIC,
        *AGGREGATE_BOOL,
    ]
    write_csv(path, rows, fieldnames)


def write_summary_csv(path: Path, summary: Sequence[Dict[str, Any]]) -> None:
    if not summary:
        return
    write_csv(path, summary, list(summary[0].keys()))


def _condition_order(summary: Sequence[Dict[str, Any]]) -> List[str]:
    indexed = [
        (condition_sort_key(r["variant"], r["beta"]), r["condition"]) for r in summary
    ]
    indexed.sort()
    return [c for _, c in indexed]


def _extract(summary: Sequence[Dict[str, Any]], condition: str, field: str) -> float:
    for row in summary:
        if row["condition"] == condition:
            return float(row.get(field, np.nan))
    return float("nan")


def _gnd_only(summary: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in summary if r["variant"] != "gaussian"]


def plot_chi2_median_bands(
    trace_rows: Sequence[Dict[str, Any]],
    phase: str,
    filename: str,
    max_iter: Optional[int] = None,
) -> None:
    phase_rows = [r for r in trace_rows if r["phase"] == phase and np.isfinite(r["chi2"])]
    if not phase_rows:
        print(f"Skipping {filename}: no {phase}-phase trace rows")
        return

    by_condition: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in phase_rows:
        by_condition[row["condition"]].append(row)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(by_condition), 1)))

    for color, (condition, rows) in zip(colors, sorted(by_condition.items(), key=lambda kv: condition_sort_key(kv[1][0]["variant"], kv[1][0]["beta"]))):
        by_seed: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_seed[row["test_idx"]].append(row)

        iter_max = max_iter
        if iter_max is None:
            iter_max = max(r["iter"] for r in rows)

        medians: List[float] = []
        q25: List[float] = []
        q75: List[float] = []
        xs: List[int] = []
        for it in range(0, iter_max + 1):
            vals = []
            for series in by_seed.values():
                match = [r["chi2"] for r in series if r["iter"] == it and np.isfinite(r["chi2"])]
                if match:
                    vals.append(match[0])
            if not vals:
                continue
            xs.append(it)
            medians.append(float(np.median(vals)))
            q25.append(float(np.percentile(vals, 25)))
            q75.append(float(np.percentile(vals, 75)))

        if not xs:
            continue
        med = np.asarray(medians)
        ax.plot(xs, med, lw=2.0, color=color, label=condition)
        ax.fill_between(xs, q25, q75, color=color, alpha=0.18)

    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("χ²")
    ax.set_yscale("log")
    ax.set_title(f"{phase.capitalize()}-phase χ² trajectories (median ± IQR over seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_chi2_traces_selected_seeds(
    trace_rows: Sequence[Dict[str, Any]],
    test_indices: Sequence[int],
    phase: str = "active",
    filename: str = "chi2_traces_selected_seeds.png",
) -> None:
    subset = [
        r
        for r in trace_rows
        if r["phase"] == phase and r["test_idx"] in test_indices and np.isfinite(r["chi2"])
    ]
    if not subset:
        print(f"Skipping {filename}: no trace rows for seeds {list(test_indices)}")
        return

    by_key: DefaultDict[Tuple[int, str, Optional[float]], List[Dict[str, Any]]] = defaultdict(list)
    for row in subset:
        by_key[(row["test_idx"], row["variant"], row["beta"])].append(row)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (test_idx, variant, beta), series in sorted(by_key.items()):
        series = sorted(series, key=lambda r: r["iter"])
        xs = [r["iter"] for r in series]
        ys = [r["chi2"] for r in series]
        label = f"seed {test_idx}, {condition_label(variant, beta)}"
        ax.plot(xs, ys, lw=1.3, alpha=0.85, label=label)

    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("χ²")
    ax.set_yscale("log")
    ax.set_title(f"{phase.capitalize()}-phase χ² trajectories (selected seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_rel_gain_traces(
    trace_rows: Sequence[Dict[str, Any]],
    test_indices: Sequence[int],
    phase: str = "active",
    filename: str = "rel_gain_traces_selected_seeds.png",
) -> None:
    subset = [
        r
        for r in trace_rows
        if r["phase"] == phase and r["test_idx"] in test_indices and np.isfinite(r["rel_gain"])
    ]
    if not subset:
        print(f"Skipping {filename}: no rel_gain rows")
        return

    by_key: DefaultDict[Tuple[int, str, Optional[float]], List[Dict[str, Any]]] = defaultdict(list)
    for row in subset:
        by_key[(row["test_idx"], row["variant"], row["beta"])].append(row)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (test_idx, variant, beta), series in sorted(by_key.items()):
        series = sorted(series, key=lambda r: r["iter"])
        xs = [r["iter"] for r in series if r["iter"] > 0]
        ys = [r["rel_gain"] for r in series if r["iter"] > 0]
        if not xs:
            continue
        ax.plot(xs, ys, lw=1.3, alpha=0.85, label=f"seed {test_idx}, {condition_label(variant, beta)}")

    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Relative χ² gain")
    ax.set_title(f"{phase.capitalize()}-phase relative gain (selected seeds)")
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_metric_vs_beta(
    summary: Sequence[Dict[str, Any]],
    metric_base: str,
    ylabel: str,
    title: str,
    filename: str,
    gnd_only: bool = False,
    log_y: bool = False,
) -> None:
    rows = _gnd_only(summary) if gnd_only else summary
    if not rows:
        print(f"Skipping {filename}: empty summary")
        return

    xs = [float(r["beta"]) if r["variant"] != "gaussian" else 2.0 for r in rows]
    medians = [_extract(summary, r["condition"], f"{metric_base}_median") for r in rows]
    q25 = [_extract(summary, r["condition"], f"{metric_base}_q25") for r in rows]
    q75 = [_extract(summary, r["condition"], f"{metric_base}_q75") for r in rows]

    x = np.asarray(xs, dtype=float)
    med = np.asarray(medians, dtype=float)
    err_lo = med - np.asarray(q25, dtype=float)
    err_hi = np.asarray(q75, dtype=float) - med

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.errorbar(x, med, yerr=[err_lo, err_hi], fmt="o-", capsize=4, lw=1.8, markersize=6)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_convergence_panel(summary: Sequence[Dict[str, Any]]) -> None:
    gnd = _gnd_only(summary)
    if not gnd:
        print("Skipping convergence_panel.png: no GND rows")
        return

    metrics = [
        ("active_iters_to_tol", "Iters to tolerance", "Iterations to χ² tolerance"),
        ("active_outer_iters", "Active outer iters", "Active-phase outer iterations"),
        ("total_outer_iters", "Total outer iters", "Total outer iterations (warmup + active)"),
        ("total_lm_inners", "LM inner trials", "Total LM inner trials"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, (field, ylabel, title) in zip(axes.ravel(), metrics):
        xs = [float(r["beta"]) for r in gnd]
        med = [_extract(summary, r["condition"], f"{field}_median") for r in gnd]
        q25 = [_extract(summary, r["condition"], f"{field}_q25") for r in gnd]
        q75 = [_extract(summary, r["condition"], f"{field}_q75") for r in gnd]
        med_arr = np.asarray(med)
        ax.errorbar(
            xs,
            med_arr,
            yerr=[med_arr - np.asarray(q25), np.asarray(q75) - med_arr],
            fmt="o-",
            capsize=3,
            lw=1.5,
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence behaviour vs β", y=1.01)
    fig.tight_layout()
    out = OUTPUT_DIR / "convergence_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _maybe_log_y(ax: plt.Axes, values: Sequence[float]) -> None:
    arr = np.asarray(values, dtype=float)
    if np.any(arr[np.isfinite(arr)] > 0):
        ax.set_yscale("log")


def plot_oscillation_panel(summary: Sequence[Dict[str, Any]]) -> None:
    metrics = [
        ("active_chi2_sign_flips", "χ² sign flips"),
        ("active_non_monotone_steps", "Non-monotone steps"),
        ("active_max_chi2_spike", "Max χ² spike"),
        ("active_chi2_range", "χ² range"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    gnd = _gnd_only(summary)

    for ax, (metric_base, ylabel) in zip(axes.ravel(), metrics):
        xs = [float(r["beta"]) for r in gnd]
        med = [_extract(summary, r["condition"], f"{metric_base}_median") for r in gnd]
        q25 = [_extract(summary, r["condition"], f"{metric_base}_q25") for r in gnd]
        q75 = [_extract(summary, r["condition"], f"{metric_base}_q75") for r in gnd]
        med_arr = np.asarray(med)
        if np.all(~np.isfinite(med_arr)):
            ax.set_visible(False)
            continue
        ax.errorbar(
            xs,
            med_arr,
            yerr=[med_arr - np.asarray(q25), np.asarray(q75) - med_arr],
            fmt="o-",
            capsize=3,
            lw=1.5,
        )
        ax.set_xscale("log")
        if metric_base in {"active_max_chi2_spike", "active_chi2_range"}:
            _maybe_log_y(ax, med)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Oscillation metrics vs β (active phase, median ± IQR)", y=1.01)
    fig.tight_layout()
    out = OUTPUT_DIR / "oscillation_metrics_vs_beta.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_ape_vs_beta(summary: Sequence[Dict[str, Any]]) -> None:
    plot_metric_vs_beta(
        summary,
        "ape_mean_m",
        "APE mean [m]",
        "Absolute Pose Error (mean translation) vs β",
        "ape_mean_vs_beta.png",
        gnd_only=False,
    )
    plot_metric_vs_beta(
        summary,
        "ape_rmse_m",
        "APE RMSE [m]",
        "Absolute Pose Error (RMSE) vs β",
        "ape_rmse_vs_beta.png",
        gnd_only=False,
    )

    conditions = _condition_order(summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (metric_base, ylabel, title) in zip(
        axes,
        [
            ("ape_mean_m", "APE mean [m]", "APE mean vs β"),
            ("ape_rmse_m", "APE RMSE [m]", "APE RMSE vs β"),
        ],
    ):
        xs: List[float] = []
        medians: List[float] = []
        q25: List[float] = []
        q75: List[float] = []
        for cond in conditions:
            rep = next(r for r in summary if r["condition"] == cond)
            xs.append(2.0 if rep["variant"] == "gaussian" else float(rep["beta"]))
            medians.append(_extract(summary, cond, f"{metric_base}_median"))
            q25.append(_extract(summary, cond, f"{metric_base}_q25"))
            q75.append(_extract(summary, cond, f"{metric_base}_q75"))
        med = np.asarray(medians, dtype=float)
        ax.errorbar(
            xs,
            med,
            yerr=[med - np.asarray(q25), np.asarray(q75) - med],
            fmt="o-",
            capsize=4,
            lw=1.8,
            markersize=6,
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta$ (Gaussian ref at marker)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Pose accuracy vs β", y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "ape_vs_beta.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _fmt_float(value: Any, width: int, precision: int = 1) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return " " * (width - 3) + "n/a"
    return f"{float(value):>{width}.{precision}f}"


def print_oscillation_summary_table(
    summary: Sequence[Dict[str, Any]],
    trace_summary: Sequence[Dict[str, Any]],
) -> None:
    print("\n=== GND β oscillation study (median [IQR] over seeds) ===")
    header = (
        f"{'Condition':<14} {'Iters→tol':>10} {'Active':>8} {'Total':>8} "
        f"{'Sign flips':>11} {'Non-mono':>9} {'APE mean':>10} {'APE RMSE':>10}"
    )
    print(header)
    print("-" * len(header))

    gnd = _gnd_only(summary)
    for row in gnd:
        cond = row["condition"]
        print(
            f"{cond:<14} "
            f"{_fmt_float(row.get('active_iters_to_tol_median'), 10)} "
            f"{_fmt_float(row.get('active_outer_iters_median'), 8, 0)} "
            f"{_fmt_float(row.get('total_outer_iters_median'), 8, 0)} "
            f"{_fmt_float(row.get('active_chi2_sign_flips_median'), 11)} "
            f"{_fmt_float(row.get('active_non_monotone_steps_median'), 9)} "
            f"{_fmt_float(row.get('ape_mean_m_median'), 10, 4)} "
            f"{_fmt_float(row.get('ape_rmse_m_median'), 10, 4)}"
        )

    if trace_summary:
        active_trace = [r for r in trace_summary if r["phase"] == "active"]
        if active_trace:
            print("\nTrace-derived active-phase summary:")
            t_header = f"{'Condition':<14} {'Final iter':>11} {'Sign flips':>11} {'χ² range':>12}"
            print(t_header)
            print("-" * len(t_header))
            for row in active_trace:
                if row["condition"] == "Gaussian":
                    continue
                print(
                    f"{row['condition']:<14} "
                    f"{row.get('final_iter_median', np.nan):>11.1f} "
                    f"{row.get('chi2_sign_flips_median', np.nan):>11.1f} "
                    f"{row.get('chi2_range_median', np.nan):>12.2f}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GND beta oscillation study outputs.")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Experiment output directory (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for evaluation artifacts (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--trace-seeds",
        type=int,
        nargs="*",
        default=None,
        help="Seed indices for per-run trace plots (default: first 3 available)",
    )
    parser.add_argument(
        "--max-trace-iter",
        type=int,
        default=None,
        help="Cap iteration axis on median-band plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global OUTPUT_DIR
    OUTPUT_DIR = args.output.resolve()

    results_root = args.results.resolve()
    aggregate_path = results_root / "beta_sweep_aggregate.csv"
    trace_path = results_root / "optimization_trace.csv"

    _style()
    print(f"Evaluating oscillation study results in {results_root}")

    rows = load_aggregate_csv(aggregate_path)
    enriched = enrich_with_ape(results_root, rows)
    summary = summarize_by_condition(enriched)

    trace_rows = load_optimization_trace(trace_path)
    trace_run_summary: List[Dict[str, Any]] = []
    trace_condition_summary: List[Dict[str, Any]] = []
    if trace_rows:
        trace_run_summary = summarize_trace_runs(trace_rows)
        trace_condition_summary = summarize_trace_by_condition(trace_run_summary)
        print(f"Loaded optimization trace: {len(trace_rows)} rows, {len(trace_run_summary)} runs")
    else:
        print(
            "WARNING: optimization_trace.csv not found — oscillation plots will be limited.\n"
            "         Re-run tutorial_beta_sweep after rebuilding the oscillation-study binary."
        )

    write_enriched_csv(OUTPUT_DIR / "beta_sweep_enriched.csv", enriched)
    write_summary_csv(OUTPUT_DIR / "beta_sweep_summary_stats.csv", summary)
    if trace_run_summary:
        write_csv(
            OUTPUT_DIR / "trace_run_summary.csv",
            trace_run_summary,
            ["test_idx", "variant", "beta", "condition", "phase", *TRACE_RUN_FIELDS],
        )
    if trace_condition_summary:
        write_csv(
            OUTPUT_DIR / "trace_condition_summary.csv",
            trace_condition_summary,
            ["condition", "beta", "phase", "num_runs", *[
                f"{m}_{s}" for m in TRACE_RUN_FIELDS for s in ("mean", "median", "q25", "q75")
            ]],
        )

    print(f"Wrote {OUTPUT_DIR / 'beta_sweep_enriched.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'beta_sweep_summary_stats.csv'}")
    if trace_run_summary:
        print(f"Wrote {OUTPUT_DIR / 'trace_run_summary.csv'}")
        print(f"Wrote {OUTPUT_DIR / 'trace_condition_summary.csv'}")

    print_oscillation_summary_table(summary, trace_condition_summary)

    if trace_rows:
        seeds = sorted({r["test_idx"] for r in trace_rows})
        selected = args.trace_seeds if args.trace_seeds is not None else seeds[: min(3, len(seeds))]
        plot_chi2_median_bands(trace_rows, "warmup", "chi2_median_band_warmup.png", args.max_trace_iter)
        plot_chi2_median_bands(trace_rows, "active", "chi2_median_band_active.png", args.max_trace_iter)
        plot_chi2_traces_selected_seeds(trace_rows, selected, phase="active")
        plot_chi2_traces_selected_seeds(trace_rows, selected, phase="warmup", filename="chi2_traces_warmup_selected_seeds.png")
        plot_rel_gain_traces(trace_rows, selected, phase="active")
        plot_rel_gain_traces(trace_rows, selected, phase="warmup", filename="rel_gain_traces_warmup_selected_seeds.png")

    plot_convergence_panel(summary)
    plot_oscillation_panel(summary)
    plot_ape_vs_beta(summary)
    print("Done.")


if __name__ == "__main__":
    main()
