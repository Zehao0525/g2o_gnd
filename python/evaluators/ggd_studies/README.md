# ggd_studies

Offline metrics and paper figures for **GGD / GND** robust-kernel experiments on Tutorial_slam2d (GPS-like / absolute priors), including beta sweeps and GPS-period sweeps.

Default C++ result trees live under `test_results/gnd_beta_study/`. Script outputs go to `output/` next to these scripts.

| Script | Role |
|--------|------|
| `g2o_io.py` | Shared Tutorial SE2 g2o I/O + trajectory-ordered translation APE/MSE |
| `evaluate_references.py` | Multi-kernel ranking on `references_robust_kernels` (APE/MSE, win rate, paired tests, seed plots) |
| `batch_visualizer2.py` | Thin wrapper: full eval + optional trajectory overlay for one seed/kernel |
| `make_kernel_comparison_table.py` | Per-period LaTeX/CSV kernel comparison tables vs Gaussian |
| `evaluate_gps_period_sweep.py` | Run kernel tables over `gps_period_sweep/period_{P}/` |
| `summarize_gps_period_gaussian_vs_gnd.py` | Cross-period Gaussian vs GGD APE summary (wins + t / Wilcoxon) |
| `plot_gps_period_ape_diff.py` | Box plot of APE difference (Gaussian − GGD) across periods |
| `evaluate_beta_sweep.py` | Beta oscillation / convergence study (`optimization_trace.csv`, aggregate APE) |

## Typical inputs

```text
test_results/gnd_beta_study/
  references_robust_kernels/          # or references_robust_kernels_period_{P}/
    test_*/twb_{gauss,gnd,...}.g2o, twb_gt.g2o, kernel_summary.csv
  gps_period_sweep/period_{P}/test_*/...
  correlated_gps(_tight)/             # beta sweep aggregates + traces
```

Periods 1–2 often use the legacy `references_robust_kernels_period_{P}` trees; the summarize/plot scripts cap those to the **first 30 seeds** so they match the 4–30 sweep size.

## Run (from repo root)

```bash
# Multi-kernel batch + plots
python3 python/evaluators/ggd_studies/evaluate_references.py
python3 python/evaluators/ggd_studies/batch_visualizer2.py --plot-test 0 --plot-kernel gnd

# Paper tables for selected GPS periods
python3 python/evaluators/ggd_studies/make_kernel_comparison_table.py --periods 1 2 30

# Full GPS-period sweep tables
python3 python/evaluators/ggd_studies/evaluate_gps_period_sweep.py

# Gaussian vs GGD across periods + difference plot
python3 python/evaluators/ggd_studies/summarize_gps_period_gaussian_vs_gnd.py
python3 python/evaluators/ggd_studies/plot_gps_period_ape_diff.py

# Beta / convergence study
python3 python/evaluators/ggd_studies/evaluate_beta_sweep.py
```

Most scripts accept `--results` / `--sweep-root` / `--output` overrides. Artifacts land under:

- `output/references/` — ranking plots, per-kernel CSVs
- `output/references/paper_tables/` — LaTeX kernel tables
- `output/gps_period_sweep/` — period summaries and APE-diff figures
- `output/` — beta-sweep enriched CSVs and oscillation figures
