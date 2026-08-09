#!/usr/bin/env python3
"""
Paper figures: GGD vs Gaussian and common SLAM robust kernels.

Demonstrates that GGD (high β) approximates a set-theoretic / uniform-on-a-set
view (flat core, bounded influence), unlike smooth M-estimators.

Run from repo root:
  python python/diagram_plotters/plot_ggd_robust_comparison.py

Outputs PNGs under python/diagram_plotters/output/ (created automatically).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from diagram_plotters.robust_cost_functions import (
        cauchy,
        cauchy_deriv,
        dcs,
        gaussian,
        geman_mcclure,
        geman_mcclure_deriv,
        ggd_edge,
        ggd_kernel,
        ggd_kernel_deriv,
        huber,
        huber_deriv,
        rho_2d_isotropic,
        saturated,
        set_indicator_cost,
        tukey,
        tukey_deriv,
        welsch,
        welsch_deriv,
    )
except ModuleNotFoundError:
    from robust_cost_functions import (
        cauchy,
        cauchy_deriv,
        dcs,
        gaussian,
        geman_mcclure,
        geman_mcclure_deriv,
        ggd_edge,
        ggd_kernel,
        ggd_kernel_deriv,
        huber,
        huber_deriv,
        rho_2d_isotropic,
        saturated,
        set_indicator_cost,
        tukey,
        tukey_deriv,
        welsch,
        welsch_deriv,
    )

SIGMA = 3.0
BETA_LOW = 6.0
BETA_HIGH = 100.0
LNC = 1e-3

# Match GGD σ for other kernels (g2o δ in residual units).
DELTA = SIGMA
# DCS φ in g2o is on e² scale; use φ = σ² so the transition is comparable.
DCS_PHI = SIGMA * SIGMA

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _style():
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


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")


def plot_cost_vs_residual() -> None:
    """ρ(r²) vs residual r for 1D unit-information factor."""
    r = np.linspace(-5.0 * SIGMA, 5.0 * SIGMA, 800)
    e2 = r * r

    curves = [
        ("Gaussian (L2)", gaussian(e2), dict(color="black", lw=2.0, ls="-")),
        (f"GGD kernel ($\\sigma$={SIGMA}, $\\beta$={BETA_LOW})", ggd_kernel(e2, SIGMA, BETA_LOW, LNC),
         dict(color="#1f77b4", lw=2.0)),
        (f"GGD kernel ($\\sigma$={SIGMA}, $\\beta$={BETA_HIGH})", ggd_kernel(e2, SIGMA, BETA_HIGH, LNC),
         dict(color="#0b4f8a", lw=2.0, ls="--")),
        (f"GGD edge ($\\beta$={BETA_LOW}, $\\Omega/4$)", ggd_edge(e2, BETA_LOW, LNC),
         dict(color="#1f77b4", lw=1.2, ls=":", alpha=0.8)),
        ("Huber", huber(e2, DELTA), dict(color="#ff7f0e", lw=1.6)),
        ("Cauchy", cauchy(e2, DELTA), dict(color="#2ca02c", lw=1.6)),
        ("Geman–McClure", geman_mcclure(e2, DELTA), dict(color="#9467bd", lw=1.6)),
        ("Welsch", welsch(e2, DELTA), dict(color="#8c564b", lw=1.6)),
        ("Tukey (bisquare)", tukey(e2, DELTA), dict(color="#e377c2", lw=1.6)),
        ("Saturated (hard cap)", saturated(e2, DELTA), dict(color="#7f7f7f", lw=1.6, ls="-.")),
        ("DCS", dcs(e2, DCS_PHI), dict(color="#bcbd22", lw=1.6)),
        ("Set indicator (ideal)", set_indicator_cost(r, SIGMA, outside=SIGMA ** BETA_LOW / 2.0),
         dict(color="#d62728", lw=1.2, ls=(0, (4, 2)), alpha=0.9)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax in axes:
        for label, y, kw in curves:
            ax.plot(r, y, label=label, **kw)
        ax.axvline(-SIGMA, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.axvline(SIGMA, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel(r"residual $r$ (unit information)")
        ax.set_ylabel(r"robust cost $\rho(r^2)$")
        ax.grid(True, alpha=0.25)

    axes[0].set_title("Full range")
    axes[0].set_xlim(-5 * SIGMA, 5 * SIGMA)
    axes[0].legend(loc="upper center", ncol=2, framealpha=0.92)

    axes[1].set_title(f"Core region ($|r| \\leq 2\\sigma$) — flat top of GGD")
    axes[1].set_xlim(-2 * SIGMA, 2 * SIGMA)
    ymax = ggd_kernel(SIGMA * SIGMA, SIGMA, BETA_LOW, LNC) * 1.15
    axes[1].set_ylim(-0.02, ymax)

    fig.suptitle(
        "Robust costs: GGD flat core vs smooth M-estimators (g2o formulations, $\\delta=\\sigma$)",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "robust_cost_1d.png")
    plt.close(fig)


def plot_influence_weights() -> None:
    """IRLS weight ρ'(r²) — how each kernel down-weights large residuals."""
    r = np.linspace(0.0, 5.0 * SIGMA, 600)
    e2 = r * r

    curves = [
        ("Gaussian", np.ones_like(r), dict(color="black", lw=2)),
        (f"GGD $\\beta$={BETA_LOW}", ggd_kernel_deriv(e2, SIGMA, BETA_LOW), dict(color="#1f77b4", lw=2)),
        (f"GGD $\\beta$={BETA_HIGH}", ggd_kernel_deriv(e2, SIGMA, BETA_HIGH), dict(color="#0b4f8a", lw=2, ls="--")),
        ("Huber", huber_deriv(e2, DELTA), dict(color="#ff7f0e", lw=1.6)),
        ("Cauchy", cauchy_deriv(e2, DELTA), dict(color="#2ca02c", lw=1.6)),
        ("Geman–McClure", geman_mcclure_deriv(e2, DELTA), dict(color="#9467bd", lw=1.6)),
        ("Welsch", welsch_deriv(e2, DELTA), dict(color="#8c564b", lw=1.6)),
        ("Tukey", tukey_deriv(e2, DELTA), dict(color="#e377c2", lw=1.6)),
        ("Saturated", (e2 <= DELTA * DELTA).astype(float), dict(color="#7f7f7f", lw=1.6, ls="-.")),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, y, kw in curves:
        ax.plot(r, y, label=label, **kw)
    ax.axvline(SIGMA, color="gray", ls=":", lw=0.8, label=f"$r=\\sigma$ ({SIGMA})")
    ax.set_xlabel(r"$|r|$")
    ax.set_ylabel(r"weight $\rho'(r^2)$")
    ax.set_title("Influence functions: GGD stays active in core, then drops; Tukey/Saturated hard-zero")
    ax.set_xlim(0, 5 * SIGMA)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    _save(fig, "robust_weights_1d.png")
    plt.close(fig)


def plot_tail_log() -> None:
    """Log-scale tails: GGD power growth vs logarithmic Cauchy."""
    r = np.linspace(0.5, 6.0 * SIGMA, 400)
    e2 = r * r

    fig, ax = plt.subplots(figsize=(8, 4.5))
    series = [
        ("Gaussian", gaussian(e2)),
        (f"GGD $\\beta$={BETA_LOW}", ggd_kernel(e2, SIGMA, BETA_LOW, LNC)),
        (f"GGD $\\beta$={BETA_HIGH}", ggd_kernel(e2, SIGMA, BETA_HIGH, LNC)),
        ("Cauchy", cauchy(e2, DELTA)),
        ("Huber", huber(e2, DELTA)),
        ("Geman–McClure", geman_mcclure(e2, DELTA)),
    ]
    for label, y in series:
        ax.semilogy(r, np.maximum(y, 1e-12), label=label, lw=1.8)
    ax.axvline(SIGMA, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$|r|$")
    ax.set_ylabel(r"$\rho(r^2)$ (log scale)")
    ax.set_title("Tail behaviour beyond $\\sigma$: power-like (GGD) vs bounded/log (common robust kernels)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, "robust_tails_log.png")
    plt.close(fig)


def plot_2d_level_sets() -> None:
    """2D isotropic level sets: GGD high-β approaches a box (set), Gaussian is round."""
    lim = 4.0 * SIGMA
    n = 200
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    ex, ey = np.meshgrid(xs, ys)

    levels_gauss = [0.5, 2.0, 8.0, 32.0]
    levels_ggd_low = [
        ggd_kernel(SIGMA * SIGMA * 0.25, SIGMA, BETA_LOW, LNC),
        ggd_kernel(SIGMA * SIGMA, SIGMA, BETA_LOW, LNC),
        ggd_kernel((2 * SIGMA) ** 2, SIGMA, BETA_LOW, LNC),
    ]
    levels_ggd_high = levels_ggd_low
    levels_cauchy = [cauchy(0.25 * SIGMA * SIGMA, DELTA), cauchy(SIGMA * SIGMA, DELTA), cauchy(4 * SIGMA * SIGMA, DELTA)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [
        ("Gaussian (L2)", gaussian, {}, levels_gauss),
        (f"GGD $\\sigma$={SIGMA}, $\\beta$={BETA_LOW}", ggd_kernel,
         dict(sigma=SIGMA, beta=BETA_LOW, lnc=LNC), levels_ggd_low),
        (f"GGD $\\sigma$={SIGMA}, $\\beta$={BETA_HIGH}", ggd_kernel,
         dict(sigma=SIGMA, beta=BETA_HIGH, lnc=LNC), levels_ggd_high),
    ]

    for ax, (title, fn, kw, levels) in zip(axes, panels):
        z = rho_2d_isotropic(ex, ey, fn, **kw)
        cs = ax.contour(xs, ys, z, levels=levels, colors="C0", linewidths=1.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2g")
        ax.contour(xs, ys, z, levels=[levels[-1]], colors="C0", linewidths=2.5)
        # Reference σ-boundary (set membership)
        ax.plot(
            [-SIGMA, SIGMA, SIGMA, -SIGMA, -SIGMA],
            [-SIGMA, -SIGMA, SIGMA, SIGMA, -SIGMA],
            "r--",
            lw=1.2,
            alpha=0.7,
            label=f"$|r_i|\\leq\\sigma$ box",
        )
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(r"$r_x$")
        ax.set_ylabel(r"$r_y$")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    # Cauchy comparison in a second row figure
    fig2, ax2 = plt.subplots(figsize=(4.5, 4.2))
    z = rho_2d_isotropic(ex, ey, cauchy, delta=DELTA)
    cs = ax2.contour(xs, ys, z, levels=levels_cauchy, colors="C2", linewidths=1.5)
    ax2.clabel(cs, inline=True, fontsize=8, fmt="%.2g")
    ax2.plot(
        [-SIGMA, SIGMA, SIGMA, -SIGMA, -SIGMA],
        [-SIGMA, -SIGMA, SIGMA, SIGMA, -SIGMA],
        "r--",
        lw=1.2,
        alpha=0.7,
        label=f"$\\sigma$ box",
    )
    ax2.set_aspect("equal")
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel(r"$r_x$")
    ax2.set_ylabel(r"$r_y$")
    ax2.set_title(f"Cauchy ($\\delta={DELTA}$)")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    _save(fig2, "robust_level_sets_cauchy.png")
    plt.close(fig2)

    fig.suptitle("2D level sets (unit $\\Omega$): GGD $\\beta\\uparrow$ squashes contours toward a set / box", y=1.02)
    fig.tight_layout()
    _save(fig, "robust_level_sets.png")
    plt.close(fig)


def plot_set_limit() -> None:
    """GGD β → ∞ limit vs ideal set indicator and uniform-on-set intuition."""
    r = np.linspace(-1.5 * SIGMA, 1.5 * SIGMA, 500)
    e2 = r * r
    betas = [4, 6, 20, 100, 500]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for b in betas:
        ax.plot(r, ggd_kernel(e2, SIGMA, b, LNC), label=f"GGD $\\beta={b}$", lw=1.8 if b == BETA_HIGH else 1.2)
    ax.plot(
        r,
        set_indicator_cost(r, SIGMA, outside=np.max(ggd_kernel(SIGMA * SIGMA, SIGMA, BETA_HIGH, LNC))),
        "k:",
        lw=2,
        label="Set indicator (ideal)",
    )
    ax.axvspan(-SIGMA, SIGMA, color="gold", alpha=0.15, label=f"Set $|r|\\leq\\sigma$")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\rho(r^2)$")
    ax.set_title(f"GGD family at $\\sigma={SIGMA}$: increasing $\\beta \\to$ flat uniform-like core (set view)")
    ax.legend(loc="upper center", ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "ggd_set_limit.png")
    plt.close(fig)


def main() -> None:
    _style()
    print(f"GGD paper plots → {OUTPUT_DIR}")
    print(f"  σ={SIGMA}, β∈{{{BETA_LOW}, {BETA_HIGH}}}, other kernels δ={DELTA}")
    plot_cost_vs_residual()
    plot_influence_weights()
    plot_tail_log()
    plot_2d_level_sets()
    plot_set_limit()
    print("Done.")


if __name__ == "__main__":
    main()
