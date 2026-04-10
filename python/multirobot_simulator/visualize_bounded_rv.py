#!/usr/bin/env python3
"""
Visualise `bounded_rv` from simulator.py: histogram vs exact marginal PDF and N(0, σ²).

Run from repo root:
  python -m multirobot_simulator.visualize_bounded_rv

Or directly:
  python python/multirobot_simulator/visualize_bounded_rv.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    from multirobot_simulator.simulator import bounded_rv
except ModuleNotFoundError:
    from simulator import bounded_rv


def marginal_pdf_disk_projection(x: np.ndarray, R: float) -> np.ndarray:
    """
    Marginal density of X = ρ cos Θ when (ρ, Θ) is uniform on a disk of radius R.

    p(x) = 2 * sqrt(R^2 - x^2) / (π R^2)  for |x| < R, else 0.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    m = np.abs(x) < R
    out[m] = 2.0 * np.sqrt(R * R - x[m] * x[m]) / (np.pi * R * R)
    return out


def plot_bounded_rv(
    mu: float = 0.0,
    sigma: float = 1.0,
    n_samples: int = 50_000,
    seed: int = 0,
    bins: int = 80,
    save_path: str | None = "bounded_rv_visualization.png",
) -> None:
    R = 2.0 * sigma
    rng = np.random.default_rng(seed)
    samples = np.asarray(bounded_rv(mu=mu, sigma=sigma, n_samples=n_samples, rng=rng), dtype=float)

    xs = np.linspace(mu - R * 1.001, mu + R * 1.001, 500)
    pdf_b = marginal_pdf_disk_projection(xs - mu, R)
    pdf_g = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((xs - mu) / sigma) ** 2
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        samples,
        bins=bins,
        density=True,
        alpha=0.55,
        color="steelblue",
        label=f"bounded_rv samples (n={n_samples})",
    )
    ax.plot(xs, pdf_b, "b-", lw=2, label="exact marginal (uniform disk → x)")
    ax.plot(
        xs,
        pdf_g,
        "r--",
        lw=2,
        alpha=0.85,
        label=r"$\mathcal{N}(\mu,\sigma^2)$ same $\sigma$ (unbounded)",
    )
    ax.axvline(mu - R, color="gray", ls=":", lw=1)
    ax.axvline(mu + R, color="gray", ls=":", lw=1)
    ax.axvline(mu, color="k", ls="-", lw=0.8, alpha=0.5)
    ax.set_title(
        rf"bounded_rv: $\mu$={mu}, $\sigma$={sigma}, support $[\mu-2\sigma,\mu+2\sigma]$"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend(loc="upper center", fontsize=9)
    ax.set_xlim(mu - R * 1.15, mu + R * 1.15)
    fig.tight_layout()

    if save_path:
        out = str(save_path)
        plt.savefig(out, dpi=150)
        print(f"Saved figure: {out}")
    plt.show()


def main() -> None:
    plot_bounded_rv(
        mu=0.0,
        sigma=1.0,
        n_samples=50_000,
        seed=42,
        save_path="bounded_rv_visualization.png",
    )


if __name__ == "__main__":
    main()
