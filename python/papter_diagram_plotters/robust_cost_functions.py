"""
Robust cost functions ρ(e²) used in g2o, plus GGD (kernel and edge forms).

All functions take squared Mahalanobis distance e² = rᵀ Ω r. For scalar plots we use
unit information (Ω = 1), so e² = r² and r is the 1D residual in σ-units after scaling.

GGD kernel (matches src/fght/GGDEdges/ggd_kernel.cpp):
    ρ(e²) = ℓ + (e² / σ²)^(β/2)

GGD edge (matches edge_none_gaussian_unary computeError with ggdSetInformation):
    ρ_edge(e²) = ℓ + (e² / 4)^β   with Ω stored as Ω/4 internally
    Equivalently on unit-Ω edge: ℓ + (e²/4)^β; bound σ is absorbed into the /4 hack.
"""

from __future__ import annotations

import numpy as np


def gaussian(e2: np.ndarray) -> np.ndarray:
    """Standard least-squares / Gaussian NLL (up to scale)."""
    return np.asarray(e2, dtype=float)


def ggd_kernel(e2: np.ndarray, sigma: float, beta: float, lnc: float = 1e-3) -> np.ndarray:
    """GGD robust kernel as implemented in g2o GGDKernel::robustify (rho[0])."""
    e2 = np.asarray(e2, dtype=float)
    scaled = e2 / (sigma * sigma)
    return lnc + np.power(np.maximum(scaled, 0.0), beta / 2.0)


def ggd_edge(e2: np.ndarray, beta: float, lnc: float = 1e-3) -> np.ndarray:
    """GGD dedicated edge: ℓ + (e²/4)^β (ggdSetInformation divides Ω by 4)."""
    e2 = np.asarray(e2, dtype=float)
    scaled = e2 / 4.0
    return lnc + np.power(np.maximum(scaled, 0.0), beta)


def ggd_kernel_deriv(e2: np.ndarray, sigma: float, beta: float) -> np.ndarray:
    """ρ'(e²) — IRLS weight multiplier on the quadratic part (g2o rho[1])."""
    e2 = np.asarray(e2, dtype=float)
    scaled = np.maximum(e2 / (sigma * sigma), 1e-30)
    return (beta / 2.0) * np.power(scaled, beta / 2.0 - 1.0) / (sigma * sigma)


def huber(e2: np.ndarray, delta: float) -> np.ndarray:
    d2 = delta * delta
    e2 = np.asarray(e2, dtype=float)
    out = np.empty_like(e2)
    inl = e2 <= d2
    out[inl] = e2[inl]
    out[~inl] = 2.0 * delta * np.sqrt(e2[~inl]) - d2
    return out


def huber_deriv(e2: np.ndarray, delta: float) -> np.ndarray:
    d2 = delta * delta
    e2 = np.asarray(e2, dtype=float)
    out = np.ones_like(e2)
    out[e2 > d2] = delta / np.sqrt(e2[e2 > d2])
    return out


def cauchy(e2: np.ndarray, delta: float) -> np.ndarray:
    d2 = delta * delta
    return d2 * np.log1p(np.asarray(e2, dtype=float) / d2)


def cauchy_deriv(e2: np.ndarray, delta: float) -> np.ndarray:
    d2 = delta * delta
    return 1.0 / (1.0 + np.asarray(e2, dtype=float) / d2)


def geman_mcclure(e2: np.ndarray, delta: float) -> np.ndarray:
    e2 = np.asarray(e2, dtype=float)
    return delta * e2 / (delta + e2)


def geman_mcclure_deriv(e2: np.ndarray, delta: float) -> np.ndarray:
    e2 = np.asarray(e2, dtype=float)
    aux = 1.0 / (delta + e2)
    return delta * delta * aux * aux


def welsch(e2: np.ndarray, delta: float) -> np.ndarray:
    d2 = delta * delta
    e2 = np.asarray(e2, dtype=float)
    return d2 * (1.0 - np.exp(-e2 / d2))


def welsch_deriv(e2: np.ndarray, delta: float) -> np.ndarray:
    d2 = delta * delta
    e2 = np.asarray(e2, dtype=float)
    return np.exp(-e2 / d2)


def tukey(e2: np.ndarray, delta: float) -> np.ndarray:
    d2 = delta * delta
    e2 = np.asarray(e2, dtype=float)
    out = np.full_like(e2, d2 / 3.0)
    inl = e2 <= d2
    aux = e2[inl] / d2
    out[inl] = d2 * (1.0 - np.power(1.0 - aux, 3.0)) / 3.0
    return out


def tukey_deriv(e2: np.ndarray, delta: float) -> np.ndarray:
    d2 = delta * delta
    e2 = np.asarray(e2, dtype=float)
    out = np.zeros_like(e2)
    inl = e2 <= d2
    aux = e2[inl] / d2
    out[inl] = np.power(1.0 - aux, 2.0)
    return out


def saturated(e2: np.ndarray, delta: float) -> np.ndarray:
    """g2o RobustKernelSaturated — hard cap (set-like rejection beyond δ)."""
    d2 = delta * delta
    e2 = np.asarray(e2, dtype=float)
    return np.minimum(e2, d2)


def saturated_deriv(e2: np.ndarray, delta: float) -> np.ndarray:
    d2 = delta * delta
    e2 = np.asarray(e2, dtype=float)
    return (e2 <= d2).astype(float)


def dcs(e2: np.ndarray, phi: float) -> np.ndarray:
    """g2o RobustKernelDCS (Agarwal et al., ICRA 2013). δ stores φ."""
    e2 = np.asarray(e2, dtype=float)
    scale = (2.0 * phi) / (phi + e2)
    out = np.empty_like(e2)
    gauss = scale >= 1.0
    out[gauss] = e2[gauss]
    out[~gauss] = scale[~gauss] * e2[~gauss] * scale[~gauss]
    return out


def set_indicator_cost(r: np.ndarray, sigma: float, outside: float = 1.0) -> np.ndarray:
    """
    Idealised set-membership penalty: 0 inside |r|≤σ, constant outside.
    Not a g2o kernel; useful as the limit GGD approaches at large β.
    """
    r = np.asarray(r, dtype=float)
    return np.where(np.abs(r) <= sigma, 0.0, outside)


def rho_2d_isotropic(ex: np.ndarray, ey: np.ndarray, rho_fn, **kwargs) -> np.ndarray:
    """Evaluate ρ(ex² + ey²) on a grid (unit Ω)."""
    e2 = ex * ex + ey * ey
    return rho_fn(e2, **kwargs)
