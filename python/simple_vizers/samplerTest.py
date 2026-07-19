import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gennorm

# ------------------------------------------------------------------
# 1.  Define the non-Gaussian 2-D target distribution
# ------------------------------------------------------------------
w = np.array([1/3, 1/3, 1/3])        # equal mixture weights

# --- two Gaussians -------------------------------------------------
mu1, Sigma1 = np.array([-2.0, 0.0]), np.array([[1.0,  0.3],
                                               [0.3,  0.8]])
mu2, Sigma2 = np.array([ 2.5, 0.5]), np.array([[1.1, -0.4],
                                               [-0.4, 1.3]])

# --- one 2-D independent generalised Gaussian ----------------------
mu3   = np.array([4.0, 3.0])
beta  = 8.0                       # shape parameter  (β<2 ⇒ heavy tails)
scale = np.array([3.0, 2.7])      # per-coordinate scale (α in scipy docs)

def sample_target(n: int) -> np.ndarray:
    """Draw n points from the mixture."""
    comp = np.random.choice(3, size=n, p=w)
    out  = np.empty((n, 2))

    idx = comp == 0
    if idx.any():
        out[idx] = np.random.multivariate_normal(mu1, Sigma1, idx.sum())

    idx = comp == 1
    if idx.any():
        out[idx] = np.random.multivariate_normal(mu2, Sigma2, idx.sum())

    idx = comp == 2
    if idx.any():
        out[idx, 0] = gennorm.rvs(beta, loc=mu3[0], scale=scale[0], size=idx.sum())
        out[idx, 1] = gennorm.rvs(beta, loc=mu3[1], scale=scale[1], size=idx.sum())

    return out


# ------------------------------------------------------------------
# 2.  Monte-Carlo sample
# ------------------------------------------------------------------
N = 300
X = sample_target(N)                # shape (N, 2)

# ------------------------------------------------------------------
# 3.  “Smallest” covariance ellipse covering 99 % of draws
# ------------------------------------------------------------------
mu_hat     = X.mean(0)
Sigma_hat  = np.cov(X, rowvar=False)

# Mahalanobis distances
Sinv = np.linalg.inv(Sigma_hat)
d2   = np.einsum('ij,jk,ik->i', X - mu_hat, Sinv, X - mu_hat)

r2 = np.quantile(d2, 0.99)          # squared radius for 99 % coverage

# Ellipse parameterisation
eigvals, eigvecs = np.linalg.eigh(Sigma_hat)
order = eigvals.argsort()[::-1]     # largest eigenvalue first
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
axes = np.sqrt(r2 * eigvals)        # semi-axes lengths

theta = np.linspace(0, 2*np.pi, 400)
ellipse = (eigvecs @ (axes[:, None] * [np.cos(theta), np.sin(theta)])) + mu_hat[:, None]


# ------------------------------------------------------------------
# 4.  True density on a grid (for a nice heat map)
# ------------------------------------------------------------------
def pdf_target(xy):
    p1 = multivariate_normal.pdf(xy, mean=mu1, cov=Sigma1)
    p2 = multivariate_normal.pdf(xy, mean=mu2, cov=Sigma2)
    p3 = gennorm.pdf(xy[:, 0],      beta, loc=mu3[0], scale=scale[0]) * \
         gennorm.pdf(xy[:, 1],      beta, loc=mu3[1], scale=scale[1])
    return (p1 + p2 + p3) / 3.0

xmin, xmax, ymin, ymax = -12, 12, -8, 16
grid = 300
xg = np.linspace(xmin, xmax, grid)
yg = np.linspace(ymin, ymax, grid)
Xg, Yg = np.meshgrid(xg, yg)
pdf_grid = pdf_target(np.column_stack([Xg.ravel(), Yg.ravel()])).reshape(grid, grid)


# ------------------------------------------------------------------
# 5.  Plot heat map + 99 % covariance ellipse
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
ax.contourf(Xg, Yg, pdf_grid, levels=60, cmap="viridis")
ax.plot(ellipse[0], ellipse[1], color="red", lw=2, label="99% covariance ellipse")

ax.set_title("Mixture density and 99 % covariance-based bound")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.grid(ls=":")
ax.legend()
plt.show()
