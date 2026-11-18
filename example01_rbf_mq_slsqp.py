"""
Example 1 — RBF approximation in R^2 with multiquadrics (MQ)
============================================================

Seminar: Introduction to approximation with radial basis functions (RBF)

Goal
----
We start from n scattered (unstructured) data points

        (x_i, y_i),   x_i in R^2,   y_i = f(x_i) (+ noise),   i = 1, ..., n

and build a model with m RBF terms (m < n) plus a quadratic augmentation
polynomial:

        s(x) = sum_{j=1}^{m} a_j phi(||x - c_j||)  +  sum_{k=1}^{6} b_k p_k(x)

where
    phi(r) = sqrt(1 + (eps r)^2)      multiquadric, eps = shape parameter (constant)
    c_j    = centers, chosen at random among the n data points
    p_k    in {1, x, y, x^2, x y, y^2}   (quadratic polynomial in 2D -> 6 terms)

Since m < n this is NOT an interpolation: there are more equations (n) than
unknowns (m + 6). The coefficients are obtained by minimizing the squared error

        J(a, b) = 1/2 sum_i ( s(x_i) - y_i )^2

subject to the orthogonality condition (usual with polynomial augmentation)

        sum_j a_j p_k(c_j) = 0,   k = 1, ..., 6      (P_c^T a = 0)

We use scipy.optimize.minimize (SLSQP method, which supports equality
constraints) with the exact gradient.

Run:       python example01_rbf_mq_slsqp.py
Requires:  numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# 0. Example parameters (change them and see what happens!)
# ---------------------------------------------------------------------------
SEED = 0          # random seed (reproducibility)
N_DATA = 200      # n: number of scattered data points
M_CENTERS = 30    # m: number of RBF centers (m < n)
EPS = 3.0         # eps: multiquadric shape parameter (constant)
NOISE = 0.02      # standard deviation of the noise in the data (0 = no noise)

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. "True" function (unknown in practice): Franke's function
# ---------------------------------------------------------------------------
def franke(x, y):
    """Classical test function for scattered data approximation."""
    t1 = 0.75 * np.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4)
    t2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
    t3 = 0.50 * np.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4)
    t4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return t1 + t2 + t3 + t4


# ---------------------------------------------------------------------------
# 2. Scattered data: n random points in [0,1]^2
# ---------------------------------------------------------------------------
X = rng.random((N_DATA, 2))                                         # x_i  (n x 2)
y = franke(X[:, 0], X[:, 1]) + NOISE * rng.standard_normal(N_DATA)  # y_i

# ---------------------------------------------------------------------------
# 3. Centers: m points chosen at random among the data (no repetition)
# ---------------------------------------------------------------------------
center_idx = rng.choice(N_DATA, size=M_CENTERS, replace=False)
C = X[center_idx]                                  # c_j  (m x 2)


# ---------------------------------------------------------------------------
# 4. Model ingredients
# ---------------------------------------------------------------------------
def multiquadric(r, eps):
    """phi(r) = sqrt(1 + (eps r)^2)"""
    return np.sqrt(1.0 + (eps * r) ** 2)


def distances(A, B):
    """Euclidean distance matrix r_ij = ||A_i - B_j||."""
    diff = A[:, None, :] - B[None, :, :]           # (na, nb, 2)
    return np.sqrt(np.sum(diff ** 2, axis=2))      # (na, nb)


def quadratic_poly(P):
    """Quadratic basis in 2D: [1, x, y, x^2, x y, y^2]  ->  matrix (npts x 6)."""
    x, yy = P[:, 0], P[:, 1]
    return np.column_stack([np.ones_like(x), x, yy, x ** 2, x * yy, yy ** 2])


def model_matrix(P, C, eps):
    """
    Matrix A such that s(P) = A @ w, with w = [a_1..a_m, b_1..b_6].
    A = [ Phi | Pp ],  Phi_ij = phi(||p_i - c_j||),  Pp = polynomial evaluated at P.
    """
    Phi = multiquadric(distances(P, C), eps)       # (npts x m)
    Pp = quadratic_poly(P)                         # (npts x 6)
    return np.hstack([Phi, Pp])


# Problem matrix at the n data points: (n x (m+6))
A = model_matrix(X, C, EPS)
n_par = A.shape[1]                                 # m + 6 unknowns
Pc = quadratic_poly(C)                             # (m x 6), for the constraint

print("=" * 60)
print("Multiquadric RBF approximation + quadratic polynomial")
print("=" * 60)
print(f"n (data)              = {N_DATA}")
print(f"m (RBF centers)       = {M_CENTERS}")
print(f"polynomial terms      = 6")
print(f"unknowns (m+6)        = {n_par}   <   n = {N_DATA}  -> approximation")
print(f"eps (shape)           = {EPS}")
print(f"cond(A)               = {np.linalg.cond(A):.3e}")


# ---------------------------------------------------------------------------
# 5. Optimization problem
#      min  J(w) = 1/2 ||A w - y||^2
#      s.t. P_c^T a = 0
# ---------------------------------------------------------------------------
def J(w):
    res = A @ w - y
    return 0.5 * res @ res


def grad_J(w):
    return A.T @ (A @ w - y)                       # grad J = A^T (A w - y)


# Equality constraint: P_c^T a = 0  (involves only the coefficients a)
def constraint(w):
    return Pc.T @ w[:M_CENTERS]


constraint_jac = np.hstack([Pc.T, np.zeros((6, 6))])   # constant (6 x (m+6))

constraints = [{"type": "eq", "fun": constraint, "jac": lambda w: constraint_jac}]

w0 = np.zeros(n_par)                               # initial guess
sol = minimize(J, w0, jac=grad_J, constraints=constraints,
               method="SLSQP", options={"maxiter": 2000, "ftol": 1e-12})

w = sol.x
a, b = w[:M_CENTERS], w[M_CENTERS:]

print("\n--- scipy.optimize.minimize (SLSQP) ---")
print(f"success               : {sol.success}  ({sol.message})")
print(f"iterations            : {sol.nit}")
print(f"J(w*)                 : {sol.fun:.6e}")
print(f"||P_c^T a||           : {np.linalg.norm(constraint(w)):.2e}  (should be ~ 0)")


# ---------------------------------------------------------------------------
# 6. Model evaluation
# ---------------------------------------------------------------------------
def s(P):
    """Evaluate the RBF model at points P (npts x 2)."""
    return model_matrix(P, C, EPS) @ w


# Error at the data points
residuals = s(X) - y                               # s(x_i) - y_i
total_sq_error = np.sum(residuals ** 2)            # sum (s(x_i) - y_i)^2  (= 2 J)
rmse_data = np.sqrt(np.mean(residuals ** 2))

# Error on a fine grid (against the true function)
ng = 80
gx, gy = np.meshgrid(np.linspace(0, 1, ng), np.linspace(0, 1, ng))
G_pts = np.column_stack([gx.ravel(), gy.ravel()])
f_grid = franke(gx, gy)
s_grid = s(G_pts).reshape(ng, ng)
err_grid = np.abs(s_grid - f_grid)

print("\n--- errors ---")
print(f"Total squared error sum(s(x_i)-y_i)^2 : {total_sq_error:.6e}")
print(f"RMSE at the n data points             : {rmse_data:.4e}")
print(f"RMSE on {ng}x{ng} grid (vs f)            : {np.sqrt(np.mean(err_grid**2)):.4e}")
print(f"Max error on the grid                 : {err_grid.max():.4e}")


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16.5, 4.8))

# (a) Data and centers
ax1 = fig.add_subplot(1, 3, 1)
sc = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=18, label="data $x_i$")
ax1.scatter(C[:, 0], C[:, 1], facecolors="none", edgecolors="red", s=90,
            linewidths=1.5, label="centers $c_j$")
ax1.set_title(f"Scattered data (n={N_DATA}) and centers (m={M_CENTERS})")
ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$"); ax1.set_aspect("equal")
ax1.legend(loc="upper right", fontsize=8)
fig.colorbar(sc, ax=ax1, shrink=0.8, label="$y_i$")

# (b) Approximated surface with the data
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.plot_surface(gx, gy, s_grid, cmap="viridis", alpha=0.8, linewidth=0)
ax2.scatter(X[:, 0], X[:, 1], y, color="k", s=6)
ax2.set_title("RBF approximation $s(x)$ and data")
ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")

# (c) Absolute error |s - f|
ax3 = fig.add_subplot(1, 3, 3)
cf = ax3.contourf(gx, gy, err_grid, levels=20, cmap="magma")
ax3.scatter(C[:, 0], C[:, 1], c="cyan", s=12, label="centers")
ax3.set_title("Absolute error $|s(x) - f(x)|$")
ax3.set_xlabel("$x_1$"); ax3.set_ylabel("$x_2$"); ax3.set_aspect("equal")
ax3.legend(loc="upper right", fontsize=8)
fig.colorbar(cf, ax=ax3, shrink=0.8)

fig.tight_layout(w_pad=4)
fig.savefig("example01_results.png", dpi=150)
print("\nFigure saved to: example01_results.png")
plt.show()
