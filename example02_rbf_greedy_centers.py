"""
Example 2 — Greedy selection of RBF centers
===========================================

Seminar: Introduction to approximation with radial basis functions (RBF)

Same setting as Example 1 (scattered data in R^2, multiquadric RBF with a
constant shape parameter, quadratic augmentation polynomial), but now the
centers are NOT chosen at random: they are selected one at a time with a
greedy algorithm.

Model with a set S of k centers:

        s(x) = sum_{j in S} a_j phi(||x - c_j||)  +  sum_{k=1}^{6} b_k p_k(x)

Greedy algorithm
----------------
    Candidates : all n data points.
    Step 0     : S = {}  (polynomial-only model), compute its SSE.
    Step k     : for every candidate c not yet in S
                     - build A = [ Phi(X, S U {c}) | P(X) ]
                     - solve the least-squares problem  min ||A w - y||^2
                     - compute SSE(c) = sum_i (s(x_i) - y_i)^2
                 choose c* = argmin SSE(c)
    Stop       : when the relative improvement
                     (SSE_{k-1} - SSE_k) / SSE_{k-1}  <  TOL
                 (the center that gives too small an improvement is rejected).
                 M_MAX is only a safety cap.

Two design decisions (good points to discuss in class):

 1) During the selection we do NOT impose the orthogonality constraint
    P_c^T a = 0. With k < 6 centers that constraint (6 equations, k unknowns)
    would force a = 0, so no center could improve the model and the greedy
    algorithm could not even start. Since the problem is overdetermined
    (k + 6 < n), plain least squares already has a unique solution.
    Once the centers are chosen, we do a FINAL REFIT with the constraint using
    SLSQP, exactly as in Example 1.

 2) Each trial fit is solved with scipy.linalg.lstsq (direct SVD-based least
    squares). The greedy loop needs thousands of fits, and an iterative
    optimizer such as SLSQP would be far too slow for that.

Run:       python example02_rbf_greedy_centers.py
Requires:  numpy, scipy, matplotlib
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# 0. Example parameters (change them and see what happens!)
# ---------------------------------------------------------------------------
SEED = 0          # random seed (same data as Example 1)
N_DATA = 200      # n: number of scattered data points
EPS = 3.0         # eps: multiquadric shape parameter (constant)
NOISE = 0.02      # standard deviation of the noise in the data
TOL = 0.01        # stop when adding a center improves SSE by less than 1 %
M_MAX = N_DATA // 2   # safety cap on the number of centers (not the stopping rule)

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. True function and scattered data (identical to Example 1)
# ---------------------------------------------------------------------------
def franke(x, y):
    """Classical test function for scattered data approximation."""
    t1 = 0.75 * np.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4)
    t2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
    t3 = 0.50 * np.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4)
    t4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return t1 + t2 + t3 + t4


X = rng.random((N_DATA, 2))                                         # x_i  (n x 2)
y = franke(X[:, 0], X[:, 1]) + NOISE * rng.standard_normal(N_DATA)  # y_i

# Random ordering of the data points, used only as a baseline for comparison
# (its first 30 entries are exactly the random centers of Example 1).
random_order = rng.choice(N_DATA, size=N_DATA, replace=False)


# ---------------------------------------------------------------------------
# 2. Model ingredients (same as Example 1)
# ---------------------------------------------------------------------------
def multiquadric(r, eps):
    """phi(r) = sqrt(1 + (eps r)^2)"""
    return np.sqrt(1.0 + (eps * r) ** 2)


def distances(A, B):
    """Euclidean distance matrix r_ij = ||A_i - B_j||."""
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def quadratic_poly(P):
    """Quadratic basis in 2D: [1, x, y, x^2, x y, y^2]  ->  matrix (npts x 6)."""
    x, yy = P[:, 0], P[:, 1]
    return np.column_stack([np.ones_like(x), x, yy, x ** 2, x * yy, yy ** 2])


def model_matrix(P, C, eps):
    """A = [ Phi(P, C) | poly(P) ]  so that  s(P) = A @ w."""
    return np.hstack([multiquadric(distances(P, C), eps), quadratic_poly(P)])


# Since every candidate center is a data point, we compute ONCE the full
# n x n matrix Phi_all[i, j] = phi(||x_i - x_j||). Column j is the RBF
# "centered at data point j". Each trial fit just picks columns from it.
Phi_all = multiquadric(distances(X, X), EPS)       # (n x n)
P_data = quadratic_poly(X)                         # (n x 6)

# Fine grid to measure the error against the true function
ng = 80
gx, gy = np.meshgrid(np.linspace(0, 1, ng), np.linspace(0, 1, ng))
G_pts = np.column_stack([gx.ravel(), gy.ravel()])
f_grid = franke(gx, gy).ravel()


def ls_fit(idx):
    """
    Unconstrained least-squares fit using the centers X[idx].
    Returns the coefficients w and the total squared error SSE.
    """
    A = np.hstack([Phi_all[:, idx], P_data])       # (n x (k+6))
    w, *_ = lstsq(A, y)
    res = A @ w - y
    return w, res @ res


def grid_rmse(idx, w):
    """RMSE of the model against the true function on the fine grid."""
    s_grid = model_matrix(G_pts, X[idx], EPS) @ w
    return np.sqrt(np.mean((s_grid - f_grid) ** 2))


# ---------------------------------------------------------------------------
# 3. Greedy selection of centers
# ---------------------------------------------------------------------------
print("=" * 64)
print("Greedy center selection — multiquadric RBF + quadratic polynomial")
print("=" * 64)
print(f"n = {N_DATA},  eps = {EPS},  noise = {NOISE},  TOL = {TOL},  M_MAX = {M_MAX}")

selected = []                                      # indices of chosen centers
w0, sse0 = ls_fit(selected)                        # step 0: polynomial only
sse_hist = [sse0]
rmse_hist = [grid_rmse(selected, w0)]

print(f"\n  k   chosen point     SSE          rel. improvement")
print(f"  0   (polynomial)   {sse0:.4e}")

t0 = time.perf_counter()
while len(selected) < M_MAX:
    best_j, best_sse, best_w = None, np.inf, None

    # Try every candidate that is not yet a center
    for j in range(N_DATA):
        if j in selected:
            continue
        w_trial, sse_trial = ls_fit(selected + [j])
        if sse_trial < best_sse:
            best_j, best_sse, best_w = j, sse_trial, w_trial

    # Stopping rule: relative improvement of the SSE
    rel_impr = (sse_hist[-1] - best_sse) / sse_hist[-1]
    if rel_impr < TOL:
        print(f"  --  best next center improves SSE by only {100*rel_impr:.2f} %"
              f" < {100*TOL:.0f} %  -> stop")
        break

    selected.append(best_j)
    sse_hist.append(best_sse)
    rmse_hist.append(grid_rmse(selected, best_w))
    print(f"{len(selected):3d}   x[{best_j:3d}]        {best_sse:.4e}   {100*rel_impr:6.2f} %")
else:
    print(f"  WARNING: safety cap M_MAX = {M_MAX} reached before meeting TOL")

elapsed = time.perf_counter() - t0
m = len(selected)
print(f"\nGreedy finished: m = {m} centers, "
      f"{m * N_DATA - m * (m - 1) // 2} trial fits in {elapsed:.2f} s")


# ---------------------------------------------------------------------------
# 4. Baseline: random centers (nested: first k entries of random_order)
# ---------------------------------------------------------------------------
sse_rand, rmse_rand = [sse0], [rmse_hist[0]]
for k in range(1, m + 1):
    idx = list(random_order[:k])
    w_k, sse_k = ls_fit(idx)
    sse_rand.append(sse_k)
    rmse_rand.append(grid_rmse(idx, w_k))


# ---------------------------------------------------------------------------
# 5. Final refit WITH the orthogonality constraint (SLSQP, as in Example 1)
# ---------------------------------------------------------------------------
def constrained_fit(C):
    """min 1/2 ||A w - y||^2  s.t.  P_c^T a = 0, solved with SLSQP."""
    A = model_matrix(X, C, EPS)
    Pc = quadratic_poly(C)
    mc = C.shape[0]
    G = np.hstack([Pc.T, np.zeros((6, 6))])        # constraint Jacobian

    sol = minimize(lambda w: 0.5 * np.sum((A @ w - y) ** 2),
                   np.zeros(mc + 6),
                   jac=lambda w: A.T @ (A @ w - y),
                   constraints=[{"type": "eq", "fun": lambda w: G @ w,
                                 "jac": lambda w: G}],
                   method="SLSQP", options={"maxiter": 2000, "ftol": 1e-12})
    w = sol.x
    sse = np.sum((A @ w - y) ** 2)
    return w, sse, sol, np.linalg.norm(G @ w)


C_greedy = X[selected]
w_g, sse_g, sol_g, cons_g = constrained_fit(C_greedy)
C_random = X[random_order[:m]]
w_r, sse_r, sol_r, cons_r = constrained_fit(C_random)

s_grid_g = model_matrix(G_pts, C_greedy, EPS) @ w_g
s_grid_r = model_matrix(G_pts, C_random, EPS) @ w_r
rmse_g = np.sqrt(np.mean((s_grid_g - f_grid) ** 2))
rmse_r = np.sqrt(np.mean((s_grid_r - f_grid) ** 2))

print(f"\n--- final refit with constraint P_c^T a = 0 (SLSQP), m = {m} ---")
print(f"SLSQP greedy centers : success = {sol_g.success}, "
      f"iterations = {sol_g.nit}, ||P_c^T a|| = {cons_g:.1e}")
print(f"SLSQP random centers : success = {sol_r.success}, "
      f"iterations = {sol_r.nit}, ||P_c^T a|| = {cons_r:.1e}")
print(f"\n{'':32s}{'total sq. error':>17s}{'grid RMSE':>12s}")
print(f"{'greedy, without constraint':32s}{sse_hist[-1]:17.4e}{rmse_hist[-1]:12.4e}")
print(f"{'greedy, with constraint':32s}{sse_g:17.4e}{rmse_g:12.4e}")
print(f"{'random, with constraint':32s}{sse_r:17.4e}{rmse_r:12.4e}")
print(f"\nNoise floor  n * sigma^2 = {N_DATA * NOISE**2:.4e}"
      "  (expected SSE of the exact function f on noisy data)")


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------
ks = np.arange(m + 1)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# (a) Data and greedy centers, numbered in order of selection
ax = axs[0, 0]
sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=16, label="data $x_i$")
ax.scatter(C_greedy[:, 0], C_greedy[:, 1], facecolors="none", edgecolors="red",
           s=110, linewidths=1.5, label="greedy centers")
for order, (cx, cy) in enumerate(C_greedy, start=1):
    ax.annotate(str(order), (cx, cy), xytext=(4, 4), textcoords="offset points",
                fontsize=8, color="red", fontweight="bold")
ax.set_title(f"Greedy centers (m={m}), numbered by selection order")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
fig.colorbar(sc, ax=ax, shrink=0.8, label="$y_i$")

# (b) Convergence of the SSE on the data
ax = axs[0, 1]
ax.semilogy(ks, sse_hist, "o-", ms=4, label="greedy")
ax.semilogy(ks, sse_rand, "s--", ms=3, label="random")
ax.axhline(N_DATA * NOISE**2, color="gray", ls=":", label=r"noise floor $n\sigma^2$")
ax.set_title("Total squared error on the data")
ax.set_xlabel("number of RBF centers $k$"); ax.set_ylabel(r"$\sum_i (s(x_i)-y_i)^2$")
ax.grid(True, which="both", alpha=0.3); ax.legend()

# (c) Error against the true function on the grid
ax = axs[1, 0]
ax.semilogy(ks, rmse_hist, "o-", ms=4, label="greedy")
ax.semilogy(ks, rmse_rand, "s--", ms=3, label="random")
ax.axhline(NOISE, color="gray", ls=":", label=r"noise level $\sigma$")
ax.set_title("RMSE vs true function $f$ on an 80x80 grid")
ax.set_xlabel("number of RBF centers $k$"); ax.set_ylabel("RMSE")
ax.grid(True, which="both", alpha=0.3); ax.legend()

# (d) Absolute error of the final (constrained) greedy model
ax = axs[1, 1]
cf = ax.contourf(gx, gy, np.abs(s_grid_g - f_grid).reshape(ng, ng),
                 levels=20, cmap="magma")
ax.scatter(C_greedy[:, 0], C_greedy[:, 1], c="cyan", s=12, label="centers")
ax.set_title("Absolute error $|s(x)-f(x)|$, greedy + constraint")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=8)
fig.colorbar(cf, ax=ax, shrink=0.8)

fig.tight_layout()
fig.savefig("example02_results.png", dpi=150)
print("\nFigure saved to: example02_results.png")
plt.show()
