"""
Example 4 — A second-order stochastic method: stochastic Gauss-Newton
=====================================================================

Seminar: Introduction to approximation with radial basis functions (RBF)

Same problem as Example 3 (data, centers, eps = 3, quadratic polynomial,
orthogonality conditions appended as extra rows with weight lambda):

        L(w) = 1/2 ||A_aug w - y_aug||^2

Why second order?
-----------------
For a quadratic loss, NO first-order method (one that only combines
gradients: SGD, momentum, Adam, ...) can converge faster than about
sqrt(kappa) * log(1/tol) iterations in the worst case (Nesterov's lower bound),
where kappa = cond(A^T A). Here kappa ~ 1e13. Only curvature information
breaks that barrier.

Stochastic Gauss-Newton (subsampled Newton)
-------------------------------------------
Our model is linear in w, so Gauss-Newton coincides with Newton:
the Hessian is H = A^T A. With the full data one step would be exact.
The stochastic version never builds the full Hessian:

    g_k = (n/|B|)   A_B^T (A_B w - y_B)   + A_con^T A_con w    (gradient, batch B)
    H_k = (n/|B_H|) A_BH^T A_BH           + A_con^T A_con      (Hessian, batch B_H)
    solve  (H_k + mu I) d = -g_k                                (p x p system)
    w  <-  w + alpha_k d

  - B_H must be larger than the number of unknowns p (otherwise H_k is
    singular) and large enough to capture the small-curvature directions.
  - mu (Levenberg-Marquardt damping) protects against an inaccurate H_k
    and against round-off (forming A^T A squares the condition number).
  - alpha_k decreasing (Robbins-Monro) damps the noise of the stochastic
    gradient, which H_k^{-1} amplifies along low-curvature directions.

The example compares:
  - Adam with standardized columns (best first-order method of Example 3)
  - "naive" stochastic Gauss-Newton: small Hessian batch, full step -> diverges
  - "robust" stochastic Gauss-Newton: large enough Hessian batch, damping,
    decreasing step (5 random seeds)
  - lstsq on the augmented system (exact minimum, reference)

Cost is measured as the number of rows of A processed: that is the work
measure that matters when n is huge.
NOTE: the full loss is evaluated at every iteration ONLY to draw the plots;
it is not part of any algorithm.

Run:       python example04_rbf_stochastic_gauss_newton.py
Requires:  numpy, scipy, matplotlib
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq, solve

# ---------------------------------------------------------------------------
# 0. Parameters
# ---------------------------------------------------------------------------
SEED = 0          # same data and centers as Examples 1 and 3
N_DATA = 200
M_CENTERS = 30
EPS = 3.0
NOISE = 0.02
LAMBDA = 1.0      # weight of the orthogonality rows

# Stochastic Gauss-Newton
GN_ITERS = 1000   # iterations
GN_BATCH = 20     # gradient batch |B|
GN_SEEDS = [1, 2, 3, 4, 5]
ROBUST = dict(batch_h=150, mu=1e-4, decay=10.0)   # alpha_k = 1/(1 + k/decay)
NAIVE = dict(batch_h=60, mu=1e-4, decay=None)     # alpha_k = 1 (full step)

# Adam (reference first-order method, as in Example 3)
ADAM_EPOCHS = 10000
ADAM_BATCH = 20
ADAM_LR = 1e-2

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. Data, centers and model (identical to Examples 1 and 3)
# ---------------------------------------------------------------------------
def franke(x, y):
    t1 = 0.75 * np.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4)
    t2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
    t3 = 0.50 * np.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4)
    t4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return t1 + t2 + t3 + t4


X = rng.random((N_DATA, 2))
y = franke(X[:, 0], X[:, 1]) + NOISE * rng.standard_normal(N_DATA)
C = X[rng.choice(N_DATA, size=M_CENTERS, replace=False)]


def multiquadric(r, eps):
    return np.sqrt(1.0 + (eps * r) ** 2)


def distances(A, B):
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def quadratic_poly(P):
    x, yy = P[:, 0], P[:, 1]
    return np.column_stack([np.ones_like(x), x, yy, x ** 2, x * yy, yy ** 2])


def model_matrix(P, C, eps):
    return np.hstack([multiquadric(distances(P, C), eps), quadratic_poly(P)])


A = model_matrix(X, C, EPS)                                          # n x p
A_con = np.hstack([LAMBDA * quadratic_poly(C).T, np.zeros((6, 6))])  # 6 x p
A_aug = np.vstack([A, A_con])
y_aug = np.concatenate([y, np.zeros(6)])
p = A_aug.shape[1]
H_con = A_con.T @ A_con          # constraint part of the Hessian (always included)


def loss(w):
    r = A_aug @ w - y_aug
    return 0.5 * r @ r


ng = 80
gx, gy = np.meshgrid(np.linspace(0, 1, ng), np.linspace(0, 1, ng))
G_pts = np.column_stack([gx.ravel(), gy.ravel()])
A_grid = model_matrix(G_pts, C, EPS)
f_grid = franke(G_pts[:, 0], G_pts[:, 1])


def grid_rmse(w):
    return np.sqrt(np.mean((A_grid @ w - f_grid) ** 2))


print("=" * 72)
print("Example 4 — stochastic Gauss-Newton vs. Adam")
print("=" * 72)
print(f"unknowns p = {p},  data n = {N_DATA},  cond(A_aug) = {np.linalg.cond(A_aug):.2e}")

# ---------------------------------------------------------------------------
# 2. Reference: exact minimum
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
w_ref, *_ = lstsq(A_aug, y_aug)
t_ref = time.perf_counter() - t0
L_ref = loss(w_ref)


# ---------------------------------------------------------------------------
# 3. Stochastic Gauss-Newton
# ---------------------------------------------------------------------------
def stochastic_gauss_newton(batch_h, mu, decay, seed):
    """
    Returns final w and history rows (rows_processed, L/L_ref, grid RMSE).
    Rows processed per iteration: |B| + |B_H| (+ the 6 constraint rows).
    """
    r = np.random.default_rng(seed)
    w = np.zeros(p)
    rows = 0
    hist = [(0, loss(w) / L_ref, grid_rmse(w))]
    for k in range(GN_ITERS):
        B = r.choice(N_DATA, GN_BATCH, replace=False)      # gradient batch
        BH = r.choice(N_DATA, batch_h, replace=False)      # Hessian batch

        g = (N_DATA / GN_BATCH) * A[B].T @ (A[B] @ w - y[B]) + H_con @ w
        H = (N_DATA / batch_h) * A[BH].T @ A[BH] + H_con

        d = solve(H + mu * np.eye(p), -g, assume_a="pos")  # Newton direction
        alpha = 1.0 if decay is None else 1.0 / (1.0 + k / decay)
        w = w + alpha * d

        rows += GN_BATCH + batch_h + 6
        L_rel = loss(w) / L_ref
        hist.append((rows, L_rel, grid_rmse(w)))
        if not np.isfinite(L_rel) or L_rel > 1e12:       # diverged: stop
            break
    return w, np.array(hist)


print("\nStochastic Gauss-Newton")
gn_naive = {}
gn_robust = {}
for seed in GN_SEEDS:
    t0 = time.perf_counter()
    gn_naive[seed] = stochastic_gauss_newton(**NAIVE, seed=seed) + (time.perf_counter() - t0,)
    t0 = time.perf_counter()
    gn_robust[seed] = stochastic_gauss_newton(**ROBUST, seed=seed) + (time.perf_counter() - t0,)
print(f"  naive : |B_H| = {NAIVE['batch_h']}, mu = {NAIVE['mu']:.0e}, full step alpha = 1")
print(f"  robust: |B_H| = {ROBUST['batch_h']}, mu = {ROBUST['mu']:.0e}, "
      f"alpha_k = 1/(1 + k/{ROBUST['decay']:.0f})")


# ---------------------------------------------------------------------------
# 4. Adam with standardized columns (best first-order method of Example 3)
# ---------------------------------------------------------------------------
mu_c, sd_c = A.mean(axis=0), A.std(axis=0)
mu_c[M_CENTERS], sd_c[M_CENTERS] = 0.0, 1.0
T_std = np.diag(1.0 / sd_c)
T_std[M_CENTERS, :] -= mu_c / sd_c
T_std[M_CENTERS, M_CENTERS] = 1.0


def adam(seed=1):
    r = np.random.default_rng(seed)
    As, Acs = A @ T_std, A_con @ T_std
    v, m, s = np.zeros(p), np.zeros(p), np.zeros(p)
    t, rows = 0, 0
    hist = [(0, loss(T_std @ v) / L_ref, grid_rmse(T_std @ v))]
    for epoch in range(1, ADAM_EPOCHS + 1):
        perm = r.permutation(N_DATA)
        for k in range(0, N_DATA, ADAM_BATCH):
            B = perm[k:k + ADAM_BATCH]
            g = (N_DATA / len(B)) * As[B].T @ (As[B] @ v - y[B]) + Acs.T @ (Acs @ v)
            t += 1
            m = 0.9 * m + 0.1 * g
            s = 0.999 * s + 0.001 * g * g
            v -= ADAM_LR * (m / (1 - 0.9 ** t)) / (np.sqrt(s / (1 - 0.999 ** t)) + 1e-8)
            rows += len(B) + 6
        if epoch % 10 == 0:
            w = T_std @ v
            hist.append((rows, loss(w) / L_ref, grid_rmse(w)))
    return T_std @ v, np.array(hist), t


t0 = time.perf_counter()
w_adam, hist_adam, iters_adam = adam()
t_adam = time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 5. Comparison table
# ---------------------------------------------------------------------------
print("\n" + "-" * 96)
print(f"{'method':40s}{'L/L_min':>11s}{'RMSE vs f':>11s}{'iterations':>12s}"
      f"{'rows processed':>16s}{'time':>8s}")
print("-" * 96)
print(f"{'lstsq (exact, reference)':40s}{1.0:11.3f}{grid_rmse(w_ref):11.4f}"
      f"{'-':>12s}{N_DATA + 6:16d}{t_ref:7.3f}s")
print(f"{'Adam, standardized (Example 3)':40s}{loss(w_adam) / L_ref:11.3f}"
      f"{grid_rmse(w_adam):11.4f}{iters_adam:12d}{int(hist_adam[-1, 0]):16d}{t_adam:7.2f}s")

for label, runs in [("Gauss-Newton, naive", gn_naive), ("Gauss-Newton, robust", gn_robust)]:
    Ls = [h[-1, 1] for (_, h, _) in runs.values()]
    rms = [h[-1, 2] for (_, h, _) in runs.values()]
    its = [len(h) - 1 for (_, h, _) in runs.values()]
    rws = [int(h[-1, 0]) for (_, h, _) in runs.values()]
    tms = [tt for (_, _, tt) in runs.values()]
    print(f"{label + ' (median of 5 seeds)':40s}{np.median(Ls):11.3g}{np.median(rms):11.4g}"
          f"{int(np.median(its)):12d}{int(np.median(rws)):16d}{np.median(tms):7.2f}s")
print("-" * 96)
print("Naive runs stop when they diverge (L/L_min > 1e12).")

# Iterations needed by robust Gauss-Newton to get within 10 % of the minimum
need = []
for (_, h, _) in gn_robust.values():
    idx = np.where(h[:, 1] < 1.10)[0]
    need.append((idx[0], int(h[idx[0], 0])) if len(idx) else (None, None))
print("\nRobust Gauss-Newton, first iteration with L < 1.10 L_min (per seed):")
for seed, (it, rw) in zip(GN_SEEDS, need):
    print(f"  seed {seed}: iteration {it},  rows processed {rw}")


# ---------------------------------------------------------------------------
# 6. Effect of the Hessian batch size (robust schedule, 5 seeds)
# ---------------------------------------------------------------------------
print("\nEffect of the Hessian batch size |B_H| (robust damping and step):")
bh_values = [40, 60, 80, 100, 125, 150, 200]
bh_median, bh_max = [], []
for bh in bh_values:
    finals = []
    for seed in GN_SEEDS:
        _, h = stochastic_gauss_newton(bh, ROBUST["mu"], ROBUST["decay"], seed)
        finals.append(h[-1, 1])
    bh_median.append(np.median(finals))
    bh_max.append(np.max(finals))
    print(f"  |B_H| = {bh:3d}:  L/L_min  median = {np.median(finals):10.3g}   "
          f"worst = {np.max(finals):10.3g}")


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
YMAX = 1e3   # clip diverging curves in the plots

# (a) loss vs rows processed
ax = axs[0, 0]
ax.loglog(hist_adam[1:, 0], hist_adam[1:, 1], color="tab:green", label="Adam (standardized)")
for i, (_, h, _) in enumerate(gn_naive.values()):
    ax.loglog(h[1:, 0], np.minimum(h[1:, 1], YMAX), color="tab:red", alpha=0.9, lw=2,
              label="Gauss-Newton, naive" if i == 0 else None)
for i, (_, h, _) in enumerate(gn_robust.values()):
    ax.loglog(h[1:, 0], h[1:, 1], color="tab:blue", alpha=0.7, lw=1,
              label="Gauss-Newton, robust (5 seeds)" if i == 0 else None)
ax.axhline(1.0, color="k", ls="--", label="exact minimum")
ax.set_ylim(0.8, YMAX)
ax.set_title("Loss relative to the minimum, $L/L_{min}$")
ax.set_xlabel("rows of $A$ processed (work)"); ax.set_ylabel("$L/L_{min}$")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

# (b) error against the true function vs rows processed
ax = axs[0, 1]
ax.loglog(hist_adam[1:, 0], hist_adam[1:, 2], color="tab:green", label="Adam (standardized)")
for i, (_, h, _) in enumerate(gn_robust.values()):
    ax.loglog(h[1:, 0], h[1:, 2], color="tab:blue", alpha=0.7, lw=1,
              label="Gauss-Newton, robust (5 seeds)" if i == 0 else None)
ax.axhline(grid_rmse(w_ref), color="k", ls="--", label="exact minimum")
ax.axhline(NOISE, color="gray", ls=":", label=r"noise level $\sigma$")
ax.set_ylim(1e-2, 1)
ax.set_title("RMSE vs true $f$ on an 80x80 grid")
ax.set_xlabel("rows of $A$ processed (work)"); ax.set_ylabel("RMSE")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

# (c) effect of the Hessian batch size
ax = axs[1, 0]
ax.semilogy(bh_values, np.minimum(bh_median, 1e12), "o-", label="median of 5 seeds")
ax.semilogy(bh_values, np.minimum(bh_max, 1e12), "s--", label="worst of 5 seeds")
ax.axhline(1.0, color="k", ls="--", label="exact minimum")
ax.axvline(p, color="gray", ls=":", label=f"number of unknowns p = {p}")
ax.text(bh_values[0] + 3, 2e11, "diverged\n(clipped at $10^{12}$)", fontsize=8, va="top")
ax.set_title(f"Final $L/L_{{min}}$ after {GN_ITERS} iterations vs Hessian batch size")
ax.set_xlabel("Hessian batch size $|B_H|$"); ax.set_ylabel("$L/L_{min}$")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

# (d) cut along x2 = 0.25
w_gn = gn_robust[GN_SEEDS[0]][0]
xs = np.linspace(0, 1, 300)
A_line = model_matrix(np.column_stack([xs, np.full_like(xs, 0.25)]), C, EPS)
ax = axs[1, 1]
ax.plot(xs, franke(xs, 0.25), "k", lw=2, label="true $f$")
ax.plot(xs, A_line @ w_ref, "--", color="tab:red", lw=2, label="exact minimum (lstsq)")
ax.plot(xs, A_line @ w_gn, color="tab:blue", lw=1.2, label="Gauss-Newton, robust")
ax.plot(xs, A_line @ w_adam, color="tab:green", lw=1.2, label="Adam (standardized)")
near = np.abs(X[:, 1] - 0.25) < 0.05
ax.scatter(X[near, 0], y[near], s=12, c="gray", label=r"data with $|x_2-0.25|<0.05$")
ax.set_title("Cut along $x_2 = 0.25$")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$s(x_1, 0.25)$")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("example04_results.png", dpi=150)
print("\nFigure saved to: example04_results.png")
plt.show()
