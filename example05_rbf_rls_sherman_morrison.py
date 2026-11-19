"""
Example 5 — Recursive least squares (RLS): one data row at a time
=================================================================

Seminar: Introduction to approximation with radial basis functions (RBF)

Same problem as Examples 3 and 4 (data, 30 random centers, eps = 3,
quadratic polynomial, orthogonality rows appended with weight lambda):

        L(w) = 1/2 ||A_aug w - y_aug||^2 ,    A_aug has rows a_i^T

Idea
----
The least-squares solution is  w* = (A^T A)^{-1} A^T y, and both A^T A and
A^T y are SUMS over the rows:

        A^T A = sum_i a_i a_i^T ,      A^T y = sum_i a_i y_i

So we can process the rows ONE AT A TIME and keep

        P_k = (delta I + a_1 a_1^T + ... + a_k a_k^T)^{-1}     (p x p, never grows)
        w_k = least-squares solution using the first k rows

Adding a row a changes the matrix by a rank-1 term a a^T. The
Sherman-Morrison formula updates the inverse WITHOUT inverting anything:

        (M + a a^T)^{-1} = M^{-1} - (M^{-1} a)(a^T M^{-1}) / (1 + a^T M^{-1} a)

which gives the RLS update (cost O(p^2) per row):

        k   = P a / (1 + a^T P a)           (gain vector)
        w  <- w + k (y_i - a^T w)           (correct w with the new residual)
        P  <- P - k (P a)^T                 (Sherman-Morrison)

  - Start: P_0 = I / delta. With fewer than p rows, sum a_i a_i^T is not
    invertible; the small delta makes P_0 well defined.
  - After ONE pass over all n rows, w is the solution of
        (A^T A + delta I) w = A^T y
    i.e. least squares with a tiny ridge regularization delta.
  - The order of the rows does not change the final result (a sum does not
    depend on the order of its terms). It does change the intermediate models.
  - It is a second-order method (it carries the full curvature in P) that
    uses mini-batches of size ONE: an "online learning" method that is exact.

The example shows:
  (a) the loss vs rows processed, rows sorted by x_1 vs random order
  (b) the error against the true f vs rows processed
  (c) the effect of delta: regularization bias vs round-off error
  (d) the model being built row by row (rows sorted by x_1)

Run:       python example05_rbf_rls_sherman_morrison.py
Requires:  numpy, scipy, matplotlib
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

# ---------------------------------------------------------------------------
# 0. Parameters
# ---------------------------------------------------------------------------
SEED = 0          # same data and centers as Examples 1, 3 and 4
N_DATA = 200
M_CENTERS = 30
EPS = 3.0
NOISE = 0.02
LAMBDA = 1.0      # weight of the orthogonality rows
DELTA = 1e-10     # P_0 = I / DELTA
DELTAS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]   # for plot (c)

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. Data, centers and model (identical to Examples 1, 3 and 4)
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
A_aug = np.vstack([A, A_con])                                        # (n+6) x p
y_aug = np.concatenate([y, np.zeros(6)])
n_rows, p = A_aug.shape


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


# Reference: exact minimum
t0 = time.perf_counter()
w_ref, *_ = lstsq(A_aug, y_aug)
t_ref = time.perf_counter() - t0
L_ref = loss(w_ref)

print("=" * 72)
print("Example 5 — recursive least squares (Sherman-Morrison), one row at a time")
print("=" * 72)
print(f"rows n + 6 = {n_rows},  unknowns p = {p},  cond(A_aug) = {np.linalg.cond(A_aug):.2e}")


# ---------------------------------------------------------------------------
# 2. Recursive least squares
# ---------------------------------------------------------------------------
def rls(order, delta, record=False):
    """
    Process the rows of A_aug in the given order, one at a time.
    Returns the final w and (optionally) the history of every step.
    """
    P = np.eye(p) / delta                  # P_0 = I / delta
    w = np.zeros(p)
    hist, snapshots = [], {}
    for k, i in enumerate(order, start=1):
        a = A_aug[i]                       # the new row (length p)
        Pa = P @ a
        gain = Pa / (1.0 + a @ Pa)         # k = P a / (1 + a^T P a)
        w = w + gain * (y_aug[i] - a @ w)  # correct with the new residual
        P = P - np.outer(gain, Pa)         # Sherman-Morrison update
        P = 0.5 * (P + P.T)                # keep P symmetric (round-off)
        if record:
            hist.append((k, loss(w) / L_ref, grid_rmse(w)))
            snapshots[k] = w.copy()
    return w, np.array(hist), snapshots


# The 6 orthogonality rows go first; then the data rows.
con_rows = np.arange(N_DATA, n_rows)
order_sorted = np.concatenate([con_rows, np.argsort(X[:, 0])])        # sweep in x_1
order_random = np.concatenate([con_rows, np.random.default_rng(1).permutation(N_DATA)])

t0 = time.perf_counter()
w_sorted, hist_sorted, snaps = rls(order_sorted, DELTA, record=True)
t_sorted = time.perf_counter() - t0
w_random, hist_random, _ = rls(order_random, DELTA, record=True)

# The solution RLS should reproduce: (A^T A + delta I)^{-1} A^T y
w_ridge = np.linalg.solve(A_aug.T @ A_aug + DELTA * np.eye(p), A_aug.T @ y_aug)


# ---------------------------------------------------------------------------
# 3. Results
# ---------------------------------------------------------------------------
print("\n" + "-" * 84)
print(f"{'method':38s}{'L/L_min':>12s}{'RMSE vs f':>11s}{'rows processed':>16s}{'passes':>8s}")
print("-" * 84)
print(f"{'lstsq (exact, reference)':38s}{1.0:12.6f}{grid_rmse(w_ref):11.4f}{n_rows:16d}{'-':>8s}")
print(f"{'RLS, rows sorted by x1':38s}{loss(w_sorted) / L_ref:12.6f}"
      f"{grid_rmse(w_sorted):11.4f}{n_rows:16d}{1:8d}")
print(f"{'RLS, random order':38s}{loss(w_random) / L_ref:12.6f}"
      f"{grid_rmse(w_random):11.4f}{n_rows:16d}{1:8d}")
print("-" * 84)
print("For comparison (Examples 3 and 4): Adam ~2.6e6 rows, L/L_min = 3.2;")
print("                                    stochastic Gauss-Newton ~1.8e5 rows, L/L_min = 1.04")
print(f"\nSorted vs random order, final w:  ||w_sorted - w_random|| / ||w_sorted|| = "
      f"{np.linalg.norm(w_sorted - w_random) / np.linalg.norm(w_sorted):.1e}")
print(f"RLS reproduces (A^T A + delta I)^-1 A^T y:  relative difference = "
      f"{np.linalg.norm(w_sorted - w_ridge) / np.linalg.norm(w_ridge):.1e}")
print(f"RLS time: {t_sorted:.3f} s (including the loss evaluations used for the plots)")


# ---------------------------------------------------------------------------
# 4. Effect of delta: regularization (large delta) vs round-off (tiny delta)
# ---------------------------------------------------------------------------
print("\nEffect of delta (P_0 = I/delta), rows sorted by x1:")
print(f"{'delta':>10s}{'L/L_min':>12s}{'RMSE vs f':>11s}{'round-off  ||w - w_ridge||/||w_ridge||':>40s}")
d_L, d_rm, d_ro = [], [], []
for delta in DELTAS:
    w_d, _, _ = rls(order_sorted, delta)
    w_r = np.linalg.solve(A_aug.T @ A_aug + delta * np.eye(p), A_aug.T @ y_aug)
    d_L.append(loss(w_d) / L_ref)
    d_rm.append(grid_rmse(w_d))
    d_ro.append(np.linalg.norm(w_d - w_r) / np.linalg.norm(w_r))
    print(f"{delta:10.0e}{d_L[-1]:12.6f}{d_rm[-1]:11.4f}{d_ro[-1]:40.1e}")


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(13, 10))

# (a) loss vs rows processed
ax = axs[0, 0]
ax.semilogy(hist_sorted[:, 0], hist_sorted[:, 1], color="tab:blue", label="rows sorted by $x_1$")
ax.semilogy(hist_random[:, 0], hist_random[:, 1], color="tab:orange", label="random order")
ax.axhline(1.0, color="k", ls="--", label="exact minimum")
ax.axvline(6 + p, color="gray", ls=":", label=f"6 constraint rows + p = {p} data rows")
ax.set_title("Loss on ALL data, $L/L_{min}$, during one pass")
ax.set_xlabel("rows processed"); ax.set_ylabel("$L/L_{min}$")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

# (b) error against the true function vs rows processed
ax = axs[0, 1]
ax.semilogy(hist_sorted[:, 0], hist_sorted[:, 2], color="tab:blue", label="rows sorted by $x_1$")
ax.semilogy(hist_random[:, 0], hist_random[:, 2], color="tab:orange", label="random order")
ax.axhline(grid_rmse(w_ref), color="k", ls="--", label="exact minimum")
ax.axhline(NOISE, color="gray", ls=":", label=r"noise level $\sigma$")
ax.set_title("RMSE vs true $f$ during one pass")
ax.set_xlabel("rows processed"); ax.set_ylabel("RMSE on 80x80 grid")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

# (c) effect of delta
ax = axs[1, 0]
ax.loglog(DELTAS, np.array(d_L) - 1.0, "o-", label=r"excess loss $L/L_{min} - 1$")
ax.loglog(DELTAS, d_ro, "s--", label=r"round-off: $\|w - w_\delta\| / \|w_\delta\|$")
ax.axvline(DELTA, color="gray", ls=":", label=fr"chosen $\delta$ = {DELTA:.0e}")
ax.invert_xaxis()
ax.set_title(r"Choice of $\delta$ in $P_0 = I/\delta$")
ax.set_xlabel(r"$\delta$"); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

# (d) the model being built (rows sorted by x1), cut along x2 = 0.25
xs = np.linspace(0, 1, 300)
A_line = model_matrix(np.column_stack([xs, np.full_like(xs, 0.25)]), C, EPS)
ax = axs[1, 1]
ax.plot(xs, franke(xs, 0.25), "k", lw=2, label="true $f$")
for k, col in zip([56, 106, 156, n_rows], ["tab:purple", "tab:orange", "tab:olive", "tab:blue"]):
    seen = k - 6
    x_max = np.sort(X[:, 0])[seen - 1]
    ax.plot(xs, np.clip(A_line @ snaps[k], -0.5, 2.0), color=col, lw=1.3,
            label=f"after {seen} data rows ($x_1 \\leq {x_max:.2f}$)")
ax.set_ylim(-0.2, 1.6)
ax.set_title("Model built row by row (rows sorted by $x_1$), cut at $x_2 = 0.25$")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$s(x_1, 0.25)$")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("example05_results.png", dpi=150)
print("\nFigure saved to: example05_results.png")
plt.show()
