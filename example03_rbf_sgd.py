"""
Example 3 — The same RBF approximation with stochastic gradient methods
=======================================================================

Seminar: Introduction to approximation with radial basis functions (RBF)

Same data, centers, multiquadric (eps = 3) and quadratic polynomial as
Example 1. Two changes, inspired by common Machine Learning practice:

 (a) The orthogonality conditions P_c^T a = 0 are NOT imposed as constraints.
     They are appended, with a weight lambda, as 6 extra rows of the
     overdetermined system:

            [ Phi          P ] [a]      [y]
            [ lambda*Pc^T  0 ] [b]  ~=  [0]        ->   A_aug w ~= y_aug

     and we minimize the unconstrained loss

            L(w) = 1/2 ||A_aug w - y_aug||^2
                 = 1/2 sum_i (s(x_i) - y_i)^2  +  1/2 lambda^2 ||P_c^T a||^2

 (b) L(w) is minimized with stochastic gradient methods, written by hand in
     numpy so every step is visible:
        - SGD            w <- w - lr * g
        - SGD + momentum m <- beta*m + g,   w <- w - lr * m
        - Adam           per-coordinate adaptive step (standard in ML)
     Each step uses the gradient of a random mini-batch of data rows
     (rescaled by n/|B|) plus the 6 constraint rows, which are always kept.

     As ML practitioners do, we also try preprocessing the columns of A:
        - "raw"          : no preprocessing
        - "scaled"       : each column divided by its norm
        - "standardized" : RBF columns centered (mean 0) and scaled (std 1)
     Each preprocessing is a change of variables w = T v; we optimize v.

Reference solutions (robust minimization):
     - scipy.linalg.lstsq on the SAME augmented system: exact minimum of L
     - Example 1 (SLSQP, constraint satisfied exactly)

Run:       python example03_rbf_sgd.py
Requires:  numpy, scipy, matplotlib
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# 0. Example parameters
# ---------------------------------------------------------------------------
SEED = 0          # same data and centers as Example 1
N_DATA = 200      # n
M_CENTERS = 30    # m
EPS = 3.0         # multiquadric shape parameter
NOISE = 0.02      # noise standard deviation
LAMBDA = 1.0      # weight of the orthogonality rows

EPOCHS = 10000    # one epoch = one pass over the n data points
BATCH = 20        # mini-batch size  ->  n / BATCH = 10 steps per epoch
LR_ADAM = 1e-2    # Adam learning rate (typical ML value)
BETA = 0.9        # momentum coefficient
RECORD_EVERY = 10 # record the loss every RECORD_EVERY epochs

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. Data and centers (identical to Example 1)
# ---------------------------------------------------------------------------
def franke(x, y):
    """Classical test function for scattered data approximation."""
    t1 = 0.75 * np.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4)
    t2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
    t3 = 0.50 * np.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4)
    t4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return t1 + t2 + t3 + t4


X = rng.random((N_DATA, 2))
y = franke(X[:, 0], X[:, 1]) + NOISE * rng.standard_normal(N_DATA)
C = X[rng.choice(N_DATA, size=M_CENTERS, replace=False)]


# ---------------------------------------------------------------------------
# 2. Model ingredients (same as Example 1)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3. Augmented overdetermined system  A_aug w ~= y_aug
# ---------------------------------------------------------------------------
A = model_matrix(X, C, EPS)                                        # n x (m+6)
A_con = np.hstack([LAMBDA * quadratic_poly(C).T, np.zeros((6, 6))]) # 6 x (m+6)
A_aug = np.vstack([A, A_con])                                      # (n+6) x (m+6)
y_aug = np.concatenate([y, np.zeros(6)])
n_par = A_aug.shape[1]


def loss(w):
    r = A_aug @ w - y_aug
    return 0.5 * r @ r


# Fine grid to measure the error against the true function
ng = 80
gx, gy = np.meshgrid(np.linspace(0, 1, ng), np.linspace(0, 1, ng))
G_pts = np.column_stack([gx.ravel(), gy.ravel()])
A_grid = model_matrix(G_pts, C, EPS)
f_grid = franke(G_pts[:, 0], G_pts[:, 1])


def grid_rmse(w):
    return np.sqrt(np.mean((A_grid @ w - f_grid) ** 2))


def report(w):
    """Quantities used to compare all solutions."""
    sse = np.sum((A @ w - y) ** 2)                 # total squared error (data only)
    cons = np.linalg.norm(A_con @ w) / LAMBDA      # ||P_c^T a||
    return loss(w), sse, cons, grid_rmse(w)


print("=" * 70)
print("Example 3 — augmented system + stochastic gradient methods")
print("=" * 70)
print(f"A_aug: {A_aug.shape[0]} x {A_aug.shape[1]}  (n + 6 rows, m + 6 unknowns),"
      f"  lambda = {LAMBDA}")


# ---------------------------------------------------------------------------
# 4. Reference solutions (robust minimization)
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
w_ref, *_ = lstsq(A_aug, y_aug)                    # exact minimizer of L
t_ref = time.perf_counter() - t0
L_ref = loss(w_ref)

# Example 1: SLSQP with the constraint imposed exactly
G = np.hstack([quadratic_poly(C).T, np.zeros((6, 6))])
t0 = time.perf_counter()
sol = minimize(lambda w: 0.5 * np.sum((A @ w - y) ** 2), np.zeros(n_par),
               jac=lambda w: A.T @ (A @ w - y),
               constraints=[{"type": "eq", "fun": lambda w: G @ w, "jac": lambda w: G}],
               method="SLSQP", options={"maxiter": 2000, "ftol": 1e-12})
t_slsqp = time.perf_counter() - t0
w_slsqp = sol.x


# ---------------------------------------------------------------------------
# 5. Column preprocessing as a change of variables  w = T v
#    The optimizer works with A_aug @ T, whose columns are better scaled.
# ---------------------------------------------------------------------------
T_raw = np.eye(n_par)

T_scaled = np.diag(1.0 / np.linalg.norm(A_aug, axis=0))

# Standardization: column j -> (col_j - mean_j) / std_j for every column except
# the constant one (index M_CENTERS), which absorbs the removed means.
mu, sd = A.mean(axis=0), A.std(axis=0)
mu[M_CENTERS], sd[M_CENTERS] = 0.0, 1.0
T_std = np.diag(1.0 / sd)
T_std[M_CENTERS, :] -= mu / sd
T_std[M_CENTERS, M_CENTERS] = 1.0

PREPROC = {"raw": T_raw, "scaled": T_scaled, "standardized": T_std}

print("\nCondition number of the matrix seen by the optimizer:")
for name, T in PREPROC.items():
    print(f"  {name:13s}: cond(A_aug T) = {np.linalg.cond(A_aug @ T):.2e}")


# ---------------------------------------------------------------------------
# 6. Stochastic gradient optimizers
# ---------------------------------------------------------------------------
def train(method, T, lr, seed=1):
    """
    Minimize L(T v) with mini-batch stochastic gradients.
    Returns w = T v and the history of (epoch, loss, grid RMSE).
    """
    r = np.random.default_rng(seed)
    As, Acs = A @ T, A_con @ T                     # data rows, constraint rows
    v = np.zeros(n_par)
    m = np.zeros(n_par)                            # momentum / Adam 1st moment
    s = np.zeros(n_par)                            # Adam 2nd moment
    t = 0
    hist = []
    for epoch in range(1, EPOCHS + 1):
        perm = r.permutation(N_DATA)               # shuffle the data each epoch
        for k in range(0, N_DATA, BATCH):
            B = perm[k:k + BATCH]
            # stochastic gradient: mini-batch of data rows (rescaled to the
            # full data set) + the 6 constraint rows (always included)
            g = (N_DATA / len(B)) * As[B].T @ (As[B] @ v - y[B]) + Acs.T @ (Acs @ v)
            t += 1
            if method == "SGD":
                v -= lr * g
            elif method == "SGD+momentum":
                m = BETA * m + g
                v -= lr * m
            elif method == "Adam":
                m = 0.9 * m + 0.1 * g
                s = 0.999 * s + 0.001 * g * g
                m_hat, s_hat = m / (1 - 0.9 ** t), s / (1 - 0.999 ** t)
                v -= lr * m_hat / (np.sqrt(s_hat) + 1e-8)
        if epoch % RECORD_EVERY == 0:
            w = T @ v
            hist.append((epoch, loss(w), grid_rmse(w)))
    return T @ v, np.array(hist)


def learning_rate(method, T):
    """
    SGD and momentum: step based on the Lipschitz constant of the gradient,
    L_max = largest eigenvalue of (A_aug T)^T (A_aug T)  (classical choice).
    Adam: the usual ML default LR_ADAM.
    """
    L_max = np.linalg.eigvalsh((A_aug @ T).T @ (A_aug @ T)).max()
    if method == "SGD":
        return 0.5 / L_max
    if method == "SGD+momentum":
        return 0.5 * (1 - BETA) / L_max            # same effective step as SGD
    return LR_ADAM


METHODS = ["SGD", "SGD+momentum", "Adam"]
results = {}
print(f"\nTraining: {EPOCHS} epochs x {N_DATA // BATCH} steps = "
      f"{EPOCHS * N_DATA // BATCH} iterations per run")
for pname, T in PREPROC.items():
    for method in METHODS:
        lr = learning_rate(method, T)
        t0 = time.perf_counter()
        w, hist = train(method, T, lr)
        elapsed = time.perf_counter() - t0
        results[(pname, method)] = (w, hist, elapsed)
        print(f"  {pname:13s} {method:13s} lr = {lr:.1e}   "
              f"L = {loss(w):.4f}   ({elapsed:.1f} s)")


# ---------------------------------------------------------------------------
# 7. Comparison table
# ---------------------------------------------------------------------------
print("\n" + "-" * 108)
print(f"{'solution':38s}{'L(w)':>9s}{'L/L_min':>9s}{'SSE data':>11s}"
      f"{'||Pc^T a||':>12s}{'RMSE vs f':>11s}{'||w||':>9s}{'||w-w*||/||w*||':>17s}{'time':>8s}")
print("-" * 108)


def row(name, w, t):
    L, sse, cons, rm = report(w)
    dw = np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref)
    print(f"{name:38s}{L:9.4f}{L / L_ref:9.2f}{sse:11.4f}{cons:12.1e}{rm:11.4f}"
          f"{np.linalg.norm(w):9.1f}{dw:17.2e}{t:7.2f}s")


row("lstsq on augmented system (exact)", w_ref, t_ref)
row("Example 1: SLSQP, exact constraint", w_slsqp, t_slsqp)
for (pname, method), (w, hist, elapsed) in results.items():
    row(f"{method} ({pname})", w, elapsed)
print("-" * 108)
print("w* = exact minimizer (lstsq). Its large norm means large coefficients that")
print("cancel each other: directions of A with tiny singular values, which gradient")
print("steps starting from w = 0 explore extremely slowly.")
print(f"Noise level: sigma = {NOISE}   (best achievable RMSE vs f is of this order)")


# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------
colors = {"SGD": "tab:blue", "SGD+momentum": "tab:orange", "Adam": "tab:green"}
styles = {"raw": ":", "scaled": "--", "standardized": "-"}

fig, axs = plt.subplots(2, 2, figsize=(13, 10))

# (a) Loss vs epochs, all optimizers, standardized columns
ax = axs[0, 0]
for method in METHODS:
    h = results[("standardized", method)][1]
    ax.loglog(h[:, 0], h[:, 1], color=colors[method], label=method)
ax.axhline(L_ref, color="k", ls="--", label="exact minimum (lstsq)")
ax.set_title("Loss $L(w)$ — standardized columns")
ax.set_xlabel("epoch"); ax.set_ylabel("$L(w)$")
ax.grid(True, which="both", alpha=0.3); ax.legend()

# (b) Loss vs epochs, Adam, three preprocessings
ax = axs[0, 1]
for pname in PREPROC:
    h = results[(pname, "Adam")][1]
    ax.loglog(h[:, 0], h[:, 1], color=colors["Adam"], ls=styles[pname], label=f"Adam, {pname}")
ax.axhline(L_ref, color="k", ls="--", label="exact minimum (lstsq)")
ax.set_title("Effect of column preprocessing (Adam)")
ax.set_xlabel("epoch"); ax.set_ylabel("$L(w)$")
ax.grid(True, which="both", alpha=0.3); ax.legend()

# (c) Error against the true function vs epochs
ax = axs[1, 0]
for method in METHODS:
    h = results[("standardized", method)][1]
    ax.loglog(h[:, 0], h[:, 2], color=colors[method], label=method)
ax.axhline(grid_rmse(w_ref), color="k", ls="--", label="exact minimum (lstsq)")
ax.axhline(NOISE, color="gray", ls=":", label=r"noise level $\sigma$")
ax.set_title("RMSE vs true $f$ — standardized columns")
ax.set_xlabel("epoch"); ax.set_ylabel("RMSE on 80x80 grid")
ax.grid(True, which="both", alpha=0.3); ax.legend()

# (d) Best stochastic model vs exact minimum along a line x2 = 0.25
best_key = min(results, key=lambda k: loss(results[k][0]))
w_best = results[best_key][0]
xs = np.linspace(0, 1, 300)
line = np.column_stack([xs, np.full_like(xs, 0.25)])
A_line = model_matrix(line, C, EPS)
ax = axs[1, 1]
ax.plot(xs, franke(xs, 0.25), "k", lw=2, label="true $f$")
ax.plot(xs, A_line @ w_ref, "--", color="tab:red", label="exact minimum (lstsq)")
ax.plot(xs, A_line @ w_best, color=colors[best_key[1]],
        label=f"best stochastic: {best_key[1]} ({best_key[0]})")
near = np.abs(X[:, 1] - 0.25) < 0.05
ax.scatter(X[near, 0], y[near], s=12, c="gray", label=r"data with $|x_2-0.25|<0.05$")
ax.set_title("Cut along $x_2 = 0.25$")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$s(x_1, 0.25)$")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("example03_results.png", dpi=150)
print("\nFigure saved to: example03_results.png")
plt.show()
