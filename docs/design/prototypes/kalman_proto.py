import numpy as np
from scipy.stats import norm

def kf_rts(Y, q, r=1.0, m0=0.0, p0=1e6):
    """Vectorized local-level Kalman filter + RTS smoother.
    Y: (E, T) array, NaN = missing. Returns smoothed mean, var, and the
    smoother weight matrix rows are implicit (we compute null sd separately)."""
    E, T = Y.shape
    mf = np.empty((E, T)); Pf = np.empty((E, T)); mp = np.empty((E, T)); Pp = np.empty((E, T))
    m = np.full(E, m0); P = np.full(E, p0)
    for t in range(T):
        mpred, Ppred = m, P + q          # x_t = x_{t-1} + eta
        mp[:, t], Pp[:, t] = mpred, Ppred
        y = Y[:, t]; obs = ~np.isnan(y)
        S = Ppred + r                     # innovation variance
        K = np.where(obs, Ppred / S, 0.0)
        m = mpred + K * np.where(obs, y - mpred, 0.0)
        P = (1 - K) * Ppred
        mf[:, t], Pf[:, t] = m, P
    ms = mf.copy(); Ps = Pf.copy()
    for t in range(T - 2, -1, -1):
        G = Pf[:, t] / Pp[:, t + 1]
        ms[:, t] = mf[:, t] + G * (ms[:, t + 1] - mp[:, t + 1])
        Ps[:, t] = Pf[:, t] + G**2 * (Ps[:, t + 1] - Pp[:, t + 1])
    return ms, Ps

def null_sd(T, q, r=1.0, mask=None):
    """sd of smoothed mean under H0 (y iid N(0,1)), via smoother on unit vectors
    (the smoother is linear in y)."""
    I = np.eye(T)
    if mask is not None:
        I = np.where(mask[None, :], I, np.nan)
    A, _ = kf_rts(I, q, r)          # A[s, t] = weight of y_s in m_{t|T}
    if mask is not None:
        A = np.where(mask[:, None], A, 0.0)
    return np.sqrt((A**2).sum(axis=0))

rng = np.random.default_rng(0)
T, E = 40, 20000
q = 0.05
# Null calibration: y iid N(0,1)
Y0 = rng.standard_normal((E, T))
ms, Ps = kf_rts(Y0, q)
sd0 = null_sd(T, q)
z = ms / sd0
p = norm.sf(z)
print("null: frac p<0.05 per t (mean over t):", (p < 0.05).mean().round(4),
      " frac p<0.01:", (p < 0.01).mean().round(4))
# naive Bayesian-posterior z = ms/sqrt(Ps) is NOT a calibrated p-value:
pb = norm.sf(ms / np.sqrt(Ps))
print("naive posterior-tail 'p'<0.05 under null:", (pb < 0.05).mean().round(4))
# with missingness pattern (same for all edges)
mask = rng.random(T) > 0.3
Ym = np.where(mask[None, :], Y0, np.nan)
msm, _ = kf_rts(Ym, q)
zm = msm / null_sd(T, q, mask=mask)
print("null w/ 30% missing: frac p<0.05:", (norm.sf(zm) < 0.05).mean().round(4))
# Power: edge with signal mu=0.8 only in t in [10, 20)
mu = np.zeros(T); mu[10:20] = 0.8
Y1 = rng.standard_normal((E, T)) + mu
ms1, _ = kf_rts(Y1, q)
p1 = norm.sf(ms1 / sd0)
psnap = norm.sf(Y1)
print("power at t=15: smoothed", (p1[:, 15] < 0.05).mean().round(3), " per-snapshot", (psnap[:, 15] < 0.05).mean().round(3))
print("false pos at t=30: smoothed", (p1[:, 30] < 0.05).mean().round(3), " per-snapshot", (psnap[:, 30] < 0.05).mean().round(3))

# Pooled MLE of q via prediction-error decomposition (Wills&Schon eq 75)
def loglik(Y, q, r=1.0, p0=1e6):
    E, T = Y.shape; m = np.zeros(E); P = np.full(E, p0); ll = 0.0
    for t in range(T):
        Pp = P + q; S = Pp + r; v = Y[:, t] - m
        if t > 0: ll += (-0.5 * (np.log(2*np.pi*S) + v**2 / S)).sum()
        K = Pp / S; m = m + K * v; P = (1 - K) * Pp
    return ll
Ymix = np.vstack([Y0[:5000], Y1[:5000]])
qs = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
print("pooled loglik over q:", {qq: round(loglik(Ymix, qq), 1) for qq in qs})

def null_var_recursion(mask, q, r=1.0, p0=1e6):
    """O(T) exact Var_H0 of smoothed mean (y iid N(0,r) when observed)."""
    T = len(mask); K = np.zeros(T); Pf = np.zeros(T); Pp = np.zeros(T)
    P = p0
    for t in range(T):
        Ppred = P + q; Pp[t] = Ppred
        K[t] = Ppred / (Ppred + r) if mask[t] else 0.0
        P = (1 - K[t]) * Ppred; Pf[t] = P
    v = np.zeros(T); prev = 0.0
    for t in range(T):                      # Joseph form with true q = 0
        prev = (1 - K[t])**2 * prev + K[t]**2 * r; v[t] = prev
    V = np.zeros(T); V[-1] = v[-1]; chi = 1.0
    for t in range(T - 2, -1, -1):
        G = Pf[t] / Pp[t + 1]
        c = v[t] * (1 - K[t + 1]) * chi
        V[t] = (1 - G)**2 * v[t] + G**2 * V[t + 1] + 2 * G * (1 - G) * c
        chi = (1 - G) + G * (1 - K[t + 1]) * chi
    return V

for qq in (0.01, 0.05, 1.0):
    for msk in (np.ones(T, bool), mask):
        a = null_sd(T, qq, mask=None if msk.all() else msk)**2
        b = null_var_recursion(msk, qq)
        print("q", qq, "missing" if not msk.all() else "full", "max abs diff", float(np.abs(a - b).max()))
