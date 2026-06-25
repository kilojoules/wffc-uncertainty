"""
Test Julian's proposed reporting recipe:
  benefit = signed area between the ON/OFF farm-power CDFs, conditioned on ws bin
          = energy-weighted mean power gain (aleatoric integrated into the CDFs)
  reported uncertainty = Type B ONLY (epistemic k), no bootstrap.

Question: does [area-benefit +/- Type B] cover the true future-year benefit?
We sweep campaign length to find where "Type-B only" becomes sufficient, i.e.
where the metric's finite-sample (Type-A) estimation error has died away.
"""
import numpy as np
from pywake_model import build_lookup, K_GRID

RNG = np.random.default_rng(11)
F_OFF, F_ON, _, _ = build_lookup()
from py_wake.examples.data import example_data_path
d = np.load(example_data_path + "/time_series.npz"); WD, WS = d["wd"], d["ws"]
DT = 1 / 6 / 24; TY = len(WD) * DT
S = (WD >= 266) & (WD <= 274) & (WS >= 3) & (WS <= 25); idx = np.where(S)[0]
wd_s, ws_s, t_s = WD[idx], WS[idx], idx * DT
MU, SB, Z = 0.034, 0.004, 1.959964
WSB = np.arange(4, 25, 1.0); NB = len(WSB) + 2          # 1-m/s ws bins


def fp(w, s, k, c):
    w = np.atleast_1d(w); s = np.clip(np.atleast_1d(s), 3, 25)
    k = np.broadcast_to(np.atleast_1d(k), w.shape)
    return (F_ON if c else F_OFF)(np.column_stack([w, s, k]))


def true_benefit(w, s, k):
    po = fp(w, s, k, 0); pn = fp(w, s, k, 1)
    return 100 * (pn.sum() - po.sum()) / po.sum()


def seasonal_k(ky, rng, amp=0.008):
    seas = amp * np.sin(2 * np.pi * t_s / TY)
    nd = int(np.ceil(t_s.max())) + 2; w = np.zeros(nd)
    for i in range(1, nd):
        w[i] = 0.96 * w[i - 1] + 0.0015 * rng.standard_normal()
    return np.clip(ky + seas + np.interp(t_s, np.arange(nd), w), K_GRID[0], K_GRID[-1])


def area_benefit(p, on, wr, pref):
    """Signed area between ON/OFF power CDFs per 1-m/s ws bin, energy-weighted [%]."""
    dg = np.digitize(wr, WSB); off = ~on
    so = np.bincount(dg[on], p[on], NB); no = np.bincount(dg[on], None, NB)
    sf = np.bincount(dg[off], p[off], NB); nf = np.bincount(dg[off], None, NB)
    v = (no >= 3) & (nf >= 3)
    mo = so / np.where(no > 0, no, 1); mf = sf / np.where(nf > 0, nf, 1)
    w = (no + nf) * v; sw = w.sum()
    return 100 * (w * (mo - mf)).sum() / sw / pref if sw > 0 else np.nan


def build(L, rng):
    nyr = int(np.ceil(L))
    wd = np.tile(wd_s, nyr); ws = np.tile(ws_s, nyr)
    t = np.concatenate([t_s + m * TY for m in range(nyr)])
    k = np.concatenate([seasonal_k(rng.normal(MU, SB), rng) for _ in range(nyr)])
    msk = t < L * TY
    return wd[msk], ws[msk], t[msk], k[msk]


def make_obs(wd, ws, k, rng, noise=0.01):
    po = fp(wd, ws, k, 0); pn = fp(wd, ws, k, 1)
    on = ((np.arange(len(wd)) // 7) % 2 == 1)
    return np.where(on, pn, po) * (1 + noise * rng.standard_normal(len(wd))), on


kc = np.linspace(K_GRID[0], K_GRID[-1], 60)
bcv = np.array([true_benefit(wd_s, ws_s, k) for k in kc])
slope = np.interp(MU, kc, np.gradient(bcv, kc))
tb = Z * abs(slope) * SB                                 # Type-B 95% half-width [%]
print(f"slope={slope:.0f} %/k   Type-B 95% half-width = {tb:.2f}%   (1x)")
print(f"\n{'campaign':>9} {'Type-B only':>12} {'Type-B*sqrt2':>13} {'mean|err|':>10}")
print(f"{'length':>9} {'coverage':>12} {'coverage':>13} {'[%]':>10}")
for L in [0.25, 0.5, 1, 2, 4, 8]:
    R = 250
    c1 = c2 = 0; errs = []
    for r in range(R):
        rng = np.random.default_rng(4000 + r)
        wd, ws, t, k = build(L, rng)
        p, on = make_obs(wd, ws, k, rng)
        pref = p[~on].mean()
        dh = area_benefit(p, on, ws, pref)
        tgt = true_benefit(wd_s, ws_s, rng.normal(MU, SB))   # independent future year
        e = abs(dh - tgt); errs.append(e)
        c1 += (e <= tb); c2 += (e <= tb * np.sqrt(2))
    yr = f"{L:g} yr"
    print(f"{yr:>9} {100*c1/R:>11.1f}% {100*c2/R:>12.1f}% {np.mean(errs):>10.2f}")
