"""
Is the bootstrap a reasonable substitute for Type B?  NO.

Compares, as a function of campaign length, the coverage of the true future-year
benefit by three reported intervals:
  (1) bootstrap only          (Type A  -- what colleagues report)
  (2) propagated Type B only  (epistemic k through the wake response)
  (3) both, in quadrature     (the honest interval)

If the bootstrap were a substitute for Type B, (1) would track 95%. It does not:
it is never calibrated and gets WORSE the longer (more trustworthy) the campaign.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pywake_model import build_lookup, K_GRID

F_OFF, F_ON, _, _ = build_lookup()
from py_wake.examples.data import example_data_path
d = np.load(example_data_path + "/time_series.npz"); WD, WS = d["wd"], d["ws"]
DT = 1 / 6 / 24; TY = len(WD) * DT
S = (WD >= 266) & (WD <= 274) & (WS >= 3) & (WS <= 25); idx = np.where(S)[0]
wd_s, ws_s, t_s = WD[idx], WS[idx], idx * DT
MU, SB, Z = 0.034, 0.004, 1.959964
WSB = np.arange(4, 25, 1.0); NB = len(WSB) + 2


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


def area_benefit_idx(p, on, dg, pref, sel=None):
    if sel is not None:
        p, on, dg = p[sel], on[sel], dg[sel]
    off = ~on
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


def boot_hw(p, on, dg, t, pref, nb=200, rng=None):
    blk = (t / 2).astype(int); ub = np.unique(blk)
    mem = [np.where(blk == b)[0] for b in ub]; nbk = len(ub)
    est = np.empty(nb)
    for i in range(nb):
        sel = np.concatenate([mem[j] for j in rng.integers(0, nbk, nbk)])
        est[i] = area_benefit_idx(p, on, dg, pref, sel)
    lo, hi = np.nanpercentile(est, [2.5, 97.5]); return (hi - lo) / 2


kc = np.linspace(K_GRID[0], K_GRID[-1], 60)
bcv = np.array([true_benefit(wd_s, ws_s, k) for k in kc])
b_mu = np.interp(MU, kc, bcv)


def typeB_hw(L, rng):
    """Propagated Type-B 95% half-width: k ~ N(mu, sigma_B*sqrt(1+1/L)) through
    the wake response Delta_true(k) (captures nonlinearity + prediction factor)."""
    sig = SB * np.sqrt(1 + 1 / L)
    delta = np.interp(MU + rng.normal(0, sig, 6000), kc, bcv) - b_mu
    return (np.percentile(delta, 97.5) - np.percentile(delta, 2.5)) / 2


Ls = [0.25, 0.5, 1, 2, 4, 8]
res = {"boot": [], "tb": [], "both": [], "hwboot": [], "hwtb": []}
print(f"{'L':>7} {'boot-only':>10} {'TypeB-only':>11} {'both':>7} "
      f"{'hw_boot':>8} {'hw_tb':>7}")
for L in Ls:
    R = 250
    cb = ct = cbo = 0; hwb_l = []
    hwtb = typeB_hw(L, np.random.default_rng(1))
    for r in range(R):
        rng = np.random.default_rng(4000 + r)
        wd, ws, t, k = build(L, rng)
        p, on = make_obs(wd, ws, k, rng)
        dg = np.digitize(ws, WSB); pref = p[~on].mean()
        dh = area_benefit_idx(p, on, dg, pref)
        hwb = boot_hw(p, on, dg, t, pref, nb=200, rng=rng)
        tgt = true_benefit(wd_s, ws_s, rng.normal(MU, SB))
        e = abs(dh - tgt)
        cb += (e <= hwb); ct += (e <= hwtb); cbo += (e <= np.hypot(hwb, hwtb))
        hwb_l.append(hwb)
    res["boot"].append(100 * cb / R); res["tb"].append(100 * ct / R)
    res["both"].append(100 * cbo / R); res["hwboot"].append(np.mean(hwb_l))
    res["hwtb"].append(hwtb)
    print(f"{L:>6}y {100*cb/R:>9.1f}% {100*ct/R:>10.1f}% {100*cbo/R:>6.1f}% "
          f"{np.mean(hwb_l):>7.2f}% {hwtb:>6.2f}%")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
x = np.array(Ls)
ax[0].plot(x, res["boot"], "o-", color="#1f77b4", label="bootstrap only (Type A)")
ax[0].plot(x, res["tb"], "s-", color="#d62728", label="propagated Type B only")
ax[0].plot(x, res["both"], "^-", color="k", label="both (honest)")
ax[0].axhline(95, ls="--", color=".4", label="nominal 95%")
ax[0].set_xscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels([f"{l:g}y" for l in Ls])
ax[0].set_ylim(40, 102); ax[0].set_xlabel("campaign length")
ax[0].set_ylabel("coverage of true future benefit [%]")
ax[0].set_title("Bootstrap is NOT a substitute for Type B:\n"
                "its coverage falls as the campaign grows; Type B holds", fontsize=10)
ax[0].legend(fontsize=8.5); ax[0].grid(alpha=0.3)

ax[1].plot(x, res["hwboot"], "o-", color="#1f77b4", label="bootstrap half-width (Type A)")
ax[1].plot(x, res["hwtb"], "s-", color="#d62728", label="propagated Type-B half-width")
ax[1].set_xscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels([f"{l:g}y" for l in Ls])
ax[1].set_xlabel("campaign length"); ax[1].set_ylabel("reported half-width [%]")
ax[1].set_title("They only cross near 1 yr — a coincidence, not equivalence:\n"
                "bootstrap → 0, Type B is a fixed floor", fontsize=10)
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.3)
ax[1].annotate("bootstrap shrinks with N;\nType B is a fixed floor —\nnot the same quantity",
               (8, res["hwboot"][-1]), xytext=(1.2, res["hwboot"][0] * 0.78),
               fontsize=8, color=".3", arrowprops=dict(arrowstyle="->", color=".3"))
plt.tight_layout(); plt.savefig("fig_bootstrap_vs_typeB.png", dpi=130); plt.close()
print("\nWrote fig_bootstrap_vs_typeB.png")
