"""
Type B done correctly: synthetic MEASUREMENT errors on deterministic data.

The wake is deterministic (fixed k). We corrupt the analyst's observations the
way GUM / Quick et al. (2025) prescribe:
  * Type A  -- random per-sample sensor noise (power), reducible by averaging.
  * Type B  -- a SYSTEMATIC calibration bias drawn once per campaign, correlated
               with the control (yaw) state (a yaw-dependent power/anemometer
               offset -- an IEA Task 44 source). Constant over the campaign,
               so it is invisible to resampling and IRREDUCIBLE by more data.

The benefit truth is fixed (clean, deterministic). We ask, vs campaign length,
how often each reported interval covers that true benefit:
  (1) bootstrap only         (Type A)  -> misses the systematic, coverage decays
  (2) bootstrap (+) Type B   (honest)  -> propagate the calibration prior, holds

Signature of genuine Type B: no campaign length removes the gap; only knowing
the calibration uncertainty (Type B) closes it.
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
K_FIX = 0.034                              # fixed, deterministic wake (no k games)
SIG_A = 0.02                               # Type A: 2% random per-sample power noise
SIG_B = 0.005                              # Type B: 0.5% systematic, yaw-correlated
Z = 1.959964
WSB = np.arange(4, 25, 1.0); NB = len(WSB) + 2


def fp(w, s, c):
    w = np.atleast_1d(w); s = np.clip(np.atleast_1d(s), 3, 25)
    k = np.full(np.shape(w), K_FIX)
    return (F_ON if c else F_OFF)(np.column_stack([w, s, k]))


# clean deterministic on/off power for the sector conditions (one year)
PON1, POFF1 = fp(wd_s, ws_s, 1), fp(wd_s, ws_s, 0)


def benefit(p_on, p_off, on, dg, pref):
    """ws-binned signed-area benefit [%] from a toggle campaign (observed power)."""
    p = np.where(on, p_on, p_off)
    off = ~on
    so = np.bincount(dg[on], p[on], NB); no = np.bincount(dg[on], None, NB)
    sf = np.bincount(dg[off], p[off], NB); nf = np.bincount(dg[off], None, NB)
    v = (no >= 3) & (nf >= 3)
    mo = so / np.where(no > 0, no, 1); mf = sf / np.where(nf > 0, nf, 1)
    w = (no + nf) * v; sw = w.sum()
    return 100 * (w * (mo - mf)).sum() / sw / pref if sw > 0 else np.nan


# true (clean) benefit -- the target, independent of campaign length
on0 = ((np.arange(len(wd_s)) // 7) % 2 == 1)
dg0 = np.digitize(ws_s, WSB)
pref0 = POFF1.mean()
TRUE = benefit(PON1, POFF1, on0, dg0, pref0)

# Type-B sensitivity: how much a systematic ON-power bias b shifts the benefit
eps = 1e-3
dDdb = (benefit(PON1 * (1 + eps), POFF1, on0, dg0, pref0) - TRUE) / eps
hw_typeB = Z * abs(dDdb) * SIG_B           # propagated Type-B 95% half-width [pp]
print(f"true benefit = {TRUE:.3f}%   dBenefit/db = {dDdb:.1f}   "
      f"Type-B 95% half-width = {hw_typeB:.3f} pp")


def block_boot(p_on, p_off, on, dg, t, pref, nb=200, rng=None):
    blk = (t / 2).astype(int); ub = np.unique(blk)
    mem = [np.where(blk == b)[0] for b in ub]; nbk = len(ub)
    p = np.where(on, p_on, p_off)
    est = np.empty(nb)
    for i in range(nb):
        sel = np.concatenate([mem[j] for j in rng.integers(0, nbk, nbk)])
        off = ~on[sel]; ons = on[sel]; ps = p[sel]; d = dg[sel]
        so = np.bincount(d[ons], ps[ons], NB); no = np.bincount(d[ons], None, NB)
        sf = np.bincount(d[off], ps[off], NB); nf = np.bincount(d[off], None, NB)
        v = (no >= 3) & (nf >= 3)
        mo = so / np.where(no > 0, no, 1); mf = sf / np.where(nf > 0, nf, 1)
        w = (no + nf) * v; sw = w.sum()
        est[i] = 100 * (w * (mo - mf)).sum() / sw / pref if sw > 0 else np.nan
    lo, hi = np.nanpercentile(est, [2.5, 97.5]); return (hi - lo) / 2


Ls = [0.25, 0.5, 1, 2, 4, 8]
cov_b, cov_h, hwb_m = [], [], []
print(f"\n{'L':>6} {'boot cov':>9} {'honest cov':>11} {'hw_boot':>8} {'hw_TypeB':>9}")
for L in Ls:
    nyr = int(np.ceil(L))
    pon = np.tile(PON1, nyr); poff = np.tile(POFF1, nyr)
    t = np.concatenate([t_s + m * TY for m in range(nyr)])
    dg = np.tile(dg0, nyr)
    on = ((np.arange(len(pon)) // 7) % 2 == 1)
    msk = t < L * TY
    pon, poff, t, dg, on = pon[msk], poff[msk], t[msk], dg[msk], on[msk]
    R = 250; cb = ch = 0; hwbs = []
    for r in range(R):
        rng = np.random.default_rng(6000 + r)
        b = rng.normal(0, SIG_B)                          # systematic (Type B), once
        po = pon * (1 + b) * (1 + SIG_A * rng.standard_normal(len(pon)))   # ON: +bias +noise
        pf = poff * (1 + SIG_A * rng.standard_normal(len(poff)))           # OFF: noise only
        pref = pf[~on].mean()
        dh = benefit(po, pf, on, dg, pref)
        hwb = block_boot(po, pf, on, dg, t, pref, nb=200, rng=rng)
        e = abs(dh - TRUE)
        cb += (e <= hwb); ch += (e <= np.hypot(hwb, hw_typeB)); hwbs.append(hwb)
    cov_b.append(100 * cb / R); cov_h.append(100 * ch / R); hwb_m.append(np.mean(hwbs))
    print(f"{L:>5}y {100*cb/R:>8.1f}% {100*ch/R:>10.1f}% {np.mean(hwbs):>7.3f} {hw_typeB:>8.3f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
x = np.array(Ls)
ax[0].plot(x, cov_b, "o-", color="#1f77b4", label="bootstrap only (Type A)")
ax[0].plot(x, cov_h, "^-", color="k", label="bootstrap ⊕ Type B (honest)")
ax[0].axhline(95, ls="--", color=".4", label="nominal 95%")
ax[0].set_xscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels([f"{l:g}y" for l in Ls])
ax[0].set_ylim(0, 102); ax[0].set_xlabel("campaign length")
ax[0].set_ylabel("coverage of TRUE benefit [%]")
ax[0].set_title("Genuine Type B = systematic measurement bias\n"
                "no campaign length removes it; bootstrap decays toward 0", fontsize=10)
ax[0].legend(fontsize=8.5); ax[0].grid(alpha=0.3)

ax[1].plot(x, hwb_m, "o-", color="#1f77b4", label="bootstrap half-width (Type A)")
ax[1].axhline(hw_typeB, ls="--", color="#d62728", lw=2,
              label=f"Type-B half-width (fixed = {hw_typeB:.2f} pp)")
ax[1].set_xscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels([f"{l:g}y" for l in Ls])
ax[1].set_xlabel("campaign length"); ax[1].set_ylabel("half-width [pp]")
ax[1].set_ylim(bottom=0)
ax[1].set_title("Type A → 0 with data; the systematic Type B is a hard floor\n"
                "(reducible vs irreducible — the real distinction)", fontsize=10)
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig_typeB_measurement.png", dpi=130); plt.close()
print("\nWrote fig_typeB_measurement.png")
