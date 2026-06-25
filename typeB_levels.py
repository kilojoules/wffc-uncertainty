"""
Coverage vs campaign length at different LEVELS of Type B.

Type B = a systematic, yaw-correlated measurement bias drawn once per campaign,
b ~ N(0, sigma_B). It shifts the benefit estimate by exactly dDelta/db * b
(linear), independent of the Type-A sampling noise -- so we compute the
expensive block bootstrap once per campaign (noise only) and sweep sigma_B on
top analytically.

Levels: sigma_B in {0, 0.25, 0.5, 1, 2}% systematic.  sigma_B = 0 is the control
-- with no Type B the bootstrap is correctly calibrated.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pywake_model import build_lookup

F_OFF, F_ON, _, _ = build_lookup()
from py_wake.examples.data import example_data_path
d = np.load(example_data_path + "/time_series.npz"); WD, WS = d["wd"], d["ws"]
DT = 1 / 6 / 24; TY = len(WD) * DT
S = (WD >= 266) & (WD <= 274) & (WS >= 3) & (WS <= 25); idx = np.where(S)[0]
wd_s, ws_s, t_s = WD[idx], WS[idx], idx * DT
K_FIX, SIG_A, Z = 0.034, 0.02, 1.959964
WSB = np.arange(4, 25, 1.0); NB = len(WSB) + 2


def fp(w, s, c):
    w = np.atleast_1d(w); s = np.clip(np.atleast_1d(s), 3, 25)
    return (F_ON if c else F_OFF)(np.column_stack([w, s, np.full(np.shape(w), K_FIX)]))


PON1, POFF1 = fp(wd_s, ws_s, 1), fp(wd_s, ws_s, 0)
on0 = ((np.arange(len(wd_s)) // 7) % 2 == 1); dg0 = np.digitize(ws_s, WSB)


def benefit(p, on, dg, pref):
    off = ~on
    so = np.bincount(dg[on], p[on], NB); no = np.bincount(dg[on], None, NB)
    sf = np.bincount(dg[off], p[off], NB); nf = np.bincount(dg[off], None, NB)
    v = (no >= 3) & (nf >= 3)
    mo = so / np.where(no > 0, no, 1); mf = sf / np.where(nf > 0, nf, 1)
    w = (no + nf) * v; sw = w.sum()
    return 100 * (w * (mo - mf)).sum() / sw / pref if sw > 0 else np.nan


TRUE = benefit(np.where(on0, PON1, POFF1), on0, dg0, POFF1.mean())
eps = 1e-3
dDdb = (benefit(np.where(on0, PON1 * (1 + eps), POFF1), on0, dg0, POFF1.mean()) - TRUE) / eps
print(f"true benefit = {TRUE:.3f}%   dBenefit/db = {dDdb:.1f}")


def block_boot(p, on, dg, t, pref, nb=200, rng=None):
    blk = (t / 2).astype(int); ub = np.unique(blk)
    mem = [np.where(blk == b)[0] for b in ub]; nbk = len(ub)
    est = np.empty(nb)
    for i in range(nb):
        sel = np.concatenate([mem[j] for j in rng.integers(0, nbk, nbk)])
        off = ~on[sel]; ons = on[sel]; ps = p[sel]; dd = dg[sel]
        so = np.bincount(dd[ons], ps[ons], NB); no = np.bincount(dd[ons], None, NB)
        sf = np.bincount(dd[off], ps[off], NB); nf = np.bincount(dd[off], None, NB)
        v = (no >= 3) & (nf >= 3)
        mo = so / np.where(no > 0, no, 1); mf = sf / np.where(nf > 0, nf, 1)
        w = (no + nf) * v; sw = w.sum()
        est[i] = 100 * (w * (mo - mf)).sum() / sw / pref if sw > 0 else np.nan
    lo, hi = np.nanpercentile(est, [2.5, 97.5]); return (hi - lo) / 2


# expensive part once per campaign length: Type-A sampling scatter + bootstrap hw
Ls = np.array([0.25, 0.5, 1, 2, 4, 8])
R = 300
dsamp = {}; hwb = {}
for L in Ls:
    nyr = int(np.ceil(L))
    pon = np.tile(PON1, nyr); poff = np.tile(POFF1, nyr)
    t = np.concatenate([t_s + m * TY for m in range(nyr)]); dg = np.tile(dg0, nyr)
    on = ((np.arange(len(pon)) // 7) % 2 == 1); msk = t < L * TY
    pon, poff, t, dg, on = pon[msk], poff[msk], t[msk], dg[msk], on[msk]
    ds = np.empty(R); hh = np.empty(R)
    for r in range(R):
        rng = np.random.default_rng(7000 + r)
        po = pon * (1 + SIG_A * rng.standard_normal(len(pon)))   # Type-A noise only
        pf = poff * (1 + SIG_A * rng.standard_normal(len(poff)))
        pref = pf[~on].mean()
        ds[r] = benefit(np.where(on, po, pf), on, dg, pref) - TRUE
        hh[r] = block_boot(np.where(on, po, pf), on, dg, t, pref, nb=200, rng=rng)
    dsamp[L] = ds; hwb[L] = hh
    print(f"  L={L:>4}y  mean hw_boot={hh.mean():.3f} pp")

# sweep Type-B levels analytically (systematic bias = dDdb * sigma_B * z)
levels = [0.0, 0.0025, 0.005, 0.01, 0.02]      # systematic sigma_B (fractional)
colors = ["#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#d62728"]
cov_boot = {s: [] for s in levels}; cov_hon = {s: [] for s in levels}
zr = np.random.default_rng(123).standard_normal((len(Ls), R))   # shared bias draws
for s in levels:
    tb = Z * abs(dDdb) * s                       # Type-B 95% half-width [pp]
    for li, L in enumerate(Ls):
        bias = dDdb * s * zr[li]                  # systematic shift per realization
        err = np.abs(dsamp[L] + bias)
        cov_boot[s].append(100 * np.mean(err <= hwb[L]))
        cov_hon[s].append(100 * np.mean(err <= np.hypot(hwb[L], tb)))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for s, c in zip(levels, colors):
    tb = Z * abs(dDdb) * s
    lab = f"$\\sigma_B$={s*100:.2g}%  (floor {tb:.2f} pp)" + ("  [control]" if s == 0 else "")
    ax[0].plot(Ls, cov_boot[s], "o-", color=c, label=lab)
    ax[1].plot(Ls, cov_hon[s], "o-", color=c, label=lab)
for a, ttl in zip(ax, ["Bootstrap only (Type A) — colleagues' report",
                       "Bootstrap ⊕ propagated Type B — honest report"]):
    a.axhline(95, ls="--", color=".4"); a.set_xscale("log"); a.set_xticks(Ls)
    a.set_xticklabels([f"{l:g}y" for l in Ls]); a.set_ylim(0, 103)
    a.set_xlabel("campaign length"); a.set_ylabel("coverage of TRUE benefit [%]")
    a.set_title(ttl, fontsize=10); a.grid(alpha=0.3); a.legend(fontsize=8, loc="lower left")
ax[0].text(0.50, 0.40, "no Type B (green) = bootstrap fine;\nmore Type B = collapses, and\nfaster the longer you measure",
           transform=ax[0].transAxes, fontsize=8, color=".2", va="center",
           bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
plt.suptitle(f"True benefit {TRUE:.1f} pp — higher Type-B floors reach/exceed the signal itself",
             fontsize=10, y=1.02)
plt.tight_layout(); plt.savefig("fig_typeB_levels.png", dpi=130); plt.close()

print(f"\n{'L':>5} " + " ".join(f"sB={s*100:.2g}%".rjust(8) for s in levels) + "   (bootstrap coverage)")
for li, L in enumerate(Ls):
    print(f"{L:>4}y " + " ".join(f"{cov_boot[s][li]:7.0f}%" for s in levels))
print("\nWrote fig_typeB_levels.png")
