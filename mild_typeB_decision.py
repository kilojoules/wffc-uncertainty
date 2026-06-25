"""
Different CONCLUSIONS under mild Type B.

Decision: does the 95% interval exclude zero -> "WFFC gives a statistically
significant benefit"?  True benefit is set ~0 (marginal controller, the realistic
case). Each campaign draws DIFFERENT weather (2-day block resample of the base
year) so the across-campaign spread matches what the bootstrap estimates -- the
Type-A significance test is then calibrated (~5% false positive at sigma_B = 0).

  * bootstrap only (Type A): false-positive "significant benefit" rate climbs
    above 5% and worsens with campaign length -- a mild systematic, invisible to
    resampling, dominates the shrinking CI.
  * honest (bootstrap (+) propagated Type B): stays ~5%.

Same data, opposite go/no-go conclusions, for systematics as small as 0.1-0.25%.
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
S = (WD >= 266) & (WD <= 274) & (WS >= 3) & (WS <= 25); idx0 = np.where(S)[0]
wd_s, ws_s, t_s = WD[idx0], WS[idx0], idx0 * DT
SIG_A, Z = 0.02, 1.959964
WSB = np.arange(4, 25, 1.0); NB = len(WSB) + 2


def fp(w, s, k, c):
    w = np.atleast_1d(w); s = np.clip(np.atleast_1d(s), 3, 25)
    return (F_ON if c else F_OFF)(np.column_stack([w, s, np.full(np.shape(w), k)]))


def gain_pct(po, pf, on, dg, pref):
    p = np.where(on, po, pf); off = ~on
    so = np.bincount(dg[on], p[on], NB); no = np.bincount(dg[on], None, NB)
    sf = np.bincount(dg[off], p[off], NB); nf = np.bincount(dg[off], None, NB)
    v = (no >= 3) & (nf >= 3)
    mo = so / np.where(no > 0, no, 1); mf = sf / np.where(nf > 0, nf, 1)
    w = (no + nf) * v; sw = w.sum()
    return 100 * (w * (mo - mf)).sum() / sw / pref if sw > 0 else np.nan


# k0 so the climatological benefit is ~0 (marginal WFFC)
on_full = ((np.arange(len(wd_s)) // 7) % 2 == 1); dg_full = np.digitize(ws_s, WSB)
kc = np.linspace(K_GRID[0], K_GRID[-1], 80)
bcv = np.array([100 * (fp(wd_s, ws_s, k, 1).sum() - fp(wd_s, ws_s, k, 0).sum())
                / fp(wd_s, ws_s, k, 0).sum() for k in kc])
K0 = float(np.interp(0.0, bcv[::-1], kc[::-1]))
POdet, PFdet = fp(wd_s, ws_s, K0, 1), fp(wd_s, ws_s, K0, 0)
TRUE = gain_pct(POdet, PFdet, on_full, dg_full, PFdet.mean())
eps = 1e-3
dDdb = (gain_pct(POdet * (1 + eps), PFdet, on_full, dg_full, PFdet.mean()) - TRUE) / eps
print(f"k0={K0:.4f}  climatological benefit = {TRUE:+.3f}%  (marginal)  dBenefit/db={dDdb:.1f}")

# 2-day blocks of the base year (the resampling unit for weather)
blk_id = (t_s / 2).astype(int)
blocks = [np.where(blk_id == b)[0] for b in np.unique(blk_id)]
NBLK = len(blocks)


def boot_hw(po, pf, on, dg, camp_blocks, pref0, nb=150, rng=None):
    """block bootstrap over the campaign's own resampled blocks."""
    ub = np.unique(camp_blocks); mem = [np.where(camp_blocks == b)[0] for b in ub]
    nbk = len(ub); est = np.empty(nb)
    p = np.where(on, po, pf)
    for i in range(nb):
        sel = np.concatenate([mem[j] for j in rng.integers(0, nbk, nbk)])
        off = ~on[sel]; ons = on[sel]; ps = p[sel]; dd = dg[sel]
        so = np.bincount(dd[ons], ps[ons], NB); no = np.bincount(dd[ons], None, NB)
        sf = np.bincount(dd[off], ps[off], NB); nf = np.bincount(dd[off], None, NB)
        v = (no >= 3) & (nf >= 3)
        mo = so / np.where(no > 0, no, 1); mf = sf / np.where(nf > 0, nf, 1)
        w = (no + nf) * v; sw = w.sum()
        est[i] = 100 * (w * (mo - mf)).sum() / sw / pref0 if sw > 0 else np.nan
    lo, hi = np.nanpercentile(est, [2.5, 97.5]); return (hi - lo) / 2


Ls = np.array([0.5, 1, 2, 4, 8])
R = 600
dsamp = {}; hwb = {}
for L in Ls:
    nblk = max(2, round(L * NBLK))
    ds = np.empty(R); hh = np.empty(R)
    for r in range(R):
        rng = np.random.default_rng(12000 + r + int(L * 7))
        pick = rng.integers(0, NBLK, nblk)
        members = [blocks[p] for p in pick]
        idx = np.concatenate(members)
        camp_blocks = np.repeat(np.arange(nblk), [len(m) for m in members])
        on = ((np.arange(len(idx)) // 7) % 2 == 1)
        po = POdet[idx] * (1 + SIG_A * rng.standard_normal(len(idx)))      # noise only
        pf = PFdet[idx] * (1 + SIG_A * rng.standard_normal(len(idx)))
        dg = np.digitize(ws_s[idx], WSB); pref0 = pf[~on].mean()
        ds[r] = gain_pct(po, pf, on, dg, pref0) - TRUE
        hh[r] = boot_hw(po, pf, on, dg, camp_blocks, pref0, nb=150, rng=rng)
    dsamp[L] = ds; hwb[L] = hh
    print(f"  L={L:>4}y  std(estimate)={ds.std():.3f}pp  mean hw_boot={hh.mean():.3f}pp  (calibrated if ~2x)")

# sweep MILD Type-B levels: false 'significant benefit' rate (true ~ 0)
levels = [0.0, 0.001, 0.0025, 0.005]            # 0, 0.1%, 0.25%, 0.5% (mild)
zr = np.random.default_rng(321).standard_normal((len(Ls), R))
fpr_b = {s: [] for s in levels}; fpr_h = {s: [] for s in levels}
for s in levels:
    tb = Z * abs(dDdb) * s
    for li, L in enumerate(Ls):
        est = TRUE + dsamp[L] + dDdb * s * zr[li]
        fpr_b[s].append(100 * np.mean(np.abs(est) > hwb[L]))
        fpr_h[s].append(100 * np.mean(np.abs(est) > np.hypot(hwb[L], tb)))

print(f"\nFalse 'significant benefit' rate (true benefit ~ 0):")
print(f"{'L':>5} " + " ".join(f"B@{s*100:.2g}%".rjust(7) for s in levels)
      + "  | " + " ".join(f"H@{s*100:.2g}%".rjust(7) for s in levels))
for li, L in enumerate(Ls):
    print(f"{L:>4}y " + " ".join(f"{fpr_b[s][li]:6.0f}%" for s in levels)
          + "  | " + " ".join(f"{fpr_h[s][li]:6.0f}%" for s in levels))

# concrete example: a 4-yr campaign at mild sigma_B = 0.25% where they disagree
li4 = int(np.where(Ls == 4)[0][0]); s_ex = 0.0025; tb_ex = Z * abs(dDdb) * s_ex
for r in range(R):
    est = TRUE + dsamp[4][r] + dDdb * s_ex * zr[li4][r]; hb = hwb[4][r]
    hh_ = np.hypot(hb, tb_ex)
    if abs(est) > hb and abs(est) <= hh_:
        print(f"\nExample (4 yr, sigma_B=0.25%):  measured benefit {est:+.2f}%")
        print(f"  bootstrap 95% CI [{est-hb:+.2f}, {est+hb:+.2f}] %  -> excludes 0 -> SIGNIFICANT, deploy")
        print(f"  honest   95% CI [{est-hh_:+.2f}, {est+hh_:+.2f}] %  -> includes 0 -> not significant")
        break

fig, ax = plt.subplots(figsize=(8.5, 5.5))
cols = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728"]
for s, c in zip(levels, cols):
    lab = "control ($\\sigma_B$=0)" if s == 0 else f"$\\sigma_B$={s*100:.2g}%"
    ax.plot(Ls, fpr_b[s], "o-", color=c, label=f"bootstrap, {lab}")
    if s > 0:
        ax.plot(Ls, fpr_h[s], "s--", color=c, alpha=0.55)
ax.axhline(5, ls=":", color="k", lw=1.4, label="nominal 5%")
ax.set_xscale("log"); ax.set_xticks(Ls); ax.set_xticklabels([f"{l:g}y" for l in Ls])
ax.set_xlabel("campaign length"); ax.set_ylabel("'significant benefit' false-positive rate [%]")
ax.set_title("Different conclusions under MILD Type B\n"
             "solid = bootstrap (false positives climb); dashed = honest (stays ~5%)",
             fontsize=11)
ax.legend(fontsize=8, ncol=2, loc="upper left"); ax.grid(alpha=0.3)
ax.text(0.97, 0.45, "WFFC actually does ~nothing here.\nbootstrap increasingly 'finds' a\n"
        "significant benefit; honest does not.", transform=ax.transAxes, ha="right",
        fontsize=8.5, color=".25", bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
plt.tight_layout(); plt.savefig("fig_mild_typeB_decision.png", dpi=130); plt.close()
print("\nWrote fig_mild_typeB_decision.png")
