"""
Visual story (4 figures), one consistent scenario:
  story_1_rawdata.png    -- what a toggle test records (benefit hidden in weather)
  story_2_binning.png    -- bin by wind speed -> per-bin power gain -> aggregate benefit
  story_3_uncertainty.png-- the result and its two uncertainties (Type A shrinks, Type B floor)
  story_4_conclusion.png -- more measurement -> more false confidence; honest interval holds
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
SIG_A, SIG_B, Z = 0.02, 0.005, 1.959964      # Type A 2% noise, Type B 0.5% systematic
WSB = np.arange(4, 25, 1.0); NB = len(WSB) + 2
ON_C, OFF_C, A_C, B_C, OK_C = "#ff7f0e", "#1f77b4", "#1f77b4", "#d62728", "#2ca02c"


def fp(w, s, k, c):
    w = np.atleast_1d(w); s = np.clip(np.atleast_1d(s), 3, 25)
    return (F_ON if c else F_OFF)(np.column_stack([w, s, np.full(np.shape(w), k)]))


def benefit(po_obs, pf_obs, on, dg, pref):
    p = np.where(on, po_obs, pf_obs); off = ~on
    so = np.bincount(dg[on], p[on], NB); no = np.bincount(dg[on], None, NB)
    sf = np.bincount(dg[off], p[off], NB); nf = np.bincount(dg[off], None, NB)
    v = (no >= 3) & (nf >= 3)
    mo = so / np.where(no > 0, no, 1); mf = sf / np.where(nf > 0, nf, 1)
    w = (no + nf) * v; sw = w.sum()
    return 100 * (w * (mo - mf)).sum() / sw / pref if sw > 0 else np.nan


# scenario: pick k for a clearly positive true benefit (~+1.5%)
kc = np.linspace(K_GRID[0], K_GRID[-1], 80)
bcv = np.array([100 * (fp(wd_s, ws_s, k, 1).sum() - fp(wd_s, ws_s, k, 0).sum())
                / fp(wd_s, ws_s, k, 0).sum() for k in kc])
K0 = float(np.interp(1.5, bcv[::-1], kc[::-1]))
PON, POFF = fp(wd_s, ws_s, K0, 1), fp(wd_s, ws_s, K0, 0)
on_full = ((np.arange(len(wd_s)) // 7) % 2 == 1); dg_full = np.digitize(ws_s, WSB)
TRUE = benefit(PON, POFF, on_full, dg_full, POFF.mean())
eps = 1e-3
dDdb = (benefit(PON * (1 + eps), POFF, on_full, dg_full, POFF.mean()) - TRUE) / eps
print(f"k0={K0:.4f}  true benefit = {TRUE:+.2f}%  dBenefit/db={dDdb:.0f}")

# one representative measured campaign (1 yr): noise + a systematic bias drawn once
rng = np.random.default_rng(3)
b_sys = SIG_B                                   # use +1 sigma so the bias is visible
po_obs = PON * (1 + b_sys) * (1 + SIG_A * rng.standard_normal(len(PON)))
pf_obs = POFF * (1 + SIG_A * rng.standard_normal(len(POFF)))
pref = pf_obs[~on_full].mean()

# ============================ FIG 1: raw data ===============================
n = 120; sl = slice(0, n)                       # first ~120 sector samples (a few weeks)
xi = np.arange(n)
on_w = on_full[sl]
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(xi, ws_s[sl], "-", color=".4", lw=1.2)
ax[0].set_ylabel("wind speed [m/s]")
ax[0].set_title("1 · Raw toggle-test data — the benefit is buried in weather", fontsize=12, loc="left")
# toggle shading
b = 0
while b < n:
    if on_w[b]:
        ax[0].axvspan(b - 0.5, min(b + 7, n) - 0.5, color=ON_C, alpha=0.10)
        ax[1].axvspan(b - 0.5, min(b + 7, n) - 0.5, color=ON_C, alpha=0.10)
    b += 7
pf_full_obs = np.where(on_full, po_obs, pf_obs)
ax[1].scatter(xi[on_w], pf_full_obs[sl][on_w] / 1e3, s=18, color=ON_C, label="control ON (yawed)")
ax[1].scatter(xi[~on_w], pf_full_obs[sl][~on_w] / 1e3, s=18, color=OFF_C, label="baseline OFF")
ax[1].set_ylabel("measured pair power [MW]"); ax[1].set_xlabel("10-min sample (waked sector)")
ax[1].legend(loc="upper right", fontsize=9)
ax[1].text(0.01, 0.04, "ON/OFF alternates every 70 min; power swings are dominated by wind,\n"
           "not by control — you cannot read the benefit off the raw signal.",
           transform=ax[1].transAxes, fontsize=8.5, color=".3", va="bottom")
plt.tight_layout(); plt.savefig("story_1_rawdata.png", dpi=130); plt.close()

# ============================ FIG 2: binning ================================
fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]})
# (a) scatter power vs ws, ON/OFF, with bin edges and per-bin means
sub = rng.choice(len(wd_s), 700, replace=False)
ax[0].scatter(ws_s[sub][on_full[sub]], po_obs[sub][on_full[sub]] / 1e3, s=8, alpha=0.35,
              color=ON_C, label="ON (control)")
ax[0].scatter(ws_s[sub][~on_full[sub]], pf_obs[sub][~on_full[sub]] / 1e3, s=8, alpha=0.35,
              color=OFF_C, label="OFF (baseline)")
centers, mon, mof, gain, w_occ = [], [], [], [], []
for bb in range(1, NB - 1):
    m = dg_full == bb
    mo = m & on_full; mf = m & ~on_full
    if mo.sum() >= 3 and mf.sum() >= 3:
        centers.append(ws_s[m].mean())
        mon.append(po_obs[mo].mean() / 1e3); mof.append(pf_obs[mf].mean() / 1e3)
        gain.append((po_obs[mo].mean() - pf_obs[mf].mean()) / 1e3); w_occ.append(m.sum())
centers = np.array(centers); mon = np.array(mon); mof = np.array(mof)
gain = np.array(gain); w_occ = np.array(w_occ)
ax[0].plot(centers, mon, "-o", color=ON_C, ms=4, lw=1.5, label="ON bin mean")
ax[0].plot(centers, mof, "-s", color=OFF_C, ms=4, lw=1.5, label="OFF bin mean")
for e in WSB:
    ax[0].axvline(e, color=".85", lw=0.6, zorder=0)
ax[0].set_xlabel("wind speed [m/s]"); ax[0].set_ylabel("pair power [MW]")
ax[0].set_title("2 · Bin by wind speed, average ON vs OFF in each bin", fontsize=11, loc="left")
ax[0].legend(fontsize=8, loc="lower right")
# (b) per-bin gain -> aggregate
ax[1].axhline(0, color="k", lw=0.6)
ax[1].bar(centers, gain * 1e3, width=0.8, color=OK_C, alpha=0.8)
ax[1].set_xlabel("wind speed [m/s]"); ax[1].set_ylabel("per-bin power gain [kW]")
agg = 100 * (w_occ * (mon - mof)).sum() / w_occ.sum() / (POFF.mean() / 1e3)
ax[1].set_title(f"per-bin gain  →  occurrence-weighted benefit = {agg:+.2f}%", fontsize=11, loc="left")
ax[1].text(0.5, 0.97, f"one number: {agg:+.2f}% of baseline\n(true value {TRUE:+.2f}%)",
           transform=ax[1].transAxes, ha="center", va="top", fontsize=9,
           bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
plt.tight_layout(); plt.savefig("story_2_binning.png", dpi=130); plt.close()

# ===== per-length Type-A (bootstrap) via block-resampled campaigns ==========
blk_id = (t_s / 2).astype(int); blocks = [np.where(blk_id == b)[0] for b in np.unique(blk_id)]
NBLK = len(blocks)


def boot_hw(po, pf, on, dg, camp_blk, pref0, nb=150, rng=None):
    ub = np.unique(camp_blk); mem = [np.where(camp_blk == b)[0] for b in ub]; nbk = len(ub)
    p = np.where(on, po, pf); est = np.empty(nb)
    for i in range(nb):
        sel = np.concatenate([mem[j] for j in rng.integers(0, nbk, nbk)])
        off = ~on[sel]; ons = on[sel]; ps = p[sel]; dd = dg[sel]
        so = np.bincount(dd[ons], ps[ons], NB); no = np.bincount(dd[ons], None, NB)
        sf = np.bincount(dd[off], ps[off], NB); nf = np.bincount(dd[off], None, NB)
        v = (no >= 3) & (nf >= 3)
        mo = so / np.where(no > 0, no, 1); mf = sf / np.where(nf > 0, nf, 1)
        w = (no + nf) * v; sw = w.sum()
        est[i] = 100 * (w * (mo - mf)).sum() / sw / pref0 if sw > 0 else np.nan
    return np.nanpercentile(est, [2.5, 97.5])


Ls = np.array([0.5, 1, 2, 4, 8]); R = 300
dsamp = {}; hwb = {}
for L in Ls:
    nblk = max(2, round(L * NBLK)); ds = np.empty(R); hh = np.empty(R)
    for r in range(R):
        rr = np.random.default_rng(20000 + r + int(L * 13))
        pick = rr.integers(0, NBLK, nblk); members = [blocks[p] for p in pick]
        idx = np.concatenate(members)
        cb = np.repeat(np.arange(nblk), [len(m) for m in members])
        on = ((np.arange(len(idx)) // 7) % 2 == 1)
        po = PON[idx] * (1 + SIG_A * rr.standard_normal(len(idx)))
        pf = POFF[idx] * (1 + SIG_A * rr.standard_normal(len(idx)))
        dg = np.digitize(ws_s[idx], WSB); pr = pf[~on].mean()
        ds[r] = benefit(po, pf, on, dg, pr) - TRUE
        lo, hi = boot_hw(po, pf, on, dg, cb, pr, nb=150, rng=rr); hh[r] = (hi - lo) / 2
    dsamp[L] = ds; hwb[L] = hh
hwA = np.array([hwb[L].mean() for L in Ls])
hwB = Z * abs(dDdb) * SIG_B

# ============================ FIG 3: uncertainty ============================
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# (a) one 4-yr result with the two intervals
L = 4; est = TRUE + dsamp[4][0]; hb = hwb[4][0]; hh = np.hypot(hb, hwB)
ax[0].axvline(0, color=".6", lw=1, ls=":")
ax[0].axvline(TRUE, color="k", lw=1.5, ls="--", label=f"true benefit {TRUE:+.2f}%")
ax[0].errorbar(est, 1.0, xerr=hb, fmt="o", color=A_C, capsize=5, ms=8, lw=2,
               label=f"bootstrap 95% (Type A): ±{hb:.2f}")
ax[0].errorbar(est, 0.6, xerr=hh, fmt="s", color=B_C, capsize=5, ms=8, lw=2,
               label=f"honest 95% (⊕ Type B): ±{hh:.2f}")
ax[0].set_ylim(0.2, 1.6); ax[0].set_yticks([]); ax[0].set_xlabel("benefit [%]")
ax[0].set_title("3 · One 4-year result, two interval widths", fontsize=11, loc="left")
ax[0].legend(fontsize=8.5, loc="upper right")
# (b) half-width vs campaign length
ax[1].plot(Ls, hwA, "o-", color=A_C, label="Type A (bootstrap) — shrinks ∝ 1/√N")
ax[1].axhline(hwB, ls="--", color=B_C, lw=2, label=f"Type B floor = {hwB:.2f} (fixed)")
ax[1].plot(Ls, np.hypot(hwA, hwB), "s-", color="k", label="total (honest)")
ax[1].set_xscale("log"); ax[1].set_xticks(Ls); ax[1].set_xticklabels([f"{l:g}y" for l in Ls])
ax[1].set_ylim(bottom=0); ax[1].set_xlabel("campaign length"); ax[1].set_ylabel("95% half-width [pp]")
ax[1].set_title("Type A vanishes with data; Type B does not", fontsize=11, loc="left")
ax[1].legend(fontsize=8.5)
plt.tight_layout(); plt.savefig("story_3_uncertainty.png", dpi=130); plt.close()

# ============================ FIG 4: conclusion =============================
zr = np.random.default_rng(7).standard_normal((len(Ls), R))
levels = [0.0, 0.0025, 0.005]; cols = [OK_C, "#9467bd", B_C]
fig, ax = plt.subplots(figsize=(8.6, 5.6))
for s, c in zip(levels, cols):
    tb = Z * abs(dDdb) * s
    cov_b = []; cov_h = []
    for li, L in enumerate(Ls):
        err = np.abs(dsamp[L] + dDdb * s * zr[li])     # |estimate - true|
        cov_b.append(100 * np.mean(err <= hwb[L]))
        cov_h.append(100 * np.mean(err <= np.hypot(hwb[L], tb)))
    lab = "no Type B" if s == 0 else f"$\\sigma_B$={s*100:.2g}%"
    ax.plot(Ls, cov_b, "o-", color=c, label=f"bootstrap, {lab}")
    if s > 0:
        ax.plot(Ls, cov_h, "s--", color=c, alpha=0.55)
ax.axhline(95, ls=":", color="k", lw=1.4, label="nominal 95%")
ax.set_xscale("log"); ax.set_xticks(Ls); ax.set_xticklabels([f"{l:g}y" for l in Ls])
ax.set_ylim(40, 102); ax.set_xlabel("campaign length")
ax.set_ylabel("coverage of the true benefit [%]")
ax.set_title("4 · More measurement → more false confidence\n"
             "solid = bootstrap (coverage falls as you measure more); dashed = honest (holds)",
             fontsize=11, loc="left")
ax.legend(fontsize=8.5, loc="lower left", ncol=2); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("story_4_conclusion.png", dpi=130); plt.close()
print("Wrote story_1_rawdata.png .. story_4_conclusion.png")
