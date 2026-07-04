"""
Visual story (4 figures), one consistent scenario, rebuilt per the adversarial
review: realistic direction-signed, speed-tapered yaw controller at the FIXED
PyWake-default wake coefficient (no tuning), IEA dAEP metric, end-to-end error
injection, length-independent estimand, shared B=1000 bootstrap.

  story_1_rawdata.png     what a toggle test records (benefit hidden in weather)
  story_2_binning.png     binning -> per-bin power gain -> aggregate dAEP
  story_3_uncertainty.png the result's two uncertainties (Type A shrinks, Type B floor)
  story_4_conclusion.png  bootstrap coverage falls with campaign length; honest holds
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import campaigns as C
import metrics as M

Z = 1.959964
EXP_ID = 4
SIG_B = 0.005                     # showcased Type-B level: 0.5% ON-only
ON_C, OFF_C, A_C, B_C, OK_C = "#ff7f0e", "#1f77b4", "#1f77b4", "#d62728", "#2ca02c"

TRUE = C.TRUE_DAEP
dDdb = 100.0 + TRUE
print(f"true dAEP = {TRUE:+.2f}%  (fixed k, tapered controller)   dDdb = {dDdb:.1f}")

# one representative 1-yr campaign with a visible (+1 sigma) systematic
rng0 = C.rng_for(EXP_ID, 0, 0)
on0 = ((np.arange(C.N_S) // C.TOGGLE_LEN) % 2 == 1)
po_obs = C.PON * (1 + SIG_B) * (1 + C.SIG_A * rng0.standard_normal(C.N_S))
pf_obs = C.POFF * (1 + C.SIG_A * rng0.standard_normal(C.N_S))
p_obs = np.where(on0, po_obs, pf_obs)

# ============================ FIG 1: raw data ===============================
n = 120; sl = slice(0, n); xi = np.arange(n); on_w = on0[sl]
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(xi, C.WS_S[sl], "-", color=".4", lw=1.2); ax[0].set_ylabel("wind speed [m/s]")
ax[0].set_title("1 · Raw toggle-test data — the benefit is buried in weather", fontsize=12, loc="left")
b = 0
while b < n:
    if on_w[b]:
        for a in ax:
            a.axvspan(b - 0.5, min(b + C.TOGGLE_LEN, n) - 0.5, color=ON_C, alpha=0.10)
    b += C.TOGGLE_LEN
ax[1].scatter(xi[on_w], p_obs[sl][on_w] / 1e3, s=18, color=ON_C, label="control ON (steering)")
ax[1].scatter(xi[~on_w], p_obs[sl][~on_w] / 1e3, s=18, color=OFF_C, label="baseline OFF")
ax[1].set_ylabel("measured pair power [MW]"); ax[1].set_xlabel("10-min sample (waked sector)")
ax[1].legend(loc="upper right", fontsize=9)
ax[1].text(0.01, 0.04, "the toggle flips every 7 waked-sector samples (hours to days of calendar\n"
           "time); power swings are dominated by wind, not control.",
           transform=ax[1].transAxes, fontsize=8.5, color=".3", va="bottom")
plt.tight_layout(); plt.savefig("story_1_rawdata.png", dpi=130); plt.close()

# ============================ FIG 2: binning ================================
fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]})
sub = rng0.choice(C.N_S, 700, replace=False)
ax[0].scatter(C.WS_S[sub][on0[sub]], po_obs[sub][on0[sub]] / 1e3, s=8, alpha=0.35, color=ON_C, label="ON")
ax[0].scatter(C.WS_S[sub][~on0[sub]], pf_obs[sub][~on0[sub]] / 1e3, s=8, alpha=0.35, color=OFF_C, label="OFF")
centers, mon, mof, gain = [], [], [], []
ds = np.digitize(C.WS_S, M.WS_EDGES)
for bb in np.unique(ds):
    m = ds == bb; mo = m & on0; mf = m & ~on0
    if mo.sum() >= 3 and mf.sum() >= 3:
        centers.append(C.WS_S[m].mean()); mon.append(po_obs[mo].mean() / 1e3)
        mof.append(pf_obs[mf].mean() / 1e3); gain.append((po_obs[mo].mean() - pf_obs[mf].mean()))
centers = np.array(centers); gain = np.array(gain)
ax[0].plot(centers, mon, "-o", color=ON_C, ms=4, lw=1.5, label="ON bin mean")
ax[0].plot(centers, mof, "-s", color=OFF_C, ms=4, lw=1.5, label="OFF bin mean")
for e in M.WS_EDGES:
    ax[0].axvline(e, color=".85", lw=0.6, zorder=0)
ax[0].set_xlabel("wind speed [m/s]"); ax[0].set_ylabel("pair power [MW]")
ax[0].set_title("2 · Bin by wind speed & direction, average ON vs OFF per bin", fontsize=11, loc="left")
ax[0].legend(fontsize=8, loc="lower right")
ax[1].axhline(0, color="k", lw=0.6)
ax[1].bar(centers, gain, width=0.8, color=OK_C, alpha=0.8)
ax[1].set_xlabel("wind speed [m/s]"); ax[1].set_ylabel("per-bin power gain [kW]")
bid0 = M.bin_index(C.WD_S, C.WS_S)
_, agg, _ = M.campaign_estimates(np.where(on0, C.PON, C.POFF), p_obs, on0, bid0, fixed=C.FIXED_MASK)
ax[1].set_title("per-bin gain  →  IEA change-in-energy ΔAEP", fontsize=11, loc="left")
ax[1].text(0.5, 0.97, f"ΔAEP (IEA Task 44) = {agg:+.2f}% of baseline\n(true value {TRUE:+.2f}%; "
           f"gap = the systematic)", transform=ax[1].transAxes, ha="center", va="top",
           fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
plt.tight_layout(); plt.savefig("story_2_binning.png", dpi=130); plt.close()

# ===== per-length Type-A scatter + bootstrap (noise-only, sigma_B=0) ========
Ls = [0.5, 1, 2, 4, 8]; R = 300
errs = np.empty((len(Ls), R)); hws = np.empty((len(Ls), R))
errsB = np.empty((len(Ls), R))                     # with the 0.5% systematic, end-to-end
for Li, L in enumerate(Ls):
    for r in range(R):
        rng = C.rng_for(EXP_ID, 10 + Li, r)
        idx, blocks, on = C.make_campaign(L, rng)
        pc, po, bid, _ = C.observe(idx, on, rng, sigma_b=0.0)
        _, est, _ = M.campaign_estimates(pc, po, on, bid, fixed=C.FIXED_MASK)
        hws[Li, r] = M.block_boot_halfwidth(po, on, bid, blocks, n_boot=1000,
                                            rng=rng, fixed=C.FIXED_MASK)
        errs[Li, r] = est - TRUE
        rngB = C.rng_for(EXP_ID, 40 + Li, r)
        idxB, blocksB, onB = C.make_campaign(L, rngB)
        pcB, poB, bidB, _ = C.observe(idxB, onB, rngB, sigma_b=SIG_B)
        _, estB, _ = M.campaign_estimates(pcB, poB, onB, bidB, fixed=C.FIXED_MASK)
        errsB[Li, r] = estB - TRUE
hwA = hws.mean(axis=1); hwB = Z * dDdb * SIG_B
x = np.array(Ls)

# ============================ FIG 3: uncertainty ============================
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
est = TRUE + errsB[Ls.index(4), 0]; hb = hws[Ls.index(4), 0]; hh = np.hypot(hb, hwB)
ax[0].axvline(0, color=".6", lw=1, ls=":")
ax[0].axvline(TRUE, color="k", lw=1.5, ls="--", label=f"true ΔAEP {TRUE:+.2f}%")
ax[0].errorbar(est, 1.0, xerr=hb, fmt="o", color=A_C, capsize=5, ms=8, lw=2,
               label=f"bootstrap 95% (Type A): ±{hb:.2f}")
ax[0].errorbar(est, 0.6, xerr=hh, fmt="s", color=B_C, capsize=5, ms=8, lw=2,
               label=f"honest 95% (⊕ Type B): ±{hh:.2f}")
ax[0].set_ylim(0.2, 1.6); ax[0].set_yticks([]); ax[0].set_xlabel("ΔAEP [%]")
ax[0].set_title("3 · One 4-year result, two interval widths", fontsize=11, loc="left")
ax[0].legend(fontsize=8.5, loc="upper left")
ax[1].plot(x, hwA, "o-", color=A_C, label="Type A (bootstrap) — shrinks ∝ 1/√N")
ax[1].axhline(hwB, ls="--", color=B_C, lw=2, label=f"Type B floor = {hwB:.2f} (fixed)")
ax[1].plot(x, np.hypot(hwA, hwB), "s-", color="k", label="total (honest)")
ax[1].set_xscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels([f"{l:g}y" for l in Ls])
ax[1].set_ylim(bottom=0); ax[1].set_xlabel("campaign length"); ax[1].set_ylabel("95% half-width [pp]")
ax[1].set_title("Type A vanishes with data; Type B does not", fontsize=11, loc="left")
ax[1].legend(fontsize=8.5)
plt.tight_layout(); plt.savefig("story_3_uncertainty.png", dpi=130); plt.close()

# ============================ FIG 4: conclusion =============================
fig, ax = plt.subplots(figsize=(8.6, 5.6))
cov0 = 100 * (np.abs(errs) <= hws).mean(axis=1)
covB = 100 * (np.abs(errsB) <= hws).mean(axis=1)          # bootstrap-only, with systematic
covH = 100 * (np.abs(errsB) <= np.hypot(hws, hwB)).mean(axis=1)
for cov, col, lab, mk in [(cov0, OK_C, "bootstrap, no Type B  [control]", "o-"),
                          (covB, B_C, f"bootstrap, $\\sigma_B$={SIG_B*100:.1f}% ON-only", "o-"),
                          (covH, "k", f"honest (⊕ Type B), $\\sigma_B$={SIG_B*100:.1f}%", "s--")]:
    k = np.round(cov * R / 100).astype(int)
    lo = np.array([M.wilson(kk, R)[0] for kk in k]); hi = np.array([M.wilson(kk, R)[1] for kk in k])
    ax.plot(x, cov, mk, color=col, label=lab)
    ax.fill_between(x, lo, hi, color=col, alpha=0.10)
ax.axhline(95, ls=":", color="k", lw=1.4, label="nominal 95%")
ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{l:g}y" for l in Ls])
ax.set_ylim(30, 102); ax.set_xlabel("campaign length")
ax.set_ylabel("coverage of the true ΔAEP [%]")
ax.set_title("4 · More measurement → more false confidence\n"
             "with a differential systematic, bootstrap coverage falls as data grow;\n"
             "the control (no Type B) confirms the bootstrap itself is calibrated",
             fontsize=10.5, loc="left")
ax.legend(fontsize=8.5, loc="lower left"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("story_4_conclusion.png", dpi=130); plt.close()
print("Wrote story_1_rawdata.png .. story_4_conclusion.png")
