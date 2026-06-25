"""
Un-normalized (absolute) view of the same experiment.

Reporting the benefit as a percent of baseline power (gain / P_ref) collapses
everything into one number and hides two processes:
  1. Where the benefit lives across wind speed (it peaks in Region II and
     vanishes near rated, where both states sit at rated power).
  2. Where the systematic Type-B error lives. The yaw-correlated bias is
     b * P_on, so in absolute terms it GROWS with power and is largest in the
     high-wind bins -- exactly where the wake effect (and the real benefit) is
     gone. The percent normalization flattens this.

This script reports gain and uncertainties in absolute kW / MWh and shows the
per-wind-speed decomposition and the raw campaign-to-campaign estimate spread.
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
K_FIX, SIG_A, SIG_B, Z = 0.034, 0.02, 0.005, 1.959964
WSB = np.arange(4, 25, 1.0); NB = len(WSB) + 2
HRS_SECTOR = len(idx) / len(WD) * 8766.0                 # waked-sector hours / yr


def fp(w, s, c):
    w = np.atleast_1d(w); s = np.clip(np.atleast_1d(s), 3, 25)
    return (F_ON if c else F_OFF)(np.column_stack([w, s, np.full(np.shape(w), K_FIX)]))


PON1, POFF1 = fp(wd_s, ws_s, 1), fp(wd_s, ws_s, 0)
on0 = ((np.arange(len(wd_s)) // 7) % 2 == 1); dg0 = np.digitize(ws_s, WSB)


def gain_abs(p_on, p_off, on, dg):
    """ws-bin-controlled mean farm-power gain in kW (NOT normalized)."""
    p = np.where(on, p_on, p_off); off = ~on
    so = np.bincount(dg[on], p[on], NB); no = np.bincount(dg[on], None, NB)
    sf = np.bincount(dg[off], p[off], NB); nf = np.bincount(dg[off], None, NB)
    v = (no >= 3) & (nf >= 3)
    mo = so / np.where(no > 0, no, 1); mf = sf / np.where(nf > 0, nf, 1)
    w = (no + nf) * v; sw = w.sum()
    return (w * (mo - mf)).sum() / sw if sw > 0 else np.nan   # kW


TRUE_kW = gain_abs(PON1, POFF1, on0, dg0)
PREF_kW = POFF1.mean()
print(f"true mean gain = {TRUE_kW:.1f} kW   ( = {100*TRUE_kW/PREF_kW:+.2f}% of {PREF_kW:.0f} kW baseline )")
print(f"sector annual energy gain = {TRUE_kW * HRS_SECTOR:.1f} kWh/yr "
      f"({TRUE_kW * HRS_SECTOR/1e3:.2f} MWh/yr) over {HRS_SECTOR:.0f} waked hours/yr")
print(f"Type-B 1sigma (absolute) = {SIG_B*PREF_kW:.1f} kW  vs true gain {TRUE_kW:.1f} kW")

# ---- per-wind-speed-bin decomposition (the hidden process) ------------------
centers, gain_bin, pon_bin, poff_bin, tb_bin, occ = [], [], [], [], [], []
dgc = np.digitize(ws_s, WSB)
for b in range(1, NB - 1):
    m = dgc == b
    if m.sum() < 5:
        continue
    on_m = m & on0; off_m = m & ~on0
    if on_m.sum() < 3 or off_m.sum() < 3:
        continue
    centers.append(ws_s[m].mean())
    pon_b = PON1[m].mean(); poff_b = POFF1[m].mean()
    pon_bin.append(pon_b); poff_bin.append(poff_b)
    gain_bin.append(pon_b - poff_b)
    tb_bin.append(SIG_B * pon_b)                          # absolute Type-B 1sigma in this bin
    occ.append(m.sum())
centers = np.array(centers); gain_bin = np.array(gain_bin)
pon_bin = np.array(pon_bin); poff_bin = np.array(poff_bin)
tb_bin = np.array(tb_bin); occ = np.array(occ)

# ---- raw campaign-to-campaign spread at 8 yr (un-normalized coverage) --------
nyr = 8
pon = np.tile(PON1, nyr); poff = np.tile(POFF1, nyr); dg = np.tile(dg0, nyr)
on = ((np.arange(len(pon)) // 7) % 2 == 1)
t = np.concatenate([t_s + m * TY for m in range(nyr)])
R = 500
ests = np.empty(R)
for r in range(R):
    rng = np.random.default_rng(9000 + r)
    b = rng.normal(0, SIG_B)
    po = pon * (1 + b) * (1 + SIG_A * rng.standard_normal(len(pon)))
    pf = poff * (1 + SIG_A * rng.standard_normal(len(poff)))
    ests[r] = gain_abs(po, pf, on, dg)


def boot_kW(p_on, p_off, on, dg, t, nb=400, rng=None):
    blk = (t / 2).astype(int); ub = np.unique(blk)
    mem = [np.where(blk == b)[0] for b in ub]; nbk = len(ub)
    p = np.where(on, p_on, p_off); est = np.empty(nb)
    for i in range(nb):
        sel = np.concatenate([mem[j] for j in rng.integers(0, nbk, nbk)])
        off = ~on[sel]; ons = on[sel]; ps = p[sel]; dd = dg[sel]
        so = np.bincount(dd[ons], ps[ons], NB); no = np.bincount(dd[ons], None, NB)
        sf = np.bincount(dd[off], ps[off], NB); nf = np.bincount(dd[off], None, NB)
        v = (no >= 3) & (nf >= 3)
        mo = so / np.where(no > 0, no, 1); mf = sf / np.where(nf > 0, nf, 1)
        w = (no + nf) * v; sw = w.sum()
        est[i] = (w * (mo - mf)).sum() / sw if sw > 0 else np.nan
    return est


rng = np.random.default_rng(1)
b1 = rng.normal(0, SIG_B)
po1 = pon * (1 + b1) * (1 + SIG_A * rng.standard_normal(len(pon)))
pf1 = poff * (1 + SIG_A * rng.standard_normal(len(poff)))
bdist = boot_kW(po1, pf1, on, dg, t, nb=400, rng=rng)
b_lo, b_hi = np.percentile(bdist, [2.5, 97.5])
est1 = gain_abs(po1, pf1, on, dg)

# ---- figure -----------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# (a) per-ws-bin decomposition in kW
ax2 = ax[0].twinx()
ax2.plot(centers, poff_bin, color=".6", lw=1.3, ls="-", label="baseline power (off)")
ax2.plot(centers, pon_bin, color=".6", lw=1.3, ls=":", label="control power (on)")
ax2.set_ylabel("turbine-pair power [kW]", color=".4")
ax[0].axhline(0, color="k", lw=0.6)
ax[0].bar(centers, gain_bin, width=0.8, color="#2ca02c", alpha=0.8, label="true gain (on−off)")
ax[0].errorbar(centers, gain_bin, yerr=Z * tb_bin, fmt="none", ecolor="#d62728",
               elinewidth=1.4, capsize=2, label="Type-B 95% (= $b\\,P_{on}$)")
ax[0].set_xlabel("wind speed [m/s]"); ax[0].set_ylabel("power gain [kW]")
ax[0].set_title("Per-wind-speed (absolute): the gain peaks in Region II and dies near\n"
                "rated; the systematic Type-B error grows with power", fontsize=9.5)
h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax[0].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

# (b) raw campaign-to-campaign spread vs a single bootstrap CI (8 yr)
ax[1].hist(ests, bins=40, density=True, color="#bbbbbb", alpha=0.85,
           label=f"actual spread of $\\hat\\Delta$ over campaigns\n(8 yr, $\\sigma_B$=0.5%)")
ax[1].axvline(TRUE_kW, color="k", lw=2, label=f"true gain = {TRUE_kW:.1f} kW")
ax[1].axvspan(b_lo, b_hi, color="#1f77b4", alpha=0.3,
              label=f"one campaign's bootstrap 95% CI\n[{b_lo:.1f}, {b_hi:.1f}] kW")
ax[1].axvline(est1, color="#1f77b4", lw=1.5)
ax[1].set_xlabel("estimated mean power gain [kW]"); ax[1].set_ylabel("density")
ax[1].set_title("Un-normalized coverage: the bootstrap CI (blue) is far narrower\n"
                "than how much the estimate actually moves between campaigns (grey)",
                fontsize=9.5)
ax[1].legend(fontsize=8, loc="upper right")
plt.tight_layout(); plt.savefig("fig_absolute_view.png", dpi=130); plt.close()
print(f"\nactual std of estimate over campaigns = {ests.std():.2f} kW   "
      f"one bootstrap half-width = {(b_hi-b_lo)/2:.2f} kW")
print("Wrote fig_absolute_view.png")
