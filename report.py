"""
Generate the 2-page PDF report:
  Page 1 - stochastic time series + assumed Type-B uncertainties
  Page 2 - epistemic mean + uncertainty of the k-parameter / benefit over
           different time windows (the windowed convergence view of
           Quick et al. 2025).

RAM < 300 MB, single CPU. PyWake only called once (via the lookup).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from pywake_model import build_lookup, K_GRID, GAMMA

RNG = np.random.default_rng(7)
F_OFF, F_ON, _, _ = build_lookup()

# ---- shared parameters (match main_experiment.py) ---------------------------
MU_K, SIGMA_B0 = 0.034, 0.004          # k prior: N(mu, sigma_B)  [GUM Type B]
SEAS_AMP = 0.008                       # in-campaign seasonal k swing [aleatoric]
P_NOISE = 0.01                         # sensor power noise [Type A]
TOGGLE_LEN = 7                         # 70-min ON/OFF blocks
WS_BINS = np.arange(3, 26, 1.0); NB = len(WS_BINS) + 2
Z95 = 1.959964
WSB = WS_BINS

# ---- inflow time series, aligned waked sector -------------------------------
from py_wake.examples.data import example_data_path
_d = np.load(example_data_path + "/time_series.npz")
WD_ALL, WS_ALL = _d["wd"], _d["ws"]
DT_DAYS = 1.0 / 6 / 24
T_YEAR = len(WD_ALL) * DT_DAYS
SEC = (WD_ALL >= 266) & (WD_ALL <= 274) & (WS_ALL >= 3) & (WS_ALL <= 25)
idx = np.where(SEC)[0]
wd_s, ws_s, t_s = WD_ALL[idx], WS_ALL[idx], idx * DT_DAYS
N_S = len(idx)


def farm_power(wd, ws, k, control):
    wd = np.atleast_1d(wd); ws = np.clip(np.atleast_1d(ws), 3.0, 25.0)
    k = np.broadcast_to(np.atleast_1d(k), wd.shape)
    return (F_ON if control else F_OFF)(np.column_stack([wd, ws, k]))


def true_benefit(wd, ws, k):
    po = farm_power(wd, ws, k, False); pn = farm_power(wd, ws, k, True)
    return 100.0 * (pn.sum() - po.sum()) / po.sum()


def seasonal_k(t, k_year, phase=0.0, ar_sigma=0.0015, rng=RNG):
    seas = SEAS_AMP * np.sin(2 * np.pi * t / T_YEAR + phase)
    nd = int(np.ceil(t.max())) + 2; w = np.zeros(nd)
    for i in range(1, nd):
        w[i] = 0.96 * w[i - 1] + ar_sigma * rng.standard_normal()
    return np.clip(k_year + seas + np.interp(t, np.arange(nd), w), K_GRID[0], K_GRID[-1])


def make_obs(wd, ws, k_t, rng, noise=P_NOISE):
    po = farm_power(wd, ws, k_t, False); pn = farm_power(wd, ws, k_t, True)
    on = ((np.arange(len(wd)) // TOGGLE_LEN) % 2 == 1)
    p = np.where(on, pn, po) * (1 + noise * rng.standard_normal(len(wd)))
    return p, on


def uplift(p, on, wr):
    dg = np.digitize(wr, WSB); off = ~on
    so = np.bincount(dg[on], p[on], NB); no = np.bincount(dg[on], None, NB)
    sf = np.bincount(dg[off], p[off], NB); nf = np.bincount(dg[off], None, NB)
    v = (no >= 2) & (nf >= 2)
    mo = so / np.where(no > 0, no, 1); mf = sf / np.where(nf > 0, nf, 1)
    w = (no + nf) * v; den = (w * mf).sum()
    return 100.0 * (w * (mo - mf)).sum() / den if den > 0 else np.nan


WSB_C = np.arange(4, 25, 1.0); NBC = len(WSB_C) + 2     # field-standard 1-m/s ws bins


def uplift_robust(p, on, wr, pref):
    """Benefit [%] = ws-bin-controlled mean power gain / fixed reference power.
    1-m/s bins (IEC 61400-12) control the ws confound tightly; the FIXED
    denominator `pref` (campaign-mean baseline power) keeps the bootstrap
    light-tailed instead of the heavy-tailed energy ratio (so even short
    windows stay stable despite the fine bins)."""
    dg = np.digitize(wr, WSB_C); off = ~on
    so = np.bincount(dg[on], p[on], NBC); no = np.bincount(dg[on], None, NBC)
    sf = np.bincount(dg[off], p[off], NBC); nf = np.bincount(dg[off], None, NBC)
    v = (no >= 3) & (nf >= 3)
    mo = so / np.where(no > 0, no, 1); mf = sf / np.where(nf > 0, nf, 1)
    w = (no + nf) * v; sw = w.sum()
    if sw == 0:
        return np.nan
    return 100.0 * (w * (mo - mf)).sum() / sw / pref   # weighted mean gain / Pref


def boot_hw(p, on, wr, t, pref, nb=400, rng=RNG, block_days=2.0):
    blk = (t / block_days).astype(int); ub = np.unique(blk)
    mem = [np.where(blk == b)[0] for b in ub]; nbk = len(ub)
    est = np.empty(nb)
    for i in range(nb):
        sel = np.concatenate([mem[j] for j in rng.integers(0, nbk, nbk)])
        est[i] = uplift_robust(p[sel], on[sel], wr[sel], pref)
    lo, hi = np.nanpercentile(est, [2.5, 97.5]); return (hi - lo) / 2


# ---- multi-year campaign ----------------------------------------------------
# Each year is an independent draw of the atmosphere: k_year ~ N(mu, sigma_B)
# (interannual = the realised counterpart of the epistemic Type-B spread), with
# the seasonal cycle aligned year-to-year and slow within-year weather noise.
N_YEARS = 8
years_k = RNG.normal(MU_K, SIGMA_B0, N_YEARS)
wdM = np.tile(wd_s, N_YEARS)
wsM = np.tile(ws_s, N_YEARS)
tM = np.concatenate([t_s + m * T_YEAR for m in range(N_YEARS)])
kM = np.concatenate([seasonal_k(t_s, years_k[m], phase=0.0, rng=RNG)
                     for m in range(N_YEARS)])
p_obs, on = make_obs(wdM, wsM, kM, RNG)
PREF = p_obs[~on].mean()                                  # fixed reference baseline power [kW]

k_curve = np.linspace(K_GRID[0], K_GRID[-1], 60)
benefit_curve = np.array([true_benefit(wd_s, ws_s, k) for k in k_curve])
slope = np.interp(MU_K, k_curve, np.gradient(benefit_curve, k_curve))
benefit_muk = np.interp(MU_K, k_curve, benefit_curve)
typeB_1sig = abs(slope) * SIGMA_B0                       # propagated Type-B 1-sigma [%]
TSPAN = N_YEARS * T_YEAR

print(f"N_YEARS={N_YEARS}  slope dDelta/dk = {slope:.0f} %/k  "
      f"Type-B 1sigma on benefit = {typeB_1sig:.3f} %")
print(f"interannual k means: {np.round(years_k, 4)}")

# ============================================================================
pdf = PdfPages("wffc_uncertainty_report.pdf")
BLUE, RED, PURPLE, GREY, GREEN = "#1f77b4", "#d62728", "#9467bd", "#7f7f7f", "#2ca02c"

# ---------------------------------------------------------------- PAGE 1 -----
fig = plt.figure(figsize=(8.27, 11.69))            # A4 portrait
fig.suptitle("WFFC wake-steering field test — stochastic inputs and Type-B uncertainty",
             fontsize=13, fontweight="bold", y=0.985)
gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1.25],
              hspace=0.55, wspace=0.28, top=0.93, bottom=0.06, left=0.10, right=0.95)

order = np.argsort(tM)
yr = tM[order] / T_YEAR
axA = fig.add_subplot(gs[0, :]); axB = fig.add_subplot(gs[1, :], sharex=axA)
axC = fig.add_subplot(gs[2, :], sharex=axA)


def year_grid(ax):
    for y in range(1, N_YEARS):
        ax.axvline(y, color="#dddddd", lw=0.6, zorder=0)


axA.plot(yr, wsM[order], lw=0.25, color=BLUE); axA.set_ylabel("ws [m/s]"); year_grid(axA)
axA.set_title(f"Stochastic time series ({N_YEARS}-year campaign, aligned waked sector 266–274°)",
              fontsize=10)
axB.plot(yr, kM[order], lw=0.5, color=RED); year_grid(axB)
axB.axhline(MU_K, ls="--", color="k", lw=0.8)
axB.fill_between([0, N_YEARS], MU_K - Z95 * SIGMA_B0, MU_K + Z95 * SIGMA_B0,
                 color=RED, alpha=0.12)
for m in range(N_YEARS):                                  # per-year mean k (interannual)
    axB.plot([m, m + 1], [years_k[m]] * 2, color="k", lw=1.6)
axB.set_ylabel("k  (atmosphere)")
axB.text(0.01, 0.93, "drifting $k(t)$ (thin); black = per-year mean (interannual); "
         "shaded = Type-B prior 95% on $k$", transform=axB.transAxes, fontsize=7, va="top")
beff = np.interp(kM, k_curve, benefit_curve)
axC.plot(yr, beff[order], lw=0.5, color=PURPLE); axC.axhline(0, color="k", lw=0.6)
year_grid(axC)
axC.set_ylabel("benefit [%]"); axC.set_xlabel("time [years]"); axC.set_xlim(0, N_YEARS)
axC.set_title("Instantaneous true wake-steering benefit drifts and flips sign with $k$",
              fontsize=9)

# bottom-left: Type-B assumptions table
axT = fig.add_subplot(gs[3, 0]); axT.axis("off")
rows = [
    ["epistemic param.", "wake expansion $k$"],
    ["$k$ prior (Type B)", f"$N({MU_K},\\ {SIGMA_B0})$"],
    ["seasonal swing", f"$\\pm{SEAS_AMP}$ (aleatoric)"],
    ["power noise (Type A)", f"{P_NOISE*100:.0f}% / sample"],
    ["toggle period", f"{TOGGLE_LEN*10} min, 50/50"],
    ["control yaw", f"{GAMMA:.0f}$^\\circ$ upstream"],
    ["$d\\Delta/dk$", f"{slope:.0f} %/unit-$k$"],
    ["→ Type-B 1$\\sigma$ benefit", f"{typeB_1sig:.2f} %"],
    ["campaign length", f"{N_YEARS} years"],
]
tbl = axT.table(cellText=rows, colLabels=["input / Type-B", "value"],
                cellLoc="left", colLoc="left", loc="center", bbox=[0, 0, 1, 1],
                colWidths=[0.52, 0.48])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.0)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor("#eeeeee"); cell.set_text_props(fontweight="bold")
axT.set_title("Assumed Type-B uncertainties", fontsize=10, pad=2)
axT.text(0, -0.06,
         "Other field Type-B sources (IEA Task 44): anemometer/vane calibration,\n"
         "yaw-dependent wind-speed/direction transfer functions, power transducers,\n"
         "wake-model form. None are reducible by longer measurement.",
         transform=axT.transAxes, fontsize=6.8, va="top", color="#444444")

# bottom-right: benefit vs k with Type-B band
axK = fig.add_subplot(gs[3, 1])
axK.plot(k_curve, benefit_curve, color="k", lw=2)
axK.axhline(0, color=GREY, lw=0.8)
axK.axvspan(MU_K - Z95 * SIGMA_B0, MU_K + Z95 * SIGMA_B0, color=RED, alpha=0.15,
            label="Type-B 95% on $k$")
axK.axvline(MU_K, ls="--", color="k", lw=0.8)
b_lo = np.interp(MU_K - Z95 * SIGMA_B0, k_curve, benefit_curve)
b_hi = np.interp(MU_K + Z95 * SIGMA_B0, k_curve, benefit_curve)
axK.axhspan(min(b_lo, b_hi), max(b_lo, b_hi), color=PURPLE, alpha=0.15,
            label="→ Type-B 95% on benefit")
axK.set_xlabel("wake-expansion $k$"); axK.set_ylabel("true benefit [%]")
axK.set_title("Type-B propagation: $k$ uncertainty → benefit", fontsize=9.5)
axK.legend(fontsize=7.5, loc="upper right")
pdf.savefig(fig); plt.close(fig)

# ---------------------------------------------------------------- PAGE 2 -----
fig = plt.figure(figsize=(8.27, 11.69))
fig.suptitle("Epistemic mean + uncertainty of the benefit over time windows",
             fontsize=13, fontweight="bold", y=0.985)
gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.32,
              top=0.92, bottom=0.07, left=0.11, right=0.93)

# Panel A: expanding window over the whole multi-year campaign
cum_t = np.arange(60.0, TSPAN + 1, 60.0)                  # bimonthly steps
est_c, true_c, hwA_c = [], [], []
for T in cum_t:
    msk = tM < T
    est_c.append(uplift_robust(p_obs[msk], on[msk], wsM[msk], PREF))
    true_c.append(true_benefit(wdM[msk], wsM[msk], kM[msk]))
    hwA_c.append(boot_hw(p_obs[msk], on[msk], wsM[msk], tM[msk], PREF, nb=250,
                         rng=np.random.default_rng(int(T))))
est_c = np.array(est_c); true_c = np.array(true_c); hwA_c = np.array(hwA_c)
tot_c = np.sqrt(hwA_c ** 2 + (Z95 * typeB_1sig) ** 2)
xcum = cum_t / T_YEAR

axS = fig.add_subplot(gs[0])
axS.fill_between(xcum, est_c - tot_c, est_c + tot_c, color=RED, alpha=0.18,
                 label="Type-A ⊕ Type-B 95% (honest)")
axS.fill_between(xcum, est_c - hwA_c, est_c + hwA_c, color=BLUE, alpha=0.30,
                 label="Type-A only (bootstrap 95%)")
axS.plot(xcum, est_c, color=BLUE, lw=1.8, label="toggle estimate $\\hat\\Delta$")
axS.plot(xcum, true_c, color="k", lw=1.6, ls="--", label="true benefit so far")
axS.axhline(0, color=GREY, lw=0.8)
axS.set_xlim(0, N_YEARS); axS.set_xlabel("elapsed campaign length [years]")
axS.set_ylabel("benefit [%]")
axS.set_title(f"Expanding window over {N_YEARS} years: Type-A (blue) keeps shrinking toward 0,\n"
              "the honest Type-A ⊕ Type-B band (red) floors at the epistemic limit",
              fontsize=10)
axS.legend(fontsize=8, ncol=2, loc="upper right")
axS.annotate(f"after {N_YEARS} yr:  Type-A ±{hwA_c[-1]:.2f}%  vs  Type-B floor ±{Z95*typeB_1sig:.2f}%",
             (xcum[-1], est_c[-1]), xytext=(N_YEARS * 0.30, est_c.max() + 1.0),
             fontsize=8.5, color="#333333",
             arrowprops=dict(arrowstyle="->", color="#333333"))

# Panel B: window-length sweep from 1 month to the full multi-year span
Wyr = np.array([1 / 12, 0.25, 0.5, 1, 2, 4, 8])
ale_std, typeA_hw = [], []
for W in Wyr * T_YEAR:
    nb_win = max(1, int(round(TSPAN / W)))
    edges = np.arange(0, nb_win * W + 1e-6, W)
    wins = list(zip(edges[:-1], edges[1:]))
    ev = [uplift_robust(p_obs[(tM >= a) & (tM < b)], on[(tM >= a) & (tM < b)],
                        wsM[(tM >= a) & (tM < b)], PREF)
          for a, b in wins if ((tM >= a) & (tM < b)).sum() >= 40]
    pick = np.unique(np.linspace(0, len(wins) - 1, min(len(wins), 20)).round().astype(int))
    hwv = []
    for jj in pick:
        a, b = wins[jj]; msk = (tM >= a) & (tM < b)
        if msk.sum() >= 40:
            hwv.append(boot_hw(p_obs[msk], on[msk], wsM[msk], tM[msk], PREF, nb=200,
                               rng=np.random.default_rng(100 + jj)))
    ale_std.append(np.nanstd(ev) if len(ev) > 1 else np.nan)
    typeA_hw.append(np.nanmean(hwv) if hwv else np.nan)
ale_std = np.array(ale_std); typeA_hw = np.array(typeA_hw)
typeB_hw = Z95 * typeB_1sig

axW = fig.add_subplot(gs[1])
axW.plot(Wyr, typeA_hw, "o-", color=BLUE, label="Type-A: bootstrap 95% half-width")
axW.plot(Wyr, ale_std, "v:", color=GREEN, label="window-to-window std (aleatoric)")
axW.axhline(typeB_hw, ls="--", color=RED, lw=2,
            label=f"Type-B 95% half-width (floor = {typeB_hw:.2f}%)")
axW.plot(Wyr, np.sqrt(typeA_hw ** 2 + typeB_hw ** 2), "s-", color="k",
         label="total (Type-A ⊕ Type-B)")
axW.set_xscale("log"); axW.set_xticks(Wyr)
axW.set_xticklabels(["1mo", "3mo", "6mo", "1y", "2y", "4y", "8y"])
axW.set_xlabel("window length"); axW.set_ylabel("benefit uncertainty half-width [%]")
axW.set_ylim(bottom=0)
axW.set_title("Over many years Type A drops BELOW the Type-B floor: more measurement\n"
              "cannot beat the epistemic limit — 'many measurements are not enough' (rule 3)",
              fontsize=10)
axW.legend(fontsize=8.5)
axW.text(0.99, 0.55,
         f"epistemic mean benefit\n$\\approx {benefit_muk:.2f}\\%$ (varies year to year)",
         transform=axW.transAxes, ha="right", fontsize=8, color="#444444",
         bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
pdf.savefig(fig); plt.close(fig)

pdf.close()
print("Wrote wffc_uncertainty_report.pdf")
