"""
WFFC toggle-test uncertainty experiment
=======================================
Demonstrates, with a real PyWake wake-steering simulation, why the bootstrap
confidence interval reported in WFFC field-validation studies (GUM Type A only)
is the wrong uncertainty to report, and why the area metric / Type-B-aware
interval (Quick et al. 2025, Renew. Energy 240:122028) is correct.

Story
-----
The wake-expansion coefficient k is THE atmospheric parameter. It
  * drifts seasonally over the campaign      -> aleatoric, "weather changing"
  * is unknown to the analyst for the future -> epistemic, GUM Type B.
Because the wake-steering benefit Delta_true(k) swings from +6.7% (low k,
stable) to -3% (high k, turbulent), the benefit you "measure" is conditional
on the atmosphere you happened to sample.

  * Bootstrap 95% CI (Type A): shrinks ~1/sqrt(N) toward the campaign value,
    and is BLIND to k uncertainty. -> fails the time test (collapses to 0,
    "many measurements are not enough") and the uncertainty test (does not
    grow when Type B grows).
  * Area metric / Type-B interval: grows when Type B grows (uncertainty test)
    and floors at a non-zero value as N->inf (time test). -> passes both.

Outputs: 5 figures + a printed summary. RAM < 1 GB, single-CPU PyWake.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pywake_model import build_lookup, WS_GRID, K_GRID, GAMMA

RNG = np.random.default_rng(7)
OUT = "."

# ============================================================================
# 0. Build the PyWake lookup (the only place PyWake is called)
# ============================================================================
print("Building PyWake lookup ...")
F_OFF, F_ON, _, _ = build_lookup()


def farm_power(wd, ws, k, control):
    """Vectorised farm power [kW] from the lookup. control=False/True -> off/on."""
    wd = np.atleast_1d(wd); ws = np.clip(np.atleast_1d(ws), 3.0, 25.0)
    k = np.broadcast_to(np.atleast_1d(k), wd.shape)
    pts = np.column_stack([wd, ws, k])
    return (F_ON if control else F_OFF)(pts)


def true_benefit(wd, ws, k):
    """Energy-weighted relative wake-steering benefit [%] for atmospheric k.

    k may be a scalar (constant atmosphere) or per-timestep array (drift).
    Uses the campaign's own (wd, ws) distribution as the weighting.
    """
    poff = farm_power(wd, ws, k, control=False)
    pon = farm_power(wd, ws, k, control=True)
    return 100.0 * (pon.sum() - poff.sum()) / poff.sum()


# ============================================================================
# 1. Campaign conditions: 1 year of 10-min data, filtered to the waked sector
# ============================================================================
from py_wake.examples.data import example_data_path
_d = np.load(example_data_path + "/time_series.npz")
WD_ALL, WS_ALL = _d["wd"], _d["ws"]
DT_DAYS = 1.0 / 6 / 24                                  # 10-min step in days
T_YEAR = len(WD_ALL) * DT_DAYS                          # ~365 days

# Aligned sector around 270 deg: this is where the steering pair is waked and
# where field campaigns report the benefit (Fleming et al. bin by wd; the
# deepest-wake bins are the headline). Outside it, yaw only costs power.
SECTOR = (WD_ALL >= 266) & (WD_ALL <= 274) & (WS_ALL >= 3) & (WS_ALL <= 25)
idx = np.where(SECTOR)[0]
wd_s, ws_s = WD_ALL[idx], WS_ALL[idx]
t_s = idx * DT_DAYS                                     # time [days] of sector samples
N_S = len(idx)
print(f"Waked-sector samples: {N_S} of {len(WD_ALL)} ({100*N_S/len(WD_ALL):.1f}%)")


def seasonal_k(t_days, k_year, amp=0.008, phase=0.0, ar_sigma=0.0015, rng=RNG):
    """True atmospheric k(t): yearly mean + seasonal swing + slow weather noise."""
    seas = amp * np.sin(2 * np.pi * t_days / T_YEAR + phase)
    # slow AR(1) weather noise on a daily grid, interpolated to samples
    nd = int(np.ceil(t_days.max())) + 2
    w = np.zeros(nd)
    for i in range(1, nd):
        w[i] = 0.96 * w[i - 1] + ar_sigma * rng.standard_normal()
    noise = np.interp(t_days, np.arange(nd), w)
    return np.clip(k_year + seas + noise, K_GRID[0], K_GRID[-1])


# ============================================================================
# 2. Field toggle estimator + block bootstrap (current community practice)
# ============================================================================
TOGGLE_LEN = 7          # 70-min blocks (not a divisor of 24 h -> avoids diurnal sync)
WS_BINS = np.arange(3, 26, 1.0)
P_NOISE = 0.01          # 1% power measurement noise (sensor, Type A)


NB = len(WS_BINS) + 2


def make_observations(wd, ws, k_t, rng, noise=P_NOISE):
    """Simulate one toggle campaign. Returns per-sample observed farm power and
    the toggle state — i.e. what the field analyst sees (ambient ws/time known)."""
    poff = farm_power(wd, ws, k_t, control=False)
    pon = farm_power(wd, ws, k_t, control=True)
    on = ((np.arange(len(wd)) // TOGGLE_LEN) % 2 == 1)  # 50/50 ON/OFF toggling
    p_obs = np.where(on, pon, poff)
    p_obs = p_obs * (1 + noise * rng.standard_normal(len(wd)))
    return p_obs, on


def energy_ratio_uplift(p_obs, on, ws_ref):
    """Fleming-2019 binned energy-ratio benefit [%] from toggle data (vectorised)."""
    digit = np.digitize(ws_ref, WS_BINS)
    off = ~on
    son = np.bincount(digit[on], p_obs[on], NB); non = np.bincount(digit[on], None, NB)
    sof = np.bincount(digit[off], p_obs[off], NB); nof = np.bincount(digit[off], None, NB)
    valid = (non >= 2) & (nof >= 2)
    mon = son / np.where(non > 0, non, 1)
    mof = sof / np.where(nof > 0, nof, 1)
    w = (non + nof) * valid                             # occurrence weight per ws bin
    num = (w * (mon - mof)).sum(); den = (w * mof).sum()
    return 100.0 * num / den if den > 0 else np.nan


def block_bootstrap_ci(p_obs, on, ws_ref, t_days, n_boot=2000, block_days=2.0,
                       rng=RNG):
    """Block bootstrap (resample ~2-day blocks) -> distribution of the uplift."""
    blk = (t_days / block_days).astype(int)
    ublk = np.unique(blk)
    members = [np.where(blk == b)[0] for b in ublk]
    nb = len(ublk)
    est = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, nb, nb)
        sel = np.concatenate([members[j] for j in pick])
        est[i] = energy_ratio_uplift(p_obs[sel], on[sel], ws_ref[sel])
    return est


# ============================================================================
# 3. ONE representative campaign  (figures 1 & 2)
# ============================================================================
MU_K = 0.034            # climatological mean k
SIGMA_B0 = 0.004        # GUM Type B: interannual / epistemic std of k
Z95 = 1.959964

k_year_obs = MU_K                                       # this campaign's yearly mean
k_t = seasonal_k(t_s, k_year_obs, phase=RNG.uniform(0, 2*np.pi))
p_obs, on = make_observations(wd_s, ws_s, k_t, RNG)
ws_ref, t_days = ws_s, t_s

uplift_hat = energy_ratio_uplift(p_obs, on, ws_ref)
boot = block_bootstrap_ci(p_obs, on, ws_ref, t_days)
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

# Benefit-vs-k curve and its local sensitivity (for Type-B propagation)
k_curve = np.linspace(K_GRID[0], K_GRID[-1], 60)
benefit_curve = np.array([true_benefit(wd_s, ws_s, k) for k in k_curve])
dDelta_dk = np.gradient(benefit_curve, k_curve)
slope_at_muk = np.interp(MU_K, k_curve, dDelta_dk)
benefit_at_muk = np.interp(MU_K, k_curve, benefit_curve)

# True benefit under this campaign's drifting atmosphere (the counterfactual)
true_campaign_benefit = true_benefit(wd_s, ws_s, k_t)

print(f"\nOne campaign:")
print(f"  measured uplift  Delta_hat = {uplift_hat:+.3f} %")
print(f"  bootstrap 95% CI          = [{ci_lo:+.3f}, {ci_hi:+.3f}]  (half-width {(ci_hi-ci_lo)/2:.3f})")
print(f"  TRUE campaign benefit     = {true_campaign_benefit:+.3f} %")
print(f"  dDelta/dk at mu_k         = {slope_at_muk:.1f} %/unit-k")
print(f"  Type-B 1-sigma on benefit = {abs(slope_at_muk)*SIGMA_B0:.3f} %")

# --- Figure 1: the setup (conditions + atmosphere drift + benefit drift) -----
fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
order = np.argsort(t_days)
ax[0].plot(t_days[order], ws_s[order], lw=0.4, color="#1f77b4")
ax[0].set_ylabel("ws [m/s]"); ax[0].set_title("Waked-sector inflow (1-year campaign)")
ax[1].plot(t_days[order], wd_s[order], lw=0.4, color="#2ca02c")
ax[1].set_ylabel("wd [deg]")
ax[2].plot(t_days[order], k_t[order], lw=0.6, color="#d62728")
ax[2].axhline(MU_K, ls="--", color="k", lw=0.8, label="climatological mean")
ax[2].set_ylabel("k (atmosphere)"); ax[2].legend(loc="upper right", fontsize=8)
beff = np.array([np.interp(k, k_curve, benefit_curve) for k in k_t])
ax[3].plot(t_days[order], beff[order], lw=0.6, color="#9467bd")
ax[3].axhline(0, color="k", lw=0.6)
ax[3].set_ylabel("true benefit [%]"); ax[3].set_xlabel("time [days]")
ax[3].set_title("Instantaneous true wake-steering benefit drifts (and flips sign) with the atmosphere",
                fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_setup.png", dpi=130); plt.close()

# --- Figure 2: headline -- tight bootstrap CI vs true predictive spread ------
k_future = RNG.normal(MU_K, SIGMA_B0, 4000)
benefit_future = np.interp(k_future, k_curve, benefit_curve)   # true benefit, future years
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(benefit_future, bins=50, density=True, color="#bbbbbb", alpha=0.8,
        label="true benefit across plausible atmospheres\n(Type B: $k\\sim N(\\mu,\\sigma_B)$)")
ax.axvspan(ci_lo, ci_hi, color="#1f77b4", alpha=0.25,
           label=f"reported bootstrap 95% CI (Type A)\n[{ci_lo:+.2f}, {ci_hi:+.2f}] %")
ax.axvline(uplift_hat, color="#1f77b4", lw=2, label=f"measured $\\hat\\Delta$ = {uplift_hat:+.2f} %")
ax.axvline(benefit_at_muk, color="k", lw=1.5, ls="--",
           label=f"long-term true benefit = {benefit_at_muk:+.2f} %")
ax.set_xlabel("wake-steering energy benefit [%]"); ax.set_ylabel("density")
ax.set_title("The bootstrap CI is tight and confident — and far too narrow:\n"
             "it ignores that the benefit itself depends on the (uncertain) atmosphere",
             fontsize=10)
ax.legend(fontsize=8, loc="upper right"); plt.tight_layout()
plt.savefig(f"{OUT}/fig2_miscoverage.png", dpi=130); plt.close()


# ============================================================================
# 4. The two tests  (figure 3)  -- area metric vs bootstrap CI
# ============================================================================
def area_metric(model_samples, obs_samples, grid=None):
    """Area between model CDF M and observation CDF F = integral|M-F| df
    = Wasserstein-1 distance.  (Quick et al. 2025, Eq. 13.)"""
    if grid is None:
        lo = min(model_samples.min(), obs_samples.min())
        hi = max(model_samples.max(), obs_samples.max())
        grid = np.linspace(lo, hi, 1000)
    M = np.searchsorted(np.sort(model_samples), grid, side="right") / len(model_samples)
    F = np.searchsorted(np.sort(obs_samples), grid, side="right") / len(obs_samples)
    return np.trapz(np.abs(M - F), grid)


# --- Test 1: uncertainty test (inflate Type B by eta) ------------------------
etas = np.array([1, 2, 3, 4, 5, 6])
boot_hw_eta = np.full_like(etas, (ci_hi - ci_lo) / 2, dtype=float)  # Type A: blind to eta
typeB_hw_eta = Z95 * np.abs(slope_at_muk) * (etas * SIGMA_B0)       # grows linearly

# --- Test 2: time test (grow campaign length over MANY years) ----------------
# Each extra year adds independent toggle blocks, so the Type-A bootstrap CI
# shrinks ~1/sqrt(N) toward 0. Each year has its own atmosphere k_year (the
# irreducible Type B), so the honest interval / area metric floor out.
years_list = np.array([1, 2, 4, 8, 16, 32, 64])
boot_hw_N, typeB_hw_N = [], []
for M in years_list:
    wdm = np.tile(wd_s, M); wsm = np.tile(ws_s, M)
    tm = np.concatenate([t_s + yr * T_YEAR for yr in range(M)])
    km = np.concatenate([seasonal_k(t_s, RNG.normal(MU_K, SIGMA_B0),
                                    phase=RNG.uniform(0, 2*np.pi), rng=RNG)
                         for yr in range(M)])
    pm, onm = make_observations(wdm, wsm, km, RNG)
    bsub = block_bootstrap_ci(pm, onm, wsm, tm, n_boot=400)
    lo, hi = np.percentile(bsub, [2.5, 97.5])
    boot_hw_N.append((hi - lo) / 2)
    typeB_hw_N.append(Z95 * np.abs(slope_at_muk) * SIGMA_B0)         # floor, N-independent
boot_hw_N = np.array(boot_hw_N); typeB_hw_N = np.array(typeB_hw_N)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
# uncertainty test
ax[0].plot(etas, boot_hw_eta, "o-", color="#1f77b4", label="bootstrap 95% CI half-width (Type A)")
ax[0].plot(etas, typeB_hw_eta, "s-", color="#d62728", label="Type-B-aware half-width")
ax[0].set_xlabel("Type-B uncertainty multiplier  $\\eta$  ($\\sigma_B=\\eta\\,\\sigma_{B0}$)")
ax[0].set_ylabel("reported uncertainty half-width [%]")
ax[0].set_title("Uncertainty test\nreported uncertainty should GROW as uncertainty grows", fontsize=10)
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
ax[0].annotate("bootstrap is FLAT\n(blind to Type B) — FAILS", (etas[-1], boot_hw_eta[-1]),
               xytext=(3.0, boot_hw_eta[0] + 3), fontsize=9, color="#1f77b4",
               arrowprops=dict(arrowstyle="->", color="#1f77b4"))
# time test
ax[1].plot(years_list, boot_hw_N, "o-", color="#1f77b4", label="bootstrap 95% CI half-width (Type A)")
ax[1].plot(years_list, typeB_hw_N, "s-", color="#d62728", label="Type-B-aware half-width (floor)")
ax[1].set_xscale("log", base=2)
ax[1].set_xticks(years_list); ax[1].set_xticklabels(years_list)
ax[1].set_xlabel("campaign length [years]   (more independent toggle blocks)")
ax[1].set_ylabel("reported uncertainty half-width [%]")
ax[1].set_ylim(bottom=0)
ax[1].set_title("Time test\nreported uncertainty should fall but NOT to 0 (rule 3)", fontsize=10)
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
ax[1].annotate("bootstrap -> 0\n(many measurements\nare not enough) — FAILS",
               (years_list[-1], boot_hw_N[-1]), xytext=(years_list[1], typeB_hw_N[0]*0.55),
               fontsize=9, color="#1f77b4",
               arrowprops=dict(arrowstyle="->", color="#1f77b4"))
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_two_tests.png", dpi=130); plt.close()

# ============================================================================
# 4. Coverage of the FUTURE benefit  (figure 4)  -- the practical consequence
# ============================================================================
print("\nCoverage experiment (this is the punchline) ...")
R = 300
cover_boot = cover_typeB = 0
hw_boot_list, hw_typeB_list = [], []
for r in range(R):
    rng = np.random.default_rng(1000 + r)
    k_year_r = rng.normal(MU_K, SIGMA_B0)              # this year's atmosphere
    k_tr = seasonal_k(t_s, k_year_r, phase=rng.uniform(0, 2*np.pi), rng=rng)
    p_r, on_r = make_observations(wd_s, ws_s, k_tr, rng)
    dhat = energy_ratio_uplift(p_r, on_r, ws_s)
    br = block_bootstrap_ci(p_r, on_r, ws_s, t_s, n_boot=600, rng=rng)
    lo, hi = np.percentile(br, [2.5, 97.5])
    hw_b = (hi - lo) / 2
    # honest Type-B-aware PREDICTION interval: Type A (bootstrap) + Type B (k).
    # The campaign year and the future year are two INDEPENDENT draws of the
    # atmosphere, so their benefits differ by slope*(k_future - k_obs) with
    # variance 2*(slope*sigma_B)^2  -> the sqrt(2) factor.
    hw_tb = np.sqrt(hw_b**2 + 2 * (Z95 * abs(slope_at_muk) * SIGMA_B0)**2)
    # target: the benefit in an INDEPENDENT future year (different atmosphere)
    k_future_year = rng.normal(MU_K, SIGMA_B0)
    target = true_benefit(wd_s, ws_s, k_future_year)
    cover_boot += (lo <= target <= hi)
    cover_typeB += (dhat - hw_tb <= target <= dhat + hw_tb)
    hw_boot_list.append(hw_b); hw_typeB_list.append(hw_tb)

cov_b = 100 * cover_boot / R
cov_tb = 100 * cover_typeB / R
print(f"  bootstrap 95% CI   coverage of future benefit = {cov_b:5.1f} %  (target 95%)")
print(f"  Type-B-aware 95%   coverage of future benefit = {cov_tb:5.1f} %  (target 95%)")
print(f"  mean half-width: bootstrap={np.mean(hw_boot_list):.3f}%  Type-B={np.mean(hw_typeB_list):.3f}%")

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(["bootstrap 95% CI\n(Type A only)", "Type-B-aware 95%\n(area-metric framework)"],
              [cov_b, cov_tb], color=["#1f77b4", "#d62728"], alpha=0.85)
ax.axhline(95, ls="--", color="k", label="nominal 95%")
ax.set_ylabel("empirical coverage of future benefit [%]"); ax.set_ylim(0, 100)
ax.set_title(f"Does the reported interval contain next year's true benefit?\n"
             f"({R} simulated campaigns)", fontsize=10)
for b, v in zip(bars, [cov_b, cov_tb]):
    ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=11)
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig4_coverage.png", dpi=130); plt.close()

print("\nFigures written: fig1_setup.png .. fig4_coverage.png")
print("Done.")
