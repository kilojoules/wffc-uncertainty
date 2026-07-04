"""
Coverage of the true change-in-energy (IEA Task 44 dAEP) vs campaign length and
Type-B level — rebuilt per the adversarial review:

  * end-to-end error injection (b enters the observed power; no analytic shortcut)
  * length-independent estimand (fixed bin mask; deterministic truth) — C1
  * shared B=1000 vectorised block bootstrap, collision-free integer seeds — M3
  * Wilson 95% Monte-Carlo bands on every coverage curve — M3
  * common-mode control arm: the same b applied to BOTH states (metric cancels) — M2
  * honest-interval misspecification sweep: assumed sigma_B = 0.5x/1x/2x truth — M1
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import campaigns as C
import metrics as M

Z = 1.959964
EXP_ID = 1
Ls = [0.5, 1, 2, 4, 8]
R = 400
LEVELS = [0.0, 0.0025, 0.005, 0.01]          # sigma_B: 0 (control), 0.25%, 0.5%, 1%
COLORS = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728"]

TRUE = C.TRUE_DAEP
dDdb = 100.0 + TRUE                           # exact sensitivity of the metric to ON-only gain
print(f"true dAEP (deterministic estimand) = {TRUE:+.3f}%   dDdb = {dDdb:.1f}")


def ws_carrier_sensitivity(sig):
    """Deterministic |dΔAEP/db| for the ws (bin-migration) carrier at the
    operating scale `sig`. Unlike the ON-only power gain, a ws bias has NO
    closed-form 100+ΔAEP sensitivity — it depends on the power-curve slope and
    the bin layout. We evaluate it the way a real assessment would: perturb the
    ON-period measured ws by +-sig on the clean base-year record (no toggle
    noise), re-run the metric, and central-difference. Bin migration is discrete,
    so the slope is evaluated at the operating scale, not infinitesimally."""
    on0 = ((np.arange(C.N_S) // C.TOGGLE_LEN) % 2 == 1)
    p = np.where(on0, C.PON, C.POFF)

    def est_at(bb):
        ws = np.where(on0, C.WS_S * (1.0 + bb), C.WS_S)
        _, e, _ = M.campaign_estimates(p, p, on0, M.bin_index(C.WD_S, ws), fixed=C.FIXED_MASK)
        return e
    return abs(est_at(sig) - est_at(-sig)) / (2 * sig)


def simulate(mode, sigma_b, arm_id):
    """errs[Li, r] = est - TRUE and hws[Li, r] for one error-model arm."""
    errs = np.empty((len(Ls), R)); hws = np.empty((len(Ls), R))
    for Li, L in enumerate(Ls):
        for r in range(R):
            rng = C.rng_for(EXP_ID, arm_id * 1000 + Li, r)
            idx, blocks, on = C.make_campaign(L, rng)
            pc, po, bid, b = C.observe(idx, on, rng, sigma_b=sigma_b, mode=mode)
            _, est, _ = M.campaign_estimates(pc, po, on, bid, fixed=C.FIXED_MASK)
            hws[Li, r] = M.block_boot_halfwidth(po, on, bid, blocks, n_boot=1000,
                                                rng=rng, fixed=C.FIXED_MASK)
            errs[Li, r] = est - TRUE
    return errs, hws


def coverage(errs, hws, extra=0.0):
    """`extra` (the propagated Type-B half-width) may be a scalar or an array
    broadcastable to hws — the latter lets the sensitivity be computed
    per-campaign from the measured estimate (review nit [14])."""
    return 100 * (np.abs(errs) <= np.hypot(hws, extra)).mean(axis=1)


arms = {}
for ai, s in enumerate(LEVELS):
    arms[s] = simulate("on_only", s, ai)
    print(f"  sigma_B={s*100:.2g}% ON-only : coverage {np.round(coverage(*arms[s]), 1)}")
errs_cm, hws_cm = simulate("common", 0.005, 9)
print(f"  sigma_B=0.5% COMMON   : coverage {np.round(coverage(errs_cm, hws_cm), 1)}"
      f"   mean bias {np.round(errs_cm.mean(axis=1), 3)}")

# Type-B effect as EXCESS coverage loss over the control (review C1)
ctrl = coverage(*arms[0.0])
print("\nExcess coverage loss vs control [pp]:")
for s in LEVELS[1:]:
    print(f"  sigma_B={s*100:.2g}%: {np.round(ctrl - coverage(*arms[s]), 1)}")

# Honest-interval misspecification at true sigma_B=0.5% (reuses the same arm).
# Sensitivity is taken PER CAMPAIGN from the measured estimate (dDdb = 100 + est,
# review nit [14]) rather than the oracle TRUE — a real assessment never knows
# TRUE. The two agree to <0.4% here (est-TRUE is O(0.5 pp)), so the interval is
# essentially unchanged and, if anything, slightly conservative (measured>TRUE).
errs5, hws5 = arms[0.005]
dDdb_hat = 100.0 + (errs5 + TRUE)                    # per-campaign, from measured est
mis = {fac: coverage(errs5, hws5, extra=Z * dDdb_hat * 0.005 * fac) for fac in [0.5, 1.0, 2.0]}
print("\nHonest-interval coverage, assumed/true sigma_B (dDdb from measured est):")
for fac, cov in mis.items():
    print(f"  {fac:>3}x: {np.round(cov, 1)}")
print(f"  (oracle-dDdb 1x for comparison: "
      f"{np.round(coverage(errs5, hws5, extra=Z * dDdb * 0.005), 1)})")

# ---- M1: the ws (bin-migration) carrier — the repair needs the RIGHT sensitivity ----
# Result 1 flags the ws carrier as the WORST, and 'What to report' recommends the
# propagated term for yaw-dependent transfer functions (= this carrier). But the
# ws carrier has NO 100+dAEP sensitivity: naively reusing the ON-only term
# under-corrects ~2x and leaves the same length-scaling pathology. The honest fix
# is to propagate the ws carrier's OWN (numerically estimated) sensitivity.
SIG_WS = 0.005
S_WS = ws_carrier_sensitivity(SIG_WS)                # ~2x the ON-only dDdb
errs_ws, hws_ws = simulate("ws_carrier", SIG_WS, 8)
cov_ws_boot = coverage(errs_ws, hws_ws)                                   # bootstrap only
cov_ws_wrong = coverage(errs_ws, hws_ws, extra=Z * dDdb * SIG_WS)         # naive ON-only term
cov_ws_right = coverage(errs_ws, hws_ws, extra=Z * S_WS * SIG_WS)         # correct ws sensitivity
print(f"\nws carrier, sigma_B=0.5%: ON-only sensitivity dDdb={dDdb:.0f}, "
      f"ws sensitivity S_ws={S_WS:.0f} ({S_WS/dDdb:.1f}x larger)")
print(f"  bootstrap-only coverage            : {np.round(cov_ws_boot, 1)}")
print(f"  honest, NAIVE ON-only term (wrong) : {np.round(cov_ws_wrong, 1)}   <- still under-covers")
print(f"  honest, correct ws sensitivity     : {np.round(cov_ws_right, 1)}   <- restored")

# ---- figure -----------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(18.5, 5))
x = np.array(Ls)
for s, c in zip(LEVELS, COLORS):
    cov = coverage(*arms[s])
    lo = np.array([M.wilson(round(v * R / 100), R)[0] for v in cov])
    hi = np.array([M.wilson(round(v * R / 100), R)[1] for v in cov])
    lab = f"$\\sigma_B$={s*100:.2g}% ON-only" + ("  [control]" if s == 0 else "")
    ax[0].plot(x, cov, "o-", color=c, label=lab)
    ax[0].fill_between(x, lo, hi, color=c, alpha=0.12)
ax[0].plot(x, coverage(errs_cm, hws_cm), "^--", color=".4",
           label="$\\sigma_B$=0.5% common-mode (cancels)")
ax[0].axhline(95, ls="--", color=".4")
ax[0].set_xscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels([f"{l:g}y" for l in Ls])
ax[0].set_ylim(0, 103); ax[0].set_xlabel("campaign length")
ax[0].set_ylabel("bootstrap-CI coverage of true ΔAEP [%]")
ax[0].set_title("End-to-end injection; shaded = Wilson 95% MC bands.\n"
                "Differential (ON-only) systematics break coverage; common-mode cancels",
                fontsize=10)
ax[0].legend(fontsize=8, loc="lower left"); ax[0].grid(alpha=0.3)

for fac, c in zip([0.5, 1.0, 2.0], ["#d62728", "#2ca02c", "#1f77b4"]):
    ax[1].plot(x, mis[fac], "s-", color=c, label=f"assumed $\\sigma_B$ = {fac}× true")
ax[1].axhline(95, ls="--", color=".4")
ax[1].set_xscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels([f"{l:g}y" for l in Ls])
ax[1].set_ylim(0, 103); ax[1].set_xlabel("campaign length")
ax[1].set_ylabel("honest-interval coverage of true ΔAEP [%]")
ax[1].set_title("Honest interval under Type-B misspecification (true $\\sigma_B$=0.5%)\n"
                "the prior need not be exact — but a 2× underestimate matters", fontsize=10)
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.3)

# panel 3 (M1): the ws carrier needs its OWN sensitivity, not the ON-only term
for cov, col, mk, lab in [
        (cov_ws_boot, "#d62728", "o-", "bootstrap only"),
        (cov_ws_wrong, "#ff7f0e", "s--", f"honest, naive ON-only term ($z\\,\\cdot{dDdb:.0f}\\,\\sigma_B$)"),
        (cov_ws_right, "#2ca02c", "s-", f"honest, correct ws term ($z\\,\\cdot{S_WS:.0f}\\,\\sigma_B$)")]:
    ax[2].plot(x, cov, mk, color=col, label=lab)
ax[2].axhline(95, ls="--", color=".4")
ax[2].set_xscale("log"); ax[2].set_xticks(x); ax[2].set_xticklabels([f"{l:g}y" for l in Ls])
ax[2].set_ylim(0, 103); ax[2].set_xlabel("campaign length")
ax[2].set_ylabel("coverage of true ΔAEP [%]")
ax[2].set_title("ws (bin-migration) carrier, true $\\sigma_B$=0.5%\n"
                f"its sensitivity is {S_WS/dDdb:.1f}× the ON-only term — the naive repair still fails",
                fontsize=10)
ax[2].legend(fontsize=8.5, loc="lower left"); ax[2].grid(alpha=0.3)

plt.tight_layout(); plt.savefig("fig_typeB_levels.png", dpi=130); plt.close()
print("\nWrote fig_typeB_levels.png")
