"""
Coverage of the true change-in-energy (IEA dAEP, Eq. 12) vs campaign length and
Type-B level — rebuilt per the adversarial review:

  * end-to-end error injection (b enters the observed power; no analytic shortcut)
  * length-independent estimand (fixed bin mask; deterministic truth) — C1
  * shared B=1000 vectorised block bootstrap, collision-free integer seeds — M3
  * Wilson 95% Monte-Carlo bands on every coverage curve — M3
  * common-mode control arm: the same b applied to BOTH states (Eq. 12 cancels) — M2
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
dDdb = 100.0 + TRUE                           # exact sensitivity of Eq.12 to ON-only gain
print(f"true dAEP (deterministic estimand) = {TRUE:+.3f}%   dDdb = {dDdb:.1f}")


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

# Honest-interval misspecification at true sigma_B=0.5% (reuses the same arm)
errs5, hws5 = arms[0.005]
mis = {fac: coverage(errs5, hws5, extra=Z * dDdb * 0.005 * fac) for fac in [0.5, 1.0, 2.0]}
print("\nHonest-interval coverage, assumed/true sigma_B:")
for fac, cov in mis.items():
    print(f"  {fac:>3}x: {np.round(cov, 1)}")

# ---- figure -----------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
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
plt.tight_layout(); plt.savefig("fig_typeB_levels.png", dpi=130); plt.close()
print("\nWrote fig_typeB_levels.png")
