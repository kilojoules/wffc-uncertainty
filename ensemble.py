"""
Result 4 — techniques across a LARGE ENSEMBLE of believable scenarios,
for the campaign lengths that actually occur in the field (< 2 years).

Motivation: the earlier results each fix one scenario (one k, one sigma_B, one
carrier). A practitioner does not know their scenario. So we ask the honest
decision-relevant question: averaged over believable field campaigns — and in
the WORST believable case — how well does each uncertainty technique do?

Each scenario is a believable toggle campaign, drawn independently:
  * wake coefficient k ~ U(0.025, 0.045)      -> true benefit ~ +2 % .. +12 %
    (25 % of scenarios are PLACEBO: no real benefit, true dAEP = 0 exactly);
  * systematic level sigma_B ~ U(0, 0.5 %)    (0 = well-calibrated; 0.5 % is a
    believable upper end for the POST-CALIBRATION RESIDUAL of a differential
    error. Raw uncorrected yaw-dependent nacelle-anemometer biases are
    multi-percent, so this band assumes a yawed-NTF correction has been applied;
    "worst case" below means worst WITHIN this prior band. A wider band makes
    bootstrap-only strictly worse and leaves carrier-aware calibrated.);
  * physical carrier ~ {ON-only power 0.4, ws / yaw-transfer 0.4, common 0.2}
    (common-mode cancels in the toggle; the two differential carriers are the
    ones that can bite);
  * random scatter sigma_A ~ U(1.5 %, 3.5 %) i.i.d.

Four techniques, all given the true sigma_B (so the ONLY thing that varies
between them is how the Type-B term is built — this isolates the technique, not
the prior; sigma_B misspecification is covered separately in Result 3):
  1. bootstrap only        — current field practice (Type-A only)
  2. honest, one-size term  — bootstrap (+) z*(100+dAEP)*sigma_B for EVERY
                              campaign (the naive reading of "add a Type-B term")
  3. honest, carrier-aware  — bootstrap (+) z*S_carrier*sigma_B with the ACTUAL
                              carrier's sensitivity (100+dAEP for ON-only power,
                              the propagated bin-migration sensitivity for ws,
                              ~0 for common-mode)
  4. carrier-aware, ws misread as common — technique 3 after the single most
                              dangerous attribution error: calling a
                              differential ws systematic "common-mode" (S=0),
                              which silently reverts to the bare bootstrap
                              interval. Result 3 stresses a mis-SIZED sigma_B;
                              only this arm stresses a mis-IDENTIFIED carrier.

Reported per campaign length and pooled over < 2 yr:
  * coverage of the true benefit  (target 95 %): mean, worst-case (min) and the
    robust near-worst (5th percentile) across scenarios;
  * for the placebo scenarios, the false "benefit declared" rate (target <=2.5 %):
    mean and worst-case (max).
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import campaigns as C
import metrics as M

Z = 1.959964
EXP_ID = 6
LS = [0.5, 1.0, 2.0]                      # the < 2-year field regime
N_SCEN = 100                              # believable scenarios
R = 80                                    # campaigns per (scenario, length)
N_BOOT = 400
P_PLACEBO = 0.25                          # fraction of scenarios with no real benefit
CARRIERS = ["on_only", "ws_carrier", "common"]
CARRIER_P = [0.40, 0.40, 0.20]

_ON0 = ((np.arange(C.N_S) // C.TOGGLE_LEN) % 2 == 1)
_BID0 = M.bin_index(C.WD_S, C.WS_S)


def scenario_powers(kval, placebo):
    pon = C.farm_power(C.WD_S, C.WS_S, True, k=kval)
    poff = C.farm_power(C.WD_S, C.WS_S, False, k=kval)
    if placebo:
        pon = poff.copy()
    true = M.deterministic_truth(pon, poff, _BID0, C.FIXED_MASK)
    return pon, poff, true


def ws_sensitivity(pon, poff, sig):
    """|dDAEP/db| for the ws (bin-migration) carrier on this scenario's powers.

    The bin-migration response est(b) is a STAIRCASE in b: below a scale of
    ~3e-3 a central secant sits under the discreteness knee and understates the
    slope at the coverage-relevant realized bias (|b| ~ 2 sigma_b), by up to
    ~2x at very small scales. Callers must evaluate on the plateau — pass
    max(2*sigma_b, 3e-3), never a tiny scale (typeB_levels.py uses 5e-3)."""
    p = np.where(_ON0, pon, poff)

    def est_at(bb):
        ws = np.where(_ON0, C.WS_S * (1.0 + bb), C.WS_S)
        _, e, _ = M.campaign_estimates(p, p, _ON0, M.bin_index(C.WD_S, ws), fixed=C.FIXED_MASK)
        return e
    return abs(est_at(sig) - est_at(-sig)) / (2 * sig)


# ---- draw the scenario ensemble (reproducible) ------------------------------
srng = np.random.default_rng(np.random.SeedSequence([EXP_ID, 424242]))
scen = []
for s in range(N_SCEN):
    placebo = srng.random() < P_PLACEBO
    kval = srng.uniform(0.025, 0.045)
    sigma_b = srng.uniform(0.0, 0.005)
    carrier = CARRIERS[srng.choice(3, p=CARRIER_P)]
    sig_a = srng.uniform(0.015, 0.035)
    pon, poff, true = scenario_powers(kval, placebo)
    # plateau-scale secant (see ws_sensitivity docstring): 2 sigma_b covers the
    # realized-bias range, floored at 3e-3 to stay above the discreteness knee
    S_ws = ws_sensitivity(pon, poff, max(2.0 * sigma_b, 3e-3))
    scen.append(dict(placebo=placebo, k=kval, sigma_b=sigma_b, carrier=carrier,
                     sig_a=sig_a, pon=pon, poff=poff, true=true, S_ws=S_ws))

n_pl = sum(sc["placebo"] for sc in scen)
print(f"{N_SCEN} scenarios ({n_pl} placebo / {N_SCEN-n_pl} real), R={R}, B={N_BOOT}, "
      f"lengths={LS}")
print("carrier mix:", {c: sum(sc['carrier'] == c for sc in scen) for c in CARRIERS})

TECHS = ["bootstrap", "honest_onesize", "honest_aware", "aware_wrongc"]
# coverage[tech][Li] = array over scenarios of per-scenario coverage
cov = {t: np.full((len(LS), N_SCEN), np.nan) for t in TECHS}
# false "benefit declared" rate, placebo scenarios only
fpr = {t: np.full((len(LS), N_SCEN), np.nan) for t in TECHS}

t0 = time.time()
for si, sc in enumerate(scen):
    sB = sc["sigma_b"]
    # correct carrier sensitivity for the aware technique
    for Li, L in enumerate(LS):
        est = np.empty(R); hw = np.empty(R)
        for r in range(R):
            rng = np.random.default_rng(np.random.SeedSequence([EXP_ID, si, Li, r]))
            idx, blocks, on = C.make_campaign(L, rng)
            pc, po, bid, b = C.observe(idx, on, rng, sigma_b=sB, mode=sc["carrier"],
                                       placebo=sc["placebo"], sig_a=sc["sig_a"],
                                       pon_src=sc["pon"], poff_src=sc["poff"])
            _, e, _ = M.campaign_estimates(pc, po, on, bid, fixed=C.FIXED_MASK)
            hw[r] = M.block_boot_halfwidth(po, on, bid, blocks, n_boot=N_BOOT,
                                           rng=rng, fixed=C.FIXED_MASK)
            est[r] = e
        # per-campaign sensitivities
        S_onesize = 100.0 + est                       # ON-only coefficient, used for all
        if sc["carrier"] == "on_only":
            S_aware = 100.0 + est
        elif sc["carrier"] == "ws_carrier":
            S_aware = np.full(R, sc["S_ws"])
        else:                                          # common-mode cancels -> ~0
            S_aware = np.zeros(R)
        # misattribution stress arm: the ws carrier misread as common-mode (S=0);
        # on_only and common are correctly identified, so only the ws->common
        # direction — the silent reversion to the bare bootstrap — is stressed
        S_wrong = np.zeros(R) if sc["carrier"] == "ws_carrier" else S_aware
        hw_tech = {
            "bootstrap": hw,
            "honest_onesize": np.hypot(hw, Z * S_onesize * sB),
            "honest_aware": np.hypot(hw, Z * S_aware * sB),
            "aware_wrongc": np.hypot(hw, Z * S_wrong * sB),
        }
        err = np.abs(est - sc["true"])
        for t in TECHS:
            cov[t][Li, si] = 100.0 * (err <= hw_tech[t]).mean()
            if sc["placebo"]:
                fpr[t][Li, si] = 100.0 * (est - hw_tech[t] > 0).mean()
    if (si + 1) % 10 == 0:
        print(f"  {si+1}/{N_SCEN} scenarios  ({time.time()-t0:.0f}s)")

print(f"done in {time.time()-t0:.0f}s\n")

# ---- report -----------------------------------------------------------------
LAB = {"bootstrap": "bootstrap only (current practice)",
       "honest_onesize": "honest, one-size (100+dAEP) term",
       "honest_aware": "honest, carrier-aware (recommended)",
       "aware_wrongc": "carrier-aware, ws misread as common"}


def agg(a):
    a = a[~np.isnan(a)]
    return a.mean(), np.percentile(a, 5), a.min()


def mean_se(a):
    """Pooled mean and its MC standard error, clustered at scenario level
    (cells within a scenario share the drawn k/sigma_B/carrier)."""
    sm = np.nanmean(a, axis=0)                         # per-scenario mean over lengths
    sm = sm[~np.isnan(sm)]
    return sm.mean(), sm.std(ddof=1) / np.sqrt(len(sm))


print("COVERAGE of the true benefit [%], target 95 (mean / 5th-pct / worst):")
for t in TECHS:
    row = "  ".join(f"{L:g}y {agg(cov[t][Li])[0]:.0f}/{agg(cov[t][Li])[1]:.0f}/{agg(cov[t][Li])[2]:.0f}"
                    for Li, L in enumerate(LS))
    m, p5, mn = agg(cov[t])                            # pooled < 2 yr
    _, se = mean_se(cov[t])
    print(f"  {LAB[t]:38s} | {row}  | <2y {m:.0f}(+-{2*se:.0f})/{p5:.0f}/{mn:.0f}")
print(f"  (per-cell granularity 100/R = {100.0/R:.2f} pp; 'worst' is the min over "
      f"{len(LS)*N_SCEN} binomial(R={R}) cells and is biased low by noise alone — "
      f"read the 5th-pct column as the robust near-worst)")

print("\nFALSE 'benefit declared' rate on PLACEBO scenarios [%], target <=2.5 (mean / worst):")
for t in TECHS:
    row = "  ".join(f"{L:g}y {np.nanmean(fpr[t][Li]):.1f}/{np.nanmax(fpr[t][Li]):.1f}"
                    for Li, L in enumerate(LS))
    allf = fpr[t][~np.isnan(fpr[t])]
    print(f"  {LAB[t]:38s} | {row}  | <2y {allf.mean():.1f}/{allf.max():.1f}")

print("\nPOOLED placebo false-declare rate over all placebo campaign-decisions, Wilson 95%:")
for t in TECHS:
    rates = fpr[t][~np.isnan(fpr[t])]
    k = int(round((rates / 100.0 * R).sum())); n = rates.size * R
    lo, hi = M.wilson(k, n)
    print(f"  {LAB[t]:38s} | {100.0*k/n:.2f}%  [{lo:.2f}, {hi:.2f}]   (k={k}/{n})")
# null reference for the max-based 'worst' column: a PERFECTLY calibrated
# technique (p=2.5% in every cell) still shows a large max over many small-R cells
_nrng = np.random.default_rng(np.random.SeedSequence([EXP_ID, 777]))
_ncells = len(LS) * n_pl
_nmax = _nrng.binomial(R, 0.025, size=(20000, _ncells)).max(axis=1) * 100.0 / R
print(f"  null reference: perfect calibration still shows worst-case (max over "
      f"{_ncells} cells, R={R}) median {np.median(_nmax):.1f}%, "
      f"5-95% [{np.percentile(_nmax, 5):.1f}, {np.percentile(_nmax, 95):.1f}]%")

# ---- figure -----------------------------------------------------------------
COL = {"bootstrap": "#d62728", "honest_onesize": "#ff7f0e", "honest_aware": "#2ca02c",
       "aware_wrongc": "#9467bd"}
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

# (a) ECDF of per-scenario coverage pooled over < 2 yr
for t in TECHS:
    v = np.sort(cov[t].ravel())
    ax[0].step(v, 100 * np.arange(1, len(v) + 1) / len(v), where="post",
               color=COL[t], lw=2, label=LAB[t])
ax[0].axvline(95, ls="--", color=".4", lw=1.2)
ax[0].text(95, 4, " target 95%", fontsize=8, color=".3")
ax[0].set_xlabel("per-scenario coverage of the true benefit [%]  (campaigns < 2 yr)")
ax[0].set_ylabel("% of believable scenarios at or below")
ax[0].set_title("Distribution over the ensemble: the bootstrap has a long\n"
                "low-coverage tail; the carrier-aware interval clusters at ~95%", fontsize=10)
ax[0].legend(fontsize=8.5, loc="upper left"); ax[0].grid(alpha=0.3); ax[0].set_xlim(0, 102)

# (b) mean and worst-case coverage vs length
x = np.arange(len(LS)); w = 0.2
for j, t in enumerate(TECHS):
    means = [agg(cov[t][Li])[0] for Li in range(len(LS))]
    worst = [agg(cov[t][Li])[2] for Li in range(len(LS))]
    ax[1].bar(x + (j - 1.5) * w, means, w, color=COL[t], alpha=0.85, label=LAB[t])
    ax[1].plot(x + (j - 1.5) * w, worst, "v", color="k", ms=6,
               label="worst-case scenario" if j == 0 else None)
ax[1].axhline(95, ls="--", color=".4", lw=1.2)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{L:g}y" for L in LS])
ax[1].set_ylim(0, 103); ax[1].set_xlabel("campaign length"); ax[1].set_ylabel("coverage [%]")
ax[1].set_title("Bars = mean over scenarios; ▾ = worst believable scenario", fontsize=10)
ax[1].legend(fontsize=8, loc="lower right"); ax[1].grid(alpha=0.3, axis="y")

plt.tight_layout(); plt.savefig("fig_ensemble.png", dpi=130); plt.close()
print("\nWrote fig_ensemble.png")
