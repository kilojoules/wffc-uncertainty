# The cost of omitting Type B in WFFC field validation

A self-contained [PyWake](https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake)
experiment quantifying a specific, decision-level failure mode in
wind-farm-flow-control (WFFC) field assessment:

> A block-bootstrap confidence interval on the measured energy benefit captures
> only the **statistical (GUM Type A)** uncertainty. If a **systematic
> differential (Type B)** measurement error is present, the bootstrap interval
> shrinks with campaign length while the systematic does not — so the
> probability of declaring a *nonexistent* benefit "statistically significant"
> **grows** as more data are collected. We quantify that growth, show the
> repair (propagate Type B alongside the bootstrap), and show which kinds of
> systematics actually matter (differential ones; common-mode errors cancel in
> the toggle design).

This is deliberately *not* a claim that the bootstrap is wrong — with no
systematic present it is correctly calibrated here (the control arms sit on
their nominal levels). It quantifies the cost of stopping there.

## What current practice reports

Toggle-test field studies overwhelmingly quantify uncertainty on the
energy-uplift metric with resampling: Fleming et al. (2019), Doekemeijer et
al. (2021), Simley et al. (2021), Fleming et al. (2021), Simley et al. (2022)
and Howland et al. (2022) all report bootstrap 95 % confidence intervals on
power/energy ratios, as catalogued in the
[IEA Wind Task 44 review](https://iea-wind.org/task44/) (§3.6), which notes
that *"when uncertainty is reported, the overwhelming majority of papers report
statistical, Type A, uncertainty."* The notable exception is Kanev (2020b),
which propagates Type-B sensor uncertainties following the GUM/IEC 61400-12-2.
The contribution here is not "Type B exists" — it is **how the omission scales
with campaign length at the go/no-go decision level.**

## Read this first: the metric, the errors, and the truth

**Metric.** The IEA Task 44 change-in-energy metric (their Eq. 12; `metrics.py`),
2-D binned by wind direction × wind speed with occurrence weights (Eq. 13):

```
ΔAEP = Σ_ij w_ij ( P̄_on,ij − P̄_off,ij ) / Σ_ij w_ij P̄_off,ij
```

The denominator is baseline power — nothing is ever normalized by the true
benefit.

**Physics (fixed, not tuned).** Two V80s 5 D apart; the upstream turbine steers
with a realistic schedule γ(wd, ws) = 25°·sign(wd−270°), tapering linearly to
zero between 11 and 13 m/s (controllers stop steering approaching rated).
Bastankhah–Porte-Agel + Jiménez at the PyWake-default wake coefficient
k = 0.0325 for **every** experiment. The resulting true benefit is
**ΔAEP = +7.2 %** for this close-spaced, ideally-steered pair — stated, not
dialed. One year of real 10-min Risø inflow, aligned sector 266–274°;
campaigns are built by resampling 2-day blocks (a disclosed idealization — see
Limitations).

**Truth.** The estimand is the deterministic clean ΔAEP on a fixed,
length-independent bin set (both control states evaluated from the wake model
on all samples — no toggle noise, no bin-mask drift). For the null experiments
we use a **placebo toggle** (control ON does nothing physically), making the
true benefit *exactly zero by construction* — no parameter tuning.

**Synthetic measurement errors** (injected end-to-end into the observed power):

- **Type A** — every logged power × (1 + ε), ε ~ N(0, 2 %) i.i.d. per sample.
  Random; averages out; exactly what the bootstrap measures.
- **Type B** — a gain (1 + b), b ~ N(0, σ_B), drawn **once per campaign** and
  constant throughout: *irreducible by collecting more of the same data* (it
  could of course be removed by better calibration — that is the other honest
  remedy). Three carriers are compared: applied to ON-periods only
  (differential), to both states (common-mode), or to the ON-period *measured
  wind speed* used for binning (the physical carrier of a yaw-dependent
  transfer-function error).

## Result 1 — which systematics matter (`carrier_typeB.py`)

![carrier comparison](fig_carrier_typeB.png)

- **Common-mode errors cancel exactly.** Eq. 12 is invariant to a gain applied
  to both states — this is the toggle design's real strength, and it means
  ordinary absolute power-measurement uncertainty (0.5–2 %) largely does *not*
  threaten the result.
- **Differential errors survive.** A 0.5 % ON-only power gain adds ±0.51 pp of
  campaign-to-campaign systematic spread (analytically dΔAEP/db = 100 + ΔAEP).
- **The physically-motivated carrier is the worst.** A 0.5 % yaw-dependent bias
  on the *measured wind speed* (bin migration) contributes ±1.17 pp — larger
  than the power-gain carrier, with the opposite wind-speed structure
  (concentrated in Region II where the power curve is steep, ~zero at rated).
  Where the systematic enters matters as much as its size.

## Result 2 — the go/no-go decision under mild Type B (`mild_typeB_decision.py`)

Placebo controller (true ΔAEP = 0 exactly). How often does each report declare
a benefit (one-sided: 95 % CI lower bound > 0)?

![false benefit declared](fig_mild_typeB_decision.png)

| campaign | σ_B=0 (control) | 0.1 % | 0.25 % | 0.5 % |
|---|---|---|---|---|
| 1 yr | 2.7 % | 2.0 % | 3.7 % | 8.0 % |
| 4 yr | 2.5 % | 3.7 % | 10.3 % | 22.3 % |
| 8 yr | 3.2 % | 4.8 % | **15.7 %** | **29.8 %** |

*(nominal 2.5 %; R = 600 campaigns per cell, Wilson bands in the figure; the
honest interval — bootstrap ⊕ propagated Type B — stays at 1–3 % everywhere.)*

The control column shows the bootstrap itself is calibrated at every length;
the growth across each row is therefore attributable to the omitted
systematic. Concrete exemplar (4 yr, σ_B = 0.25 %, true benefit zero):
measured ΔAEP +0.44 %, **bootstrap** CI [+0.02, +0.86] % → *benefit declared,
deploy*; **honest** CI [−0.21, +1.09] % → *not significant*. Same data,
opposite decision — and the failure rate is highest for the longest campaigns.

## Result 3 — coverage and the honest repair (`typeB_levels.py`)

Real controller (true ΔAEP = +7.2 %), end-to-end injection, coverage of the
true value by the bootstrap 95 % CI:

![coverage vs level](fig_typeB_levels.png)

Excess coverage loss relative to the σ_B = 0 control:

| campaign | σ_B=0.25 % | 0.5 % | 1 % |
|---|---|---|---|
| 1 yr | +1.5 pp | +5.5 pp | +19 pp |
| 4 yr | +6.5 pp | +21 pp | +51 pp |
| 8 yr | +11 pp | +36 pp | +65 pp |

The common-mode arm tracks the control at every length (cancellation,
verified). The honest interval needs only a *reasonable* Type-B prior: with
the assumed σ_B equal to truth, coverage is 93–96 % for campaigns ≥ 1 yr; a
2× overestimate is safely conservative; a 2× *under*estimate degrades to 75 %
by 8 yr — the prior matters, but it does not need to be exact.

Calibration note: at 0.5 yr the block bootstrap itself undercovers (~85 %, a
small-sample limitation visible in the control arm) — which is why effects are
reported as excess over the control. The placebo experiment, whose truth is
exact, is calibrated at all lengths.

## The visual story (`story.py`)

Raw toggle data → binning → the two uncertainties → the conclusion, in one
consistent scenario (fixed k, realistic controller, σ_B = 0.5 % ON-only):

![raw data](story_1_rawdata.png)
![binning](story_2_binning.png)
![uncertainty](story_3_uncertainty.png)
![conclusion](story_4_conclusion.png)

## What to report

1. The block-bootstrap CI, as now — it is the right Type-A estimate.
2. **Plus a propagated Type-B term** for *differential* systematics
   (yaw-dependent transfer functions, ON/OFF-asymmetric sensor paths),
   combined in quadrature — the GUM approach demonstrated here, in the spirit
   of Kanev (2020b) and of the validation framework of
   [Quick et al. (2025)](https://doi.org/10.1016/j.renene.2024.122028)
   (aleatoric variability lives in the data; epistemic uncertainty is
   propagated and reported).
3. Or, equivalently: invest in calibrating the differential channels away —
   the common-mode result shows the toggle design already protects against
   everything else.

## Limitations (disclosed idealizations)

- **One base year, block-resampled.** No interannual variability; the weather
  process satisfies the bootstrap's exchangeability assumptions by
  construction. Quantitative length-scalings are conditional on this; the
  qualitative mechanism (Type A shrinks, a constant systematic does not) is
  not.
- **b is constant for the whole campaign.** A systematic with a ~1-yr
  correlation time would partially average down; the monotone growth assumes
  the error persists (e.g., an uncorrected transfer function, which does).
- Single site, layout (2 turbines, 5 D), wake model, and toggle scheme
  (every 7 waked-sector samples ≈ hours-to-days of calendar time). The
  sensitivity dΔAEP/db = 100 + ΔAEP is exact for any of these choices; the
  weather-driven Type-A schedule is not.
- σ_A = 2 % i.i.d. is synthetic; real 10-min scatter is larger and correlated,
  which would slow (not remove) the Type-A shrinkage.

## Run it

```bash
pip install py_wake numpy scipy matplotlib      # tested with py_wake 2.6.7
python carrier_typeB.py        # which systematics matter (common-mode cancels)
python mild_typeB_decision.py  # false 'benefit declared' rate vs campaign length
python typeB_levels.py         # coverage, excess-over-control, misspecification
python story.py                # the 4-figure walkthrough
python absolute_view.py        # un-normalized (kW) per-wind-speed view
```

Shared modules: `pywake_model.py` (deterministic lookup, realistic yaw
schedule, fixed k), `metrics.py` (IEA Eq. 12 + vectorized B=1000 block
bootstrap + fixed-mask estimand), `campaigns.py` (block-resampled campaigns,
collision-free seeds, end-to-end error injection). Each script runs on a
laptop (<1 GB RAM, single CPU, minutes to ~half an hour).

`main_experiment.py`, `report.py`, `bootstrap_vs_typeB.py`,
`typeB_measurement.py`, `area_recipe.py` are earlier iterations kept for
provenance (an atmospheric-parameter Type-B framing and pre-rework metrics),
superseded by the above.

## References

- IEA Wind Task 44, *Review and Best Practices for Wind Farm Flow Control
  Field Assessment* — toggle tests, the ΔAEP metric (Eq. 12), bootstrap
  practice and the Type-A/Type-B discussion (§3.6, §4.2, §4.7).
- J. Quick et al., *Wind speed vertical extrapolation model validation under
  uncertainty*, Renewable Energy 240 (2025) 122028 —
  <https://doi.org/10.1016/j.renene.2024.122028>.
- S. Kanev (2020b), TNO report — GUM/IEC 61400-12-2 Type-B propagation for
  WFFC AEP assessment (the exception that proves the rule).
- Fleming et al. (2019), Doekemeijer et al. (2021), Simley et al. (2021, 2022),
  Fleming et al. (2021), Howland et al. (2022) — field campaigns reporting
  bootstrap CIs on the uplift (see IEA Task 44 §3.6 for the catalogue).
