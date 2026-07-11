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
power/energy ratios. The forthcoming
[IEA Wind Task 44](https://iea-wind.org/task44/) field-assessment review (in
preparation, targeted at *Wind Energy Science*) catalogues this practice; the
pattern is that when uncertainty is reported at all, the large majority of
papers report only statistical (Type A) uncertainty. The notable exception is
Kanev (2020), whose TNO validation-methodology report propagates Type-B sensor
uncertainties following the GUM / IEC 61400-12-2.
The underlying mechanism — a frozen systematic eventually swamping a sampling
variance that shrinks like 1/√N — is an elementary GUM corollary, and the repair
is standard GUM/Kanev propagation, not a new estimator. The contribution here is
not "Type B exists" or a novel phenomenon; it is the **WFFC-specific
quantification of how the omission scales with campaign length at the go/no-go
decision level, a comparison of which physical carriers actually survive the
toggle design (Result 1), the sensitivity coefficients needed to repair each,
and the mean *and worst-case* reliability of each technique across a believable
scenario ensemble for the sub-2-year campaigns that dominate the field
(Result 4).**

## Read this first: the metric, the errors, and the truth

**Metric.** The IEA Task 44 change-in-energy metric (as formalized in the
in-preparation Task 44 field-assessment review; `metrics.py`), 2-D binned by
wind direction × wind speed with occurrence weights:

```
ΔAEP = Σ_ij w_ij ( P̄_on,ij − P̄_off,ij ) / Σ_ij w_ij P̄_off,ij
```

The denominator is baseline power — nothing is ever normalized by the true
benefit.

**Physics (fixed, not tuned).** Two V80s 5 D apart; the upstream turbine steers
with a realistic schedule γ(wd, ws) = 25°·sign(wd−270°), tapering linearly to
zero between 11 and 13 m/s (controllers stop steering approaching rated).
Bastankhah–Porte-Agel + Jiménez at the PyWake-default wake coefficient
k = 0.0325 for **every** experiment (not tuned per-experiment). The resulting
true benefit is a **waked-sector energy uplift of ΔAEP = +7.2 %** for this
close-spaced, ideally-steered pair — i.e. +7.2 % *of the baseline energy in the
266–274° sector* (the canonical IEA Task 44 change-in-energy quantity), which is ~3 % of annual
hours, so the implied full-rose annual gain is ≈0.2 %; "ΔAEP" here is the
sector-conditional metric, not a farm-annual number. The magnitude is conditional
on the fixed k: sweeping k across PyWake's plausible range (0.018–0.052) moves it
from +18 % to +0.2 %, so we pin k at the model default rather than dial a
headline. **Crucially, none of the Type-B conclusions depend on this magnitude:**
the ON-only sensitivity dΔAEP/db = 100 + ΔAEP varies only from 100 to ~118 across
that whole k range, and the placebo decision experiment (Result 2) has true
ΔAEP = 0 exactly — so the false-positive mechanism is essentially independent of
the true benefit's size. One year of real 10-min Risø inflow, aligned sector
266–274°; campaigns are built by resampling 2-day blocks (a disclosed idealization
— see Limitations).

**Truth.** The estimand is the deterministic clean ΔAEP on a fixed,
length-independent bin set (both control states evaluated from the wake model
on all samples — no toggle noise, no bin-mask drift). For the null experiments
we use a **placebo toggle** (control ON does nothing physically), making the
true benefit *exactly zero by construction* — no parameter tuning. (Fine print:
the campaign *estimator* is toggle-split — it averages ON-only and OFF-only
samples per bin — so it differs from this all-sample estimand by a small
within-bin composition offset, +0.02 pp on the real controller and +0.05 pp on
the placebo, that decays with campaign length. This offset is a fixed property
of the estimator, not of the systematic; it is carried in the σ_B = 0 control
column, which is why the reported effects are always *excess over that control*
rather than raw rates.)

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

- **Common-mode errors cancel exactly.** The change-in-energy metric is
  invariant to a gain applied to both states — this is the toggle design's real
  strength, and it means ordinary absolute power-measurement uncertainty
  (0.5–2 %) largely does *not* threaten the result.
- **Differential errors survive.** A 0.5 % ON-only power gain adds ±0.50 pp of
  campaign-to-campaign systematic spread (analytically dΔAEP/db = 100 + ΔAEP).
- **The physically-motivated carrier is the worst.** A 0.5 % yaw-dependent bias
  on the *measured wind speed* (bin migration) contributes ±1.19 pp — larger
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
systematic. The growth is monotone but **bounded**: as L → ∞ the bootstrap
half-width → 0, so this one-sided "benefit declared" rate approaches
P(b > 0) = **50 %** (and the two-sided "change declared" rate → 100 %) for a
symmetric Type-B prior — the tables show the approach to that ceiling, not an
unbounded blow-up. Concrete exemplar (4 yr, σ_B = 0.25 %, true benefit zero):
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
verified).

**The honest repair works — with two honesty caveats.** With the assumed σ_B
equal to truth, coverage is 93–96 % for campaigns ≥ 1 yr; a 2× overestimate is
safely conservative; a 2× *under*estimate degrades to ~75 % by 8 yr — the prior
matters, but it need not be exact. Two things this does *not* claim:

1. **The exactly-specified (1×) case is an analytic identity, not an empirical
   win.** For an ON-only gain the estimate shifts by exactly (dΔAEP/db)·b with
   b ~ N(0, σ_B), so adding z·(dΔAEP/db)·σ_B in quadrature with a calibrated
   bootstrap returns 95 % coverage *by construction* wherever the bootstrap is
   itself calibrated. The empirical content is the misspecification sweep, not
   the 1× column. (The sensitivity is taken per-campaign from the *measured*
   ΔAEP, not the oracle truth; the two agree to <0.4 pp.)
2. **"Coverage" here is prior-averaged (GUM-marginal).** It is the hit rate over
   the *ensemble* of campaigns drawn from the Type-B prior — not a guarantee for
   one deployed campaign, whose realized bias is a fixed unknown. The honest
   interval widens the band to the right size *on average*; it does not identify
   this campaign's particular offset.

**The repair is carrier-specific — and the naive version fails on the worst
carrier.** The term above uses the ON-only power-gain sensitivity
dΔAEP/db = 100 + ΔAEP ≈ 107. The ws (bin-migration) carrier — which Result 1
flags as the *worst* — has **no** such closed form; propagating a unit ws bias
through the metric gives a sensitivity S_ws ≈ 221, **2.1× larger**. On the ws
carrier at σ_B = 0.5 % (third panel of the figure):

| campaign | bootstrap only | honest, *naive* 107 term | honest, correct 221 term |
|---|---|---|---|
| 1 yr | 72 % | 82 % | 95 % |
| 4 yr | 39 % | 68 % | 95 % |
| 8 yr | **32 %** | **65 %** | **94 %** |

Reusing the ON-only term on a ws systematic under-corrects and leaves the same
length-scaling collapse (65 % at 8 yr); propagating the ws carrier's *own*
(numerically estimated) sensitivity restores coverage (94 %). The recipe is
therefore: propagate the *right* sensitivity for the *actual* carrier — a single
one-size term is not enough.

Calibration note: at 0.5 yr the block bootstrap itself undercovers (~85 %, a
small-sample limitation visible in the control arm) — which is why effects are
reported as excess over the control, and why even the correctly-repaired ws arm
reads 89 % at 0.5 yr. The placebo experiment, whose truth is exact, is
calibrated at all lengths.

## Result 4 — across a believable ensemble, in the < 2-year field regime (`ensemble.py`)

Results 1–3 each fix one scenario; a real analyst does not know theirs, and the
typical toggle campaign runs **under two years**. So the decision-relevant
question is: over an ensemble of believable field campaigns — and in the *worst*
believable case — how well does each technique do? We draw **100 independent
scenarios** and, at the lengths that actually occur (0.5–2 yr), compare three
techniques — plus a misattribution stress arm — *given the true σ_B* (so only
the technique differs, not the prior — σ_B misspecification is Result 3's job):

- true benefit from the wake coefficient k ~ U(0.025, 0.045) → ΔAEP ≈ +2…+12 %,
  with **25 % placebo** scenarios (no real benefit, true ΔAEP = 0);
- systematic σ_B ~ U(0, 0.5 %) (0 = well-calibrated; 0.5 % ≈ a believable upper
  end for the *post-calibration residual* of a differential error. Raw
  uncorrected yaw-dependent nacelle-anemometer biases are multi-percent, so
  this band assumes a yawed transfer-function correction has been applied —
  "worst case" below means worst *within this prior band*. A wider band makes
  bootstrap-only strictly worse and leaves carrier-aware calibrated, so the
  ranking is insensitive to the upper end);
- carrier ~ {ON-only power 40 %, ws/yaw-transfer 40 %, common-mode 20 %};
- scatter σ_A ~ U(1.5, 3.5 %).

The techniques: **bootstrap only** (current practice); **honest one-size** (add
z·(100+ΔAEP)·σ_B to *every* campaign — the naive reading of "report a Type-B
term"); **honest carrier-aware** (use the *actual* carrier's sensitivity —
100+ΔAEP for ON-only power, the propagated bin-migration sensitivity for ws, ~0
for common-mode); and a **misattribution stress arm** — carrier-aware after the
single most dangerous identification error, a differential ws systematic
misread as *common-mode* (S = 0). The stress arm is there because carrier-aware's
recommendation is **conditional on correctly identifying the carrier**:
mis-*sizing* σ_B degrades gracefully (Result 3), but calling a differential
systematic "common-mode" silently reverts the interval to the bare bootstrap.
Declaring a systematic common-mode is the unsafe direction — when in doubt,
treat it as differential.

![ensemble](fig_ensemble.png)

**Coverage of the true benefit (target 95 %), campaigns < 2 yr** (mean ± 2·MC-SE,
scenario-clustered):

| technique | mean | near-worst (5th pct) | worst case |
|---|---|---|---|
| bootstrap only (current practice) | 85 ± 2 % | 57 % | **40 %** |
| honest, one-size (100+ΔAEP) term | 90 ± 1 % | 74 % | 64 % |
| honest, carrier-aware (recommended) | 93 ± 1 % | 84 % | 72 % |
| carrier-aware, *ws misread as common* | 87 ± 2 % | 57 % | 40 % |

**False "benefit declared" on the placebo scenarios (target ≤ 2.5 %):**

| technique | pooled rate (Wilson 95 %) | worst case |
|---|---|---|
| bootstrap only | 6.3 % [5.8, 6.9] | **35 %** |
| honest, one-size | 4.0 % [3.6, 4.5] | 18.8 % |
| honest, carrier-aware | 2.2 % [1.8, 2.5] | 6.2 % |
| carrier-aware, *ws misread as common* | 5.6 % [5.1, 6.1] | 35 % |

Read the worst-case column against its *null reference*: a **perfectly
calibrated** technique still shows a worst cell of 7.5 % (median; 5–95 % range
6.2–10.0 %), because that column is a max over 93 Binomial(80, 2.5 %) cells.
Carrier-aware's 6.2 % is *below* the null median — statistically
indistinguishable from perfect calibration — while one-size (18.8 %) and
bootstrap (35 %) are impossible under the null (p < 10⁻⁸): the ordering is
signal, the 6.2 % itself is not a target violation.

Four things the ensemble makes plain:

1. **Current practice is not just imperfect on average, it is unreliable in the
   tail — and gets worse with data.** Bootstrap-only averages 85 % coverage and a
   6 % false-deploy rate, but its *worst* believable scenario is 40 % coverage /
   35 % false-deploy, and (right panel) its worst case *degrades* with length: a
   2-year campaign is more likely to deploy a nonexistent benefit than a half-year
   one.
2. **A one-size Type-B term helps the average but not the worst case.** It sizes
   the ON-only carrier correctly and over-covers common-mode, so the mean looks
   healthy — but it still under-sizes the ws/yaw carrier, so the tail stays bad
   (64 % coverage / 19 % false-deploy).
3. **The carrier-aware interval is the only one whose worst case is bounded by
   something other than Type B.** Its floor is the block bootstrap's own
   small-sample undercoverage at 0.5 yr (~72 %) — a floor that in a real
   *contiguous* sub-year campaign would sit lower still for every technique,
   because of the seasonal-representativeness error the resampling design
   cannot see (Limitations). By 1–2 yr — the heart of the field regime — its
   worst case is 82–88 % and its pooled false-deploy rate is 2.2 %
   [1.8, 2.5], holding the nominal 2.5 %. Unlike the bootstrap, it *improves*
   with data.
4. **Carrier identification is the load-bearing input.** The stress arm — a
   differential ws systematic misread as common-mode — hands back nearly the
   entire bootstrap pathology: 87 % mean / 40 % worst coverage, 35 % worst
   false-deploy. Mis-*sizing* σ_B degrades gracefully (Result 3);
   mis-*identifying* the carrier does not, and the dangerous direction is
   specifically declaring a differential channel "common-mode" (S = 0, i.e.
   silently adding nothing).

Two honest scope notes. *(a)* The ws carrier-aware sensitivity is computed once
per scenario from the *clean* powers (a noise-free finite difference through
the metric) rather than re-estimated per campaign — knowledge of the channel,
not of the realized bias. It is **not** a scale-free physical constant: the
bin-migration response is a staircase in the bias, and a secant taken below the
~0.3 % discreteness knee understates the slope by up to ~2×. It is therefore
evaluated on the plateau, at scale max(2σ_B, 0.3 %), giving **~203–241 across
scenarios** (median ~223). An earlier draft evaluated the secant at σ_B itself,
which under-sized the term for small-σ_B ws scenarios — an error *against* the
aware technique, masked in the tables (≤3 pp) only because the bootstrap width
dominates the quadrature at < 2 yr. *(b)* "One-size" here means the *smaller* ON-only
coefficient (~107) applied everywhere — faithful to the naive reading of this
repo's own recommendation, and the specific error being warned against. A
*conservative* one-size that used the largest sensitivity (~240) for every
campaign would also bound worst-case coverage — but by over-widening on the
ON-only and common-mode majority, i.e. trading away power to detect real
benefits. Carrier-aware's edge is *right-sizing* (not over-widening), which the
mean/power side of the table would show more directly than coverage alone.

(100 scenarios × 80 campaigns × 3 lengths, shared B = 400 bootstrap — the
smallest MC budget in the repo, hence the explicit uncertainty statements:
per-cell granularity is 100/80 = 1.25 pp, and the single-scenario "worst case"
is a min/max over 300 (93 placebo) binomial cells, biased outward by noise
alone by several pp. Read the 5th-pct column (coverage) and the pooled Wilson
rate (placebo) as the robust tail statistics; the worst-case column is there
for the mechanism, not the digit. Every distributional choice above is
disclosed and tunable at the top of `ensemble.py`.)

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
   of Kanev (2020) and of the validation framework of
   [Quick et al. (2025)](https://doi.org/10.1016/j.renene.2024.122028)
   (aleatoric variability lives in the data; epistemic uncertainty is
   propagated and reported). **Use the sensitivity of the *actual* carrier, not
   a one-size coefficient:** for an ON-only power gain it is dΔAEP/db = 100 +
   ΔAEP; for a yaw-dependent wind-speed/transfer-function bias it is ~2× larger
   and has no closed form — estimate it by pushing a unit bias through the
   metric pipeline (`ws_carrier_sensitivity` in `typeB_levels.py`). The
   bin-migration response is a *staircase* in the bias, so evaluate the finite
   difference on its plateau — at a scale of max(2σ_B, ~0.3 %) — never at a
   tiny scale, where the secant understates the slope by up to ~2×. Result 3
   shows the naive ON-only term leaves the ws carrier under-covered (65 % at
   8 yr); its own sensitivity restores it (94 %).
3. Or, equivalently: invest in calibrating the differential channels away —
   the common-mode result shows the toggle design already protects against
   everything else.

## Limitations (disclosed idealizations)

- **One base year, block-resampled.** No interannual variability; the weather
  process satisfies the bootstrap's exchangeability assumptions by
  construction. Sub-year simulated campaigns are additionally
  *season-scrambled*: uniform block draws give every 0.5-yr campaign full-year
  climatology (and a 2-yr campaign reuses the same year twice), so a real
  *contiguous* 6–12-month campaign carries a seasonal representativeness error
  (order ±1–3.5 pp in ΔAEP over this site's k range) that none of the
  techniques — bootstrap or honest — can see. Field sub-year coverage is
  therefore lower than simulated for *all* of them. Quantitative
  length-scalings are conditional on this; the qualitative mechanism (Type A
  shrinks, a constant systematic does not) is not.
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
python typeB_levels.py         # coverage, excess-over-control, misspecification, ws repair
python ensemble.py             # mean & worst-case per technique over believable scenarios
python story.py                # the 4-figure walkthrough
python absolute_view.py        # un-normalized (kW) per-wind-speed view
```

Shared modules: `pywake_model.py` (deterministic lookup, realistic yaw
schedule, fixed k), `metrics.py` (IEA Task 44 change-in-energy metric +
vectorized B=1000 block bootstrap + fixed-mask estimand), `campaigns.py` (block-resampled campaigns,
collision-free seeds, end-to-end error injection). Each script runs on a
laptop (<1 GB RAM, single CPU, minutes to ~half an hour).

`main_experiment.py`, `report.py`, `bootstrap_vs_typeB.py`,
`typeB_measurement.py`, `area_recipe.py` are earlier iterations kept for
provenance (an atmospheric-parameter Type-B framing and pre-rework metrics),
superseded by the above.

## References

- IEA Wind Task 44, *Review and Best Practices for Wind Farm Flow Control
  Field Assessment* — the change-in-energy metric, bootstrap practice, and the
  Type-A/Type-B discussion drawn on here. **In preparation** (targeted at *Wind
  Energy Science*); not yet a citable publication. Task homepage:
  <https://iea-wind.org/task44/>.
- J. Meyers, C. Bottasso, K. Dykes, P. Fleming, P. Gebraad, G. Giebel, T.
  Göçmen, J.-W. van Wingerden, *Wind farm flow control: prospects and
  challenges*, Wind Energy Science 7 (2022) 2271–2306 —
  <https://doi.org/10.5194/wes-7-2271-2022> (the published Task 44
  prospects-and-challenges review; the field-assessment review above is a
  separate, later document).
- J. Quick et al., *Wind speed vertical extrapolation model validation under
  uncertainty*, Renewable Energy 240 (2025) 122028 —
  <https://doi.org/10.1016/j.renene.2024.122028>.
- S. Kanev, *AWC validation methodology*, TNO report TNO 2020 R11300, Petten,
  Aug 2020 — GUM / IEC 61400-12-2 (Category A/B) uncertainty propagation for
  WFFC AEP assessment (the exception that proves the rule);
  <https://resolver.tno.nl/uuid:fdae4c94-fbcc-4337-b49f-5a39c93ef2cf>. (Cited
  as "Kanev (2020b)" elsewhere; the "a/b" suffixes are local labels — the
  companion *Renewable Energy* 146 (2020) 9–15 dynamic-wake-steering paper is
  the other 2020 Kanev item.)
- Fleming et al. (2019), Doekemeijer et al. (2021), Simley et al. (2021, 2022),
  Fleming et al. (2021), Howland et al. (2022) — field campaigns reporting
  bootstrap CIs on the uplift.
