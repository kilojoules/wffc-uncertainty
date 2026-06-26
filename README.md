# More measurement, more false confidence — Type B in WFFC field validation

A small, self-contained [PyWake](https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake)
experiment with a counterintuitive result:

> **In wind-farm-flow-control (WFFC) field tests, collecting more data can make
> the standard conclusion *more* wrong, not less.** A longer, better-funded
> campaign is *more* likely to declare a wake-steering "benefit" that isn't real —
> because the universally-used bootstrap confidence interval shrinks toward zero
> while a systematic (GUM Type B) measurement error stays fixed and unseen.

## Read this first: benefit, the two errors, and the "true" value

**What "benefit" means.** We use the change-in-energy metric defined in **IEA Wind
Task 44 WP2 (Eq. 12)** — the standard field-assessment metric. Bin the data by wind
**direction** (i) and wind **speed** (j); in each bin average the turbine-pair power
during flow-control (ON) and baseline (OFF) periods; difference them and
occurrence-weight (their Eq. 13, `w_ij = N_ij / N`):

```
ΔAEP  =  Σ_ij w_ij ( P_on_ij − P_off_ij )  /  Σ_ij w_ij P_off_ij        (IEA Eq. 12)
```

A 1–3 % uplift is typical. Note what the denominator is: the **occurrence-weighted
baseline power** — so the percent is divided by *baseline power*, **not** by the
true benefit. (Implementation in `metrics.py`.)

**The two synthetic measurement errors.** The wake model is deterministic, so the
true ON and OFF power at every 10-min step is known exactly. We then corrupt what
the analyst *records*, the way real instruments do:

- **Type A — random, reducible.** Every logged power is multiplied by `(1 + ε)`,
  with `ε ~ N(0, 2%)` drawn fresh each sample. Plain sensor noise: it averages out
  with more data, and it is exactly what the bootstrap captures.
- **Type B — systematic, irreducible.** During ON (yawed) periods the logged power
  carries a *fixed* unknown gain offset `(1 + b)`, with `b ~ N(0, σ_B)` drawn
  **once per campaign** and identical at every timestep — a stand-in for an
  uncorrected yaw-dependent metering / transfer-function error (an IEA Task 44
  source). Because it is the same at every step, no amount of averaging or
  resampling removes it, and the bootstrap never sees it. (σ_B is the knob we sweep;
  0.25–0.5 % is "mild".)

**The "true benefit" is a check-target, not a normalizer.** Because the model is
deterministic we compute the true benefit once from the *clean* (error-free) power.
It is used only to ask *"did the reported interval contain it?"* — it never divides
or rescales anything. The percent benefit is `gain ÷ baseline power`; the absolute
(kW) view tells the identical story with no percent at all, so the normalization is
cosmetic, not load-bearing.

## The story in four figures

**1 · What a toggle test records.** The controller is switched ON/OFF every ~70 min;
power is logged with realistic sensor errors. The power swings track the *weather*,
not the control — the benefit is nowhere to be seen in the raw signal.

![raw data](story_1_rawdata.png)

**2 · How the benefit is computed.** Bin by wind speed and direction, average ON vs
OFF in each bin, difference, occurrence-weight into the IEA ΔAEP (Eq. 12). The ON/OFF
clouds almost completely overlap; the per-bin gains are small and sign-varying. This
campaign reads **ΔAEP = +2.14 % of baseline power**, while the known true value is
**+1.50 %** — the ~0.6-point gap is the systematic Type-B bias (this campaign drew
a ≈ +1σ offset). A bootstrap of this data can only "see" the random scatter, not
that offset.

![binning and power gain](story_2_binning.png)

**3 · The number has two uncertainties.** *Type A* (random sensor noise + finite
samples) is what the bootstrap measures — it shrinks ∝ 1/√N. *Type B* (a
systematic, yaw-correlated calibration/transfer error) is a **fixed floor** that
more data cannot reduce. The bootstrap reports only Type A.

![two uncertainties](story_3_uncertainty.png)

**4 · The consequence.** How often the reported 95 % interval actually contains the
known true benefit (the clean-model value from "read this first"). With no Type B
the bootstrap is fine; with even a mild systematic its coverage **falls as the
campaign grows** (it tightens around a biased value), while the honest interval
(bootstrap ⊕ propagated Type B) holds ~95 %.

![conclusion](story_4_conclusion.png)

## Why this happens

The bootstrap estimates only Type A, so as the campaign grows its CI tightens
∝ 1/√N toward **zero width around a point that still carries the fixed Type-B
bias.** A tighter interval around a biased estimate excludes the truth *more* often.
More data buys precision about the wrong number — and the best-funded studies are
the most confidently wrong. (Type B is irreducible by construction: the systematic
offset is identical at every timestep, so neither averaging nor resampling the data
can reveal it.)

## The experiment

- **Deterministic truth.** Two V80 turbines (D = 80 m) 5 D apart; upstream yaws 25°
  for control. Bastankhah–Porte-Agel + Jiménez deflection at a **fixed** wake
  coefficient, so the true benefit is known exactly — no hidden parameter games.
  One year of real 10-min inflow, aligned waked sector (266–274°); each campaign
  draws its own weather by 2-day block resampling, extended to 8 years.
- **Synthetic measurement errors** (Type A random 2 %, Type B systematic σ_B once
  per campaign) are added to the clean power as defined in *Read this first*,
  following GUM / [Quick et al. 2025](https://doi.org/10.1016/j.renene.2024.122028).
- **Reported uncertainty** is the standard block-bootstrap 95 % CI (Type A); the
  "honest" interval adds the propagated Type B in quadrature.

## Sharper cut: the go/no-go decision under *mild* Type B

Set the true benefit to ≈ 0 (a marginal controller — realistic) and ask how often
each method *falsely* declares a statistically significant benefit (95 % interval
excludes zero → "deploy / publish"):

![false-positive significance](fig_mild_typeB_decision.png)

| campaign | bootstrap σ_B=0 | σ_B=0.25 % | σ_B=0.5 % | honest |
|---|---|---|---|---|
| 1 yr | 6 % | 8 % | 11 % | ~6 % |
| 4 yr | 8 % | 15 % | 31 % | ~7 % |
| 8 yr | 10 % | **22 %** | **42 %** | ~6 % |

A concrete 4-year campaign at σ_B = 0.25 %: measured ΔAEP −0.74 %, **bootstrap** CI
[−1.39, −0.08] % → *"significant — a real change, act on it"*; **honest** CI
[−1.56, +0.08] % → *"not significant."* Same data, opposite decision.

For WFFC this is not a footnote: realistic systematics (0.5–2 %) give Type-B floors
of **1–4 percentage points — comparable to or larger than the benefit itself
(~1 pp)**, so Type B can dominate the signal. Sweeping the level (`typeB_levels.py`),
the bootstrap's coverage of the true ΔAEP collapses accordingly:

| campaign | σ_B=0 | 0.5 % | 1 % | 2 % |
|---|---|---|---|---|
| 1 yr | 95 % | 88 % | 72 % | 44 % |
| 8 yr | 95 % | 51 % | 31 % | 17 % |

*(honest interval stays ~95 % throughout; σ_B=0 confirms the bootstrap is calibrated
when there is no Type B.)*

## The benefit, un-normalized

Reporting as a percent of baseline hides structure; in absolute kW
(`absolute_view.py`) the net benefit is a delicate cancellation of large,
sign-flipping per-wind-speed contributions, and the systematic Type-B error
(`b·P_on`) is **largest in the high-wind bins where the wake effect has vanished.**

![absolute view](fig_absolute_view.png)

## What to report instead

Report the benefit with an uncertainty that **includes Type B** — propagate the
systematic sensor/model uncertainties through the wake response, on top of the
bootstrap. A clean way is the **area metric** (the area between the on/off power
CDFs, [Quick et al. 2025](https://doi.org/10.1016/j.renene.2024.122028)): the
aleatoric scatter lives inside the CDFs, and you report the propagated Type-B
spread. The bootstrap alone answers *"how precisely did I pin down this campaign's
mean?"* (→ 0 with data) — **not** *"how uncertain is the benefit?"*

## Run it

```bash
pip install py_wake numpy scipy matplotlib      # tested with py_wake 2.6.7
python story.py                # the 4-figure visual walkthrough
python mild_typeB_decision.py  # false-positive significance under mild Type B
python typeB_levels.py         # coverage vs campaign length × Type-B level
python absolute_view.py        # un-normalized (kW) per-wind-speed + raw-spread view
```

All experiments share `metrics.py`, which implements the IEA Task 44 change-in-energy
metric (ΔAEP, Eq. 12).

Each script builds a small PyWake power lookup once (a few seconds, <300 MB RAM,
single CPU) and runs Monte-Carlo experiments on top.

## Repository contents

| file | purpose |
|---|---|
| `pywake_model.py` | builds the deterministic PyWake power lookup (the engine) |
| `metrics.py` | the IEA Task 44 change-in-energy metric (ΔAEP, Eq. 12) + block bootstrap — shared by all experiments |
| `story.py` | **the 4-figure narrative**: raw data → binning → uncertainty → conclusion |
| `mild_typeB_decision.py` | false "significant benefit" rate vs campaign length under mild Type B |
| `typeB_levels.py` | coverage of true ΔAEP vs campaign length across Type-B levels |
| `absolute_view.py` | un-normalized (kW) view: per-wind-speed decomposition (the per-bin terms of ΔAEP) + raw spread |
| `main_experiment.py`, `report.py`, `bootstrap_vs_typeB.py`, `typeB_measurement.py`, `area_recipe.py` | earlier exploration — `main_experiment`/`report`/`bootstrap_vs_typeB` modelled Type B as an uncertain *atmospheric* parameter (largely aleatoric/reducible, **superseded**); `typeB_measurement`/`area_recipe` predate the shared `metrics.py` and the block-resampled calibration. Kept for provenance |

## References

- J. Quick et al., *Wind speed vertical extrapolation model validation under
  uncertainty*, Renewable Energy 240 (2025) 122028 —
  <https://doi.org/10.1016/j.renene.2024.122028>
- IEA Wind Task 44, *Review and Best Practices for Wind Farm Flow Control Field
  Assessment* (toggle tests, energy-ratio metrics, bootstrap practice, Type A/B).
