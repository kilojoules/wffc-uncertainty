# More measurement, more false confidence — Type B in WFFC field validation

A small, self-contained [PyWake](https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake)
experiment with a counterintuitive result:

> **In wind-farm-flow-control (WFFC) field tests, collecting more data can make
> the standard conclusion *more* wrong, not less.** A longer, better-funded
> campaign is *more* likely to declare a wake-steering "benefit" that isn't real —
> because the universally-used bootstrap confidence interval shrinks toward zero
> while a systematic (GUM Type B) measurement error stays fixed and unseen.

## The story in four figures

**1 · What a toggle test records.** The controller is switched ON/OFF every ~70 min;
power is logged with realistic sensor errors. The power swings track the *weather*,
not the control — the benefit is nowhere to be seen in the raw signal.

![raw data](story_1_rawdata.png)

**2 · How the benefit is computed.** Bin the data by wind speed, average ON vs OFF
in each bin, take the difference, and occurrence-weight it into one number. The
ON/OFF clouds almost completely overlap; the per-bin gains are small and
sign-varying; the aggregate here reads **+2.44 %** (the true value is **+1.69 %** —
the gap is a systematic measurement bias).

![binning and power gain](story_2_binning.png)

**3 · The number has two uncertainties.** *Type A* (random sensor noise + finite
samples) is what the bootstrap measures — it shrinks ∝ 1/√N. *Type B* (a
systematic, yaw-correlated calibration/transfer error) is a **fixed floor** that
more data cannot reduce. The bootstrap reports only Type A.

![two uncertainties](story_3_uncertainty.png)

**4 · The consequence.** Coverage of the *true* benefit by the reported 95 %
interval. With no Type B the bootstrap is fine; with even a mild systematic its
coverage **falls as the campaign grows** (it tightens around a biased value),
while the honest interval (bootstrap ⊕ propagated Type B) holds ~95 %.

![conclusion](story_4_conclusion.png)

## Why this happens

A toggle test measures the benefit directly; its reported uncertainty is almost
always a block-bootstrap CI. But uncertainty comes in two kinds (GUM):

- **Type A** — statistical (sensor noise, finite samples). The bootstrap estimates
  it, and it **shrinks ∝ 1/√N**.
- **Type B** — systematic (calibration offsets, yaw-dependent transfer functions,
  wake-model form). It **does not shrink with more measurement**, and the bootstrap
  — which resamples the *observed* data — is **structurally blind to it**.

So as the campaign grows, the bootstrap CI tightens toward **zero width around a
point that still carries the fixed Type-B bias.** A tighter interval around a
biased estimate excludes the truth *more* often. More data buys precision about the
wrong number — and the best-funded studies are the most confidently wrong.

## The experiment

- **Deterministic truth.** Two V80 turbines (D = 80 m) 5 D apart; upstream yaws 25°
  for control. Bastankhah–Porte-Agel + Jiménez deflection at a **fixed** wake
  coefficient, so the true benefit is known exactly — no hidden parameter games.
  One year of real 10-min inflow, aligned waked sector (266–274°); each campaign
  draws its own weather by 2-day block resampling, extended to 8 years.
- **Synthetic measurement errors** are added to what the analyst sees (as GUM /
  [Quick et al. 2025](https://doi.org/10.1016/j.renene.2024.122028) prescribe):
  **Type A** = random per-sample power noise (2 %); **Type B** = a systematic,
  yaw-correlated bias `b ~ N(0, σ_B)` drawn once per campaign (constant ⇒ invisible
  to resampling ⇒ irreducible).

## Sharper cut: the go/no-go decision under *mild* Type B

Set the true benefit to ≈ 0 (a marginal controller — realistic) and ask how often
each method *falsely* declares a statistically significant benefit (95 % interval
excludes zero → "deploy / publish"):

![false-positive significance](fig_mild_typeB_decision.png)

| campaign | bootstrap σ_B=0 | σ_B=0.25 % | σ_B=0.5 % | honest |
|---|---|---|---|---|
| 1 yr | 7 % | 9 % | 13 % | ~6 % |
| 4 yr | 6 % | 11 % | 25 % | ~5 % |
| 8 yr | 8 % | **17 %** | **36 %** | ~5 % |

A concrete 4-year campaign at σ_B = 0.25 %: measured +0.68 %, **bootstrap** CI
[+0.03, +1.32] % → *"significant, deploy"*; **honest** CI [−0.13, +1.48] % → *"not
significant."* Same data, opposite decision.

For WFFC this is not a footnote: realistic systematics (0.5–2 %) give Type-B floors
of **1–4 percentage points — comparable to or larger than the benefit itself
(~1 pp)**, so Type B can dominate the signal.

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
python typeB_measurement.py    # coverage at one Type-B level; Type-A vs Type-B half-widths
python absolute_view.py        # un-normalized (kW) per-wind-speed + raw-spread view
python area_recipe.py          # the recommended report: area-benefit ± Type-B
```

Each script builds a small PyWake power lookup once (a few seconds, <300 MB RAM,
single CPU) and runs Monte-Carlo experiments on top.

## Repository contents

| file | purpose |
|---|---|
| `pywake_model.py` | builds the deterministic PyWake power lookup (the engine) |
| `story.py` | **the 4-figure narrative**: raw data → binning → uncertainty → conclusion |
| `mild_typeB_decision.py` | false "significant benefit" rate vs campaign length under mild Type B |
| `typeB_levels.py` | coverage vs campaign length across Type-B levels |
| `typeB_measurement.py` | coverage at one Type-B level; Type-A vs Type-B half-widths |
| `absolute_view.py` | un-normalized (kW) view: per-wind-speed decomposition + raw spread |
| `area_recipe.py` | the recommended reporting recipe: benefit = area between on/off CDFs, ± propagated Type B |
| `main_experiment.py`, `report.py`, `bootstrap_vs_typeB.py` | earlier exploration that modelled Type B as an uncertain *atmospheric* (wake) parameter; kept for provenance but **superseded** — atmospheric variability is largely aleatoric/reducible, whereas genuine Type B is the systematic *measurement* error modelled above |

## References

- J. Quick et al., *Wind speed vertical extrapolation model validation under
  uncertainty*, Renewable Energy 240 (2025) 122028 —
  <https://doi.org/10.1016/j.renene.2024.122028>
- IEA Wind Task 44, *Review and Best Practices for Wind Farm Flow Control Field
  Assessment* (toggle tests, energy-ratio metrics, bootstrap practice, Type A/B).
