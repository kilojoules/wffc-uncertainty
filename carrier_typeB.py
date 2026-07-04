"""
WHERE the systematic enters matters (review M2).

Three carriers for the same nominal 0.5% systematic, all end-to-end:

  1. ON-only power gain      — differential; the metric cannot cancel it.
  2. common-mode power gain  — same sensor error on both states; the metric is
                               exactly invariant (the toggle design's strength).
  3. ON-period wind-speed bias (the *physical* carrier of a yaw-dependent
     transfer-function error): the measured ws used for BINNING is biased
     during ON periods, migrating samples across bins. Its dAEP bias scales
     with the power-curve slope — large in Region II, ~zero at rated — the
     OPPOSITE wind-speed structure of the power-gain carrier.

Outputs the net dAEP bias per carrier and the per-ws-bin structure figure.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import campaigns as C
import metrics as M

EXP_ID = 3
SIG = 0.005                     # 0.5% systematic, all carriers
R = 400
L = 4                           # 4-year campaigns

modes = [("on_only", "ON-only power gain", "#d62728"),
         ("common", "common-mode power gain", "#2ca02c"),
         ("ws_carrier", "ON-period wind-speed bias", "#1f77b4")]

# Common random numbers across carriers (review nit [5]): the SAME seed (hence the
# same weather block-resample, the same per-campaign b, and the same Type-A noise
# draw) is used for all three modes at each replicate r, so the Type-A component is
# shared and largely cancels in the mode-to-mode difference. This sharpens the
# systematic-contribution estimate sqrt(std_mode^2 - std_common^2) below, which
# would otherwise carry ~3.5% MC noise from independent Type-A draws per mode.
print(f"Net dAEP bias, {L}-yr campaigns, sigma={SIG*100:.1f}% on each carrier (common random numbers):")
net = {}
for ai, (mode, lab, _) in enumerate(modes):
    errs = np.empty(R)
    for r in range(R):
        rng = C.rng_for(EXP_ID, 0, r)                 # same r-seed for every mode -> CRN
        idx, blocks, on = C.make_campaign(L, rng)
        pc, po, bid, b = C.observe(idx, on, rng, sigma_b=SIG, mode=mode)
        est_clean, est_obs, _ = M.campaign_estimates(pc, po, on, bid, fixed=C.FIXED_MASK)
        errs[r] = est_obs - C.TRUE_DAEP
    net[mode] = errs
    print(f"  {lab:28s}: bias {errs.mean():+6.3f} pp   spread (std) {errs.std():5.3f} pp")

# ---- per-ws-bin structure of a FIXED +1-sigma systematic, per carrier -------
# deterministic: corrupt the full base-year record with b = +SIG, no noise
on0 = ((np.arange(C.N_S) // C.TOGGLE_LEN) % 2 == 1)
p_clean = np.where(on0, C.PON, C.POFF)


def perbin_delta(mode):
    """change in each ws-bin's (P_on_bin - P_off_bin) caused by b=+SIG [kW]."""
    if mode == "on_only":
        po = np.where(on0, C.PON * (1 + SIG), C.POFF); ws = C.WS_S
    elif mode == "common":
        po = p_clean * (1 + SIG); ws = C.WS_S
    else:
        po = p_clean; ws = np.where(on0, C.WS_S * (1 + SIG), C.WS_S)
    dsc = np.digitize(C.WS_S, M.WS_EDGES)      # clean bins for reference gain
    dsm = np.digitize(ws, M.WS_EDGES)          # measured bins
    centers, d_gain = [], []
    for bb in np.unique(dsc):
        mc = dsc == bb
        mo_c = mc & on0; mf_c = mc & ~on0
        mo_m = (dsm == bb) & on0; mf_m = (dsm == bb) & ~on0
        if min(mo_c.sum(), mf_c.sum(), mo_m.sum(), mf_m.sum()) < 3:
            continue
        gain_clean = p_clean[mo_c].mean() - p_clean[mf_c].mean()
        gain_obs = po[mo_m].mean() - po[mf_m].mean()
        centers.append(C.WS_S[mc].mean()); d_gain.append(gain_obs - gain_clean)
    return np.array(centers), np.array(d_gain)


fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for mode, lab, col in modes:
    cx, dg = perbin_delta(mode)
    ax[0].plot(cx, dg, "o-", color=col, ms=4, label=lab)
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_xlabel("wind speed [m/s]"); ax[0].set_ylabel("bias in per-bin power gain [kW]\n(fixed b = +0.5%)")
ax[0].set_title("The SAME 0.5% systematic has opposite wind-speed structure\n"
                "depending on its physical carrier — and cancels if common-mode", fontsize=10)
ax[0].legend(fontsize=8.5); ax[0].grid(alpha=0.3)

pos = np.arange(len(modes))
stds = np.array([net[m].std() for m, _, _ in modes])
std_A = net["common"].std()                      # pure Type-A spread (weather+noise)
sysc = np.sqrt(np.maximum(stds ** 2 - std_A ** 2, 0))   # systematic contribution
ax[1].bar(pos, stds, color=[c for _, _, c in modes], alpha=0.85)
ax[1].axhline(std_A, ls="--", color=".3", lw=1.5,
              label=f"Type-A spread (weather+noise) = {std_A:.2f} pp")
for p, s_ in zip(pos, sysc):
    if s_ > 0.05:
        ax[1].text(p, stds[p] + 0.03, f"systematic:\n+{s_:.2f} pp", ha="center", fontsize=8)
ax[1].set_xticks(pos); ax[1].set_xticklabels([lab.replace(" ", "\n", 1) for _, lab, _ in modes], fontsize=8.5)
ax[1].set_ylabel("campaign-to-campaign std of ΔAEP estimate [pp]")
ax[1].set_title(f"Spread of the estimate, {L}-yr campaigns, {SIG*100:.1f}% systematic\n"
                "common-mode cancels; the physical (ws) carrier is the most damaging",
                fontsize=10)
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig("fig_carrier_typeB.png", dpi=130); plt.close()
print("\nWrote fig_carrier_typeB.png")
