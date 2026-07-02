"""
Un-normalized (absolute kW) view of the same experiment, on the shared
infrastructure (fixed k, tapered controller, end-to-end errors).

  * per-wind-speed decomposition of the clean power gain [kW], with the
    ON-only power-gain carrier's Type-B contribution overlaid (see
    carrier_typeB.py for why the carrier choice sets this structure);
  * the raw campaign-to-campaign spread of the ΔAEP estimate vs a single
    campaign's bootstrap CI — the overconfidence with no normalization.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import campaigns as C
import metrics as M

EXP_ID = 5
SIG_B = 0.005
L = 4; R = 400

on0 = ((np.arange(C.N_S) // C.TOGGLE_LEN) % 2 == 1)
print(f"true dAEP = {C.TRUE_DAEP:+.2f}%   baseline mean power = {C.POFF.mean():.0f} kW")

# per-ws-bin clean gain + ON-only Type-B 1-sigma contribution [kW]
ds = np.digitize(C.WS_S, M.WS_EDGES)
centers, gain, tb = [], [], []
for bb in np.unique(ds):
    m = ds == bb
    if m.sum() < 10:
        continue
    centers.append(C.WS_S[m].mean())
    gain.append(C.PON[m].mean() - C.POFF[m].mean())
    tb.append(SIG_B * C.PON[m].mean())
centers = np.array(centers); gain = np.array(gain); tb = np.array(tb)

# campaign-to-campaign spread (end-to-end, sigma_B) vs one bootstrap CI
ests = np.empty(R)
for r in range(R):
    rng = C.rng_for(EXP_ID, 0, r)
    idx, blocks, on = C.make_campaign(L, rng)
    pc, po, bid, b = C.observe(idx, on, rng, sigma_b=SIG_B)
    _, ests[r], _ = M.campaign_estimates(pc, po, on, bid, fixed=C.FIXED_MASK)
rng = C.rng_for(EXP_ID, 1, 0)
idx, blocks, on = C.make_campaign(L, rng)
pc, po, bid, b1 = C.observe(idx, on, rng, sigma_b=SIG_B)
_, est1, _ = M.campaign_estimates(pc, po, on, bid, fixed=C.FIXED_MASK)
hw1 = M.block_boot_halfwidth(po, on, bid, blocks, n_boot=1000, rng=rng, fixed=C.FIXED_MASK)
print(f"spread of estimates (std) = {ests.std():.2f} pp   one bootstrap hw = {hw1:.2f} pp")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].axhline(0, color="k", lw=0.6)
ax[0].bar(centers, gain, width=0.8, color="#2ca02c", alpha=0.8, label="true gain (on−off)")
ax[0].errorbar(centers, gain, yerr=1.959964 * tb, fmt="none", ecolor="#d62728",
               elinewidth=1.4, capsize=2, label="Type-B 95% if carried by ON power ($b\\,P_{on}$)")
ax[0].set_xlabel("wind speed [m/s]"); ax[0].set_ylabel("power change [kW]")
ax[0].set_title("Per-wind-speed (absolute): steering gain peaks in Region II and tapers to\n"
                "zero near rated; an ON-power systematic instead GROWS with power", fontsize=9.5)
ax[0].legend(fontsize=8.5)

ax[1].hist(ests, bins=40, density=True, color="#bbbbbb", alpha=0.85,
           label=f"spread of $\\hat{{\\Delta}}$AEP over campaigns\n({L} yr, $\\sigma_B$={SIG_B*100:.1f}%, end-to-end)")
ax[1].axvline(C.TRUE_DAEP, color="k", lw=2, label=f"true ΔAEP = {C.TRUE_DAEP:+.2f}%")
ax[1].axvspan(est1 - hw1, est1 + hw1, color="#1f77b4", alpha=0.3,
              label=f"one campaign's bootstrap 95% CI\n[{est1-hw1:+.2f}, {est1+hw1:+.2f}] %")
ax[1].axvline(est1, color="#1f77b4", lw=1.5)
ax[1].set_xlabel("estimated ΔAEP [%]"); ax[1].set_ylabel("density")
ax[1].set_title("The estimate moves between campaigns more than any single\n"
                "campaign's bootstrap CI admits (the Type-B share is invisible to it)",
                fontsize=9.5)
ax[1].legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig("fig_absolute_view.png", dpi=130); plt.close()
print("Wrote fig_absolute_view.png")
