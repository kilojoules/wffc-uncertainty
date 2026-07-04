"""
Go/no-go decisions under mild Type B — rebuilt per the adversarial review.

Null model: a PLACEBO toggle (controller ON does nothing physically), so the
true dAEP is EXACTLY zero by construction — no wake-coefficient tuning (M4).
The systematic measurement error exists regardless of what the controller does.

Decision rules (M5 — the one-sided rate is the honest headline):
  * "benefit declared":  CI lower bound > 0        (one-sided, what 'deploy' means)
  * "change declared" :  |estimate| > half-width   (two-sided significance)

End-to-end injection, fixed estimand mask, B=1000 shared bootstrap,
collision-free seeds, Wilson error bars, rates reported alongside the
sigma_B=0 control (C1/M3).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import campaigns as C
import metrics as M

Z = 1.959964
EXP_ID = 2
Ls = [0.5, 1, 2, 4, 8]
R = 600
LEVELS = [0.0, 0.001, 0.0025, 0.005]         # sigma_B: 0 (control), 0.1%, 0.25%, 0.5%
COLORS = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728"]
dDdb = 100.0                                  # placebo: dAEP≈0 -> sensitivity ≈ 100 %/unit-b

print("Null: placebo toggle -> true dAEP = 0 exactly (no parameter tuning).")

ests = {}; hws = {}
for ai, s in enumerate(LEVELS):
    e = np.empty((len(Ls), R)); h = np.empty((len(Ls), R))
    for Li, L in enumerate(Ls):
        for r in range(R):
            rng = C.rng_for(EXP_ID, ai * 1000 + Li, r)
            idx, blocks, on = C.make_campaign(L, rng)
            pc, po, bid, b = C.observe(idx, on, rng, sigma_b=s, mode="on_only",
                                       placebo=True)
            _, est, _ = M.campaign_estimates(pc, po, on, bid, fixed=C.FIXED_MASK)
            h[Li, r] = M.block_boot_halfwidth(po, on, bid, blocks, n_boot=1000,
                                              rng=rng, fixed=C.FIXED_MASK)
            e[Li, r] = est
    ests[s] = e; hws[s] = h
    print(f"  sigma_B={s*100:.2g}% simulated.")


def rates(s, honest=False):
    e, h = ests[s], hws[s]
    hw = np.hypot(h, Z * dDdb * s) if honest else h
    one = 100 * (e - hw > 0).mean(axis=1)                 # benefit declared
    two = 100 * (np.abs(e) > hw).mean(axis=1)             # change declared
    return one, two


print(f"\n{'':>6}" + "".join(f"  sB={s*100:.2g}%".rjust(9) for s in LEVELS))
print("one-sided 'benefit declared' rate [%], bootstrap-only:")
for Li, L in enumerate(Ls):
    print(f"{L:>5}y" + "".join(f"{rates(s)[0][Li]:9.1f}" for s in LEVELS))
print("two-sided 'change declared' rate [%], bootstrap-only:")
for Li, L in enumerate(Ls):
    print(f"{L:>5}y" + "".join(f"{rates(s)[1][Li]:9.1f}" for s in LEVELS))
print("one-sided, honest interval (bootstrap ⊕ propagated Type B):")
for Li, L in enumerate(Ls):
    print(f"{L:>5}y" + "".join(f"{rates(s, honest=True)[0][Li]:9.1f}" for s in LEVELS))

# positive exemplar (M5): bootstrap declares a benefit, honest does not
s_ex = 0.0025; Li4 = Ls.index(4); tb = Z * dDdb * s_ex
for r in range(R):
    est = ests[s_ex][Li4, r]; hb = hws[s_ex][Li4, r]; hh = np.hypot(hb, tb)
    if est - hb > 0 and est - hh <= 0:
        print(f"\nExemplar (4 yr, sigma_B=0.25%, true benefit = 0):")
        print(f"  measured ΔAEP = {est:+.2f}%")
        print(f"  bootstrap 95% CI (est+-hw) [{est-hb:+.2f}, {est+hb:+.2f}] % -> benefit declared, deploy")
        print(f"  honest    95% CI (est+-hw) [{est-hh:+.2f}, {est+hh:+.2f}] % -> not significant")
        break

# ---- figure -----------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
x = np.array(Ls)
for s, c in zip(LEVELS, COLORS):
    one_b, _ = rates(s); one_h, _ = rates(s, honest=True)
    ko = np.round(one_b * R / 100).astype(int)
    lo = np.array([M.wilson(k, R)[0] for k in ko]); hi = np.array([M.wilson(k, R)[1] for k in ko])
    lab = f"$\\sigma_B$={s*100:.2g}%" + ("  [control]" if s == 0 else "")
    ax[0].plot(x, one_b, "o-", color=c, label=lab)
    ax[0].fill_between(x, lo, hi, color=c, alpha=0.12)
    if s > 0:
        ax[1].plot(x, one_h, "s--", color=c, label=lab)
ax[1].plot(x, rates(0.0)[0], "o-", color=COLORS[0], label="$\\sigma_B$=0  [control]")
for a, ttl in zip(ax, ["Bootstrap only — false 'benefit declared' rate",
                       "Bootstrap ⊕ propagated Type B (honest)"]):
    a.axhline(2.5, ls=":", color="k", lw=1.2)
    a.text(x[0], 2.8, "nominal 2.5% (one-sided)", fontsize=7.5, color=".3")
    a.set_xscale("log"); a.set_xticks(x); a.set_xticklabels([f"{l:g}y" for l in Ls])
    a.set_xlabel("campaign length"); a.set_title(ttl, fontsize=10.5)
    a.grid(alpha=0.3); a.legend(fontsize=8.5)
ax[0].set_ylabel("false 'benefit declared' rate [%]  (true ΔAEP = 0)")
plt.suptitle("Placebo controller: how often is a nonexistent benefit declared significant?",
             fontsize=11, y=1.0)
plt.tight_layout(); plt.savefig("fig_mild_typeB_decision.png", dpi=130); plt.close()
print("\nWrote fig_mild_typeB_decision.png")
