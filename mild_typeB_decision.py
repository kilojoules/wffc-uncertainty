"""
Different CONCLUSIONS under mild Type B, using the IEA Task 44 change-in-energy
metric (dAEP, Eq. 12; see metrics.py).

Decision: does the 95% interval on dAEP exclude zero -> "WFFC gives a
statistically significant benefit"?  True dAEP set ~0 (marginal controller).
Each campaign draws its own weather (2-day block resample). We count how often
each method falsely declares significance.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pywake_model import build_lookup, K_GRID
import metrics as M

F_OFF, F_ON, _, _ = build_lookup()
from py_wake.examples.data import example_data_path
d = np.load(example_data_path + "/time_series.npz"); WD, WS = d["wd"], d["ws"]
DT = 1 / 6 / 24; TY = len(WD) * DT
S = (WD >= 266) & (WD <= 274) & (WS >= 3) & (WS <= 25); idx0 = np.where(S)[0]
wd_s, ws_s, t_s = WD[idx0], WS[idx0], idx0 * DT
SIG_A, Z = 0.02, 1.959964


def fp(w, s, k, c):
    w = np.atleast_1d(w); s = np.clip(np.atleast_1d(s), 3, 25)
    return (F_ON if c else F_OFF)(np.column_stack([w, s, np.full(np.shape(w), k)]))


on_full = ((np.arange(len(wd_s)) // 7) % 2 == 1)
kc = np.linspace(K_GRID[0], K_GRID[-1], 80)
av = np.array([M.delta_aep(np.where(on_full, fp(wd_s, ws_s, k, 1), fp(wd_s, ws_s, k, 0)),
                           on_full, wd_s, ws_s) for k in kc])
K0 = float(np.interp(0.0, av[::-1], kc[::-1]))
PON, POFF = fp(wd_s, ws_s, K0, 1), fp(wd_s, ws_s, K0, 0)
TRUE = M.delta_aep(np.where(on_full, PON, POFF), on_full, wd_s, ws_s)
eps = 1e-3
dDdb = (M.delta_aep(np.where(on_full, PON * (1 + eps), POFF), on_full, wd_s, ws_s) - TRUE) / eps
print(f"k0={K0:.4f}  true dAEP = {TRUE:+.3f}%  (marginal)  dDdb={dDdb:.0f}")

blk_id = (t_s / 2).astype(int); blocks = [np.where(blk_id == b)[0] for b in np.unique(blk_id)]
NBLK = len(blocks)
Ls = np.array([0.5, 1, 2, 4, 8]); R = 600
dsamp = {}; hwb = {}
for L in Ls:
    nblk = max(2, round(L * NBLK)); ds = np.empty(R); hh = np.empty(R)
    for r in range(R):
        rng = np.random.default_rng(12000 + r + int(L * 7))
        pick = rng.integers(0, NBLK, nblk); members = [blocks[p] for p in pick]
        idx = np.concatenate(members); cb = np.repeat(np.arange(nblk), [len(m) for m in members])
        on = ((np.arange(len(idx)) // 7) % 2 == 1)
        po = PON[idx] * (1 + SIG_A * rng.standard_normal(len(idx)))
        pf = POFF[idx] * (1 + SIG_A * rng.standard_normal(len(idx)))
        p = np.where(on, po, pf); wdc = wd_s[idx]; wsc = ws_s[idx]
        ds[r] = M.delta_aep(p, on, wdc, wsc) - TRUE
        bid = M._bin(wdc, wsc); ub = np.unique(cb); mem = [np.where(cb == b)[0] for b in ub]; nbk = len(ub)
        be = np.empty(150)
        for i in range(150):
            sel = np.concatenate([mem[j] for j in rng.integers(0, nbk, nbk)])
            be[i] = M.delta_aep(p[sel], on[sel], None, None, bid=bid[sel])
        lo, hi = np.nanpercentile(be, [2.5, 97.5]); hh[r] = (hi - lo) / 2
    dsamp[L] = ds; hwb[L] = hh
    print(f"  L={L:>4}y  std(est)={ds.std():.3f}pp  mean hw_boot={hh.mean():.3f}pp")

levels = [0.0, 0.001, 0.0025, 0.005]
zr = np.random.default_rng(321).standard_normal((len(Ls), R))
fpr_b = {s: [] for s in levels}; fpr_h = {s: [] for s in levels}
for s in levels:
    tb = Z * abs(dDdb) * s
    for li, L in enumerate(Ls):
        est = TRUE + dsamp[L] + dDdb * s * zr[li]
        fpr_b[s].append(100 * np.mean(np.abs(est) > hwb[L]))
        fpr_h[s].append(100 * np.mean(np.abs(est) > np.hypot(hwb[L], tb)))

print(f"\nFalse 'significant benefit' rate (true dAEP ~ 0):")
print(f"{'L':>5} " + " ".join(f"B@{s*100:.2g}%".rjust(7) for s in levels)
      + "  | " + " ".join(f"H@{s*100:.2g}%".rjust(7) for s in levels))
for li, L in enumerate(Ls):
    print(f"{L:>4}y " + " ".join(f"{fpr_b[s][li]:6.0f}%" for s in levels)
          + "  | " + " ".join(f"{fpr_h[s][li]:6.0f}%" for s in levels))

li4 = int(np.where(Ls == 4)[0][0]); s_ex = 0.0025; tb_ex = Z * abs(dDdb) * s_ex
for r in range(R):
    est = TRUE + dsamp[4][r] + dDdb * s_ex * zr[li4][r]; hb = hwb[4][r]; hh_ = np.hypot(hb, tb_ex)
    if abs(est) > hb and abs(est) <= hh_:
        print(f"\nExample (4 yr, sigma_B=0.25%):  measured ΔAEP {est:+.2f}%")
        print(f"  bootstrap 95% CI [{est-hb:+.2f}, {est+hb:+.2f}] % -> excludes 0 -> SIGNIFICANT, deploy")
        print(f"  honest   95% CI [{est-hh_:+.2f}, {est+hh_:+.2f}] % -> includes 0 -> not significant")
        break

fig, ax = plt.subplots(figsize=(8.5, 5.5))
cols = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728"]
for s, c in zip(levels, cols):
    lab = "control ($\\sigma_B$=0)" if s == 0 else f"$\\sigma_B$={s*100:.2g}%"
    ax.plot(Ls, fpr_b[s], "o-", color=c, label=f"bootstrap, {lab}")
    if s > 0:
        ax.plot(Ls, fpr_h[s], "s--", color=c, alpha=0.55)
ax.axhline(5, ls=":", color="k", lw=1.4, label="nominal 5%")
ax.set_xscale("log"); ax.set_xticks(Ls); ax.set_xticklabels([f"{l:g}y" for l in Ls])
ax.set_xlabel("campaign length"); ax.set_ylabel("'significant ΔAEP' false-positive rate [%]")
ax.set_title("Different conclusions under MILD Type B (IEA ΔAEP metric)\n"
             "solid = bootstrap (false positives climb); dashed = honest (stays ~5%)", fontsize=11)
ax.legend(fontsize=8, ncol=2, loc="upper left"); ax.grid(alpha=0.3)
ax.text(0.97, 0.45, "WFFC actually does ~nothing here.\nbootstrap increasingly 'finds' a\n"
        "significant benefit; honest does not.", transform=ax.transAxes, ha="right",
        fontsize=8.5, color=".25", bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
plt.tight_layout(); plt.savefig("fig_mild_typeB_decision.png", dpi=130); plt.close()
print("\nWrote fig_mild_typeB_decision.png")
