"""
Coverage of the true change-in-energy (IEA dAEP, Eq. 12) vs campaign length, at
different LEVELS of Type B. Each campaign draws its own weather (2-day block
resample); the systematic bias shifts the estimate by dDdb*b (linear), so the
expensive bootstrap is computed once per campaign and sigma_B is swept on top.

sigma_B = 0 is the control: with no Type B the bootstrap is correctly calibrated.
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
K_FIX, SIG_A, Z = 0.034, 0.02, 1.959964


def fp(w, s, c):
    w = np.atleast_1d(w); s = np.clip(np.atleast_1d(s), 3, 25)
    return (F_ON if c else F_OFF)(np.column_stack([w, s, np.full(np.shape(w), K_FIX)]))


PON, POFF = fp(wd_s, ws_s, 1), fp(wd_s, ws_s, 0)
on_full = ((np.arange(len(wd_s)) // 7) % 2 == 1)
TRUE = M.delta_aep(np.where(on_full, PON, POFF), on_full, wd_s, ws_s)
eps = 1e-3
dDdb = (M.delta_aep(np.where(on_full, PON * (1 + eps), POFF), on_full, wd_s, ws_s) - TRUE) / eps
print(f"true dAEP at k={K_FIX} = {TRUE:+.2f}%   dDdb={dDdb:.0f}")

blk_id = (t_s / 2).astype(int); blocks = [np.where(blk_id == b)[0] for b in np.unique(blk_id)]
NBLK = len(blocks)
Ls = np.array([0.5, 1, 2, 4, 8]); R = 400
dsamp = {}; hwb = {}
for L in Ls:
    nblk = max(2, round(L * NBLK)); ds = np.empty(R); hh = np.empty(R)
    for r in range(R):
        rng = np.random.default_rng(30000 + r + int(L * 11))
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

levels = [0.0, 0.0025, 0.005, 0.01, 0.02]
colors = ["#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#d62728"]
zr = np.random.default_rng(123).standard_normal((len(Ls), R))
cov_boot = {s: [] for s in levels}; cov_hon = {s: [] for s in levels}
for s in levels:
    tb = Z * abs(dDdb) * s
    for li, L in enumerate(Ls):
        err = np.abs(dsamp[L] + dDdb * s * zr[li])
        cov_boot[s].append(100 * np.mean(err <= hwb[L]))
        cov_hon[s].append(100 * np.mean(err <= np.hypot(hwb[L], tb)))

print(f"\n{'L':>5} " + " ".join(f"sB={s*100:.2g}%".rjust(8) for s in levels) + "   (bootstrap coverage)")
for li, L in enumerate(Ls):
    print(f"{L:>4}y " + " ".join(f"{cov_boot[s][li]:7.0f}%" for s in levels))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for s, c in zip(levels, colors):
    tb = Z * abs(dDdb) * s
    lab = f"$\\sigma_B$={s*100:.2g}%  (floor {tb:.2f}pp)" + ("  [control]" if s == 0 else "")
    ax[0].plot(Ls, cov_boot[s], "o-", color=c, label=lab)
    ax[1].plot(Ls, cov_hon[s], "o-", color=c, label=lab)
for a, ttl in zip(ax, ["Bootstrap only (Type A) — colleagues' report",
                       "Bootstrap ⊕ propagated Type B — honest report"]):
    a.axhline(95, ls="--", color=".4"); a.set_xscale("log"); a.set_xticks(Ls)
    a.set_xticklabels([f"{l:g}y" for l in Ls]); a.set_ylim(0, 103)
    a.set_xlabel("campaign length"); a.set_ylabel("coverage of true ΔAEP [%]")
    a.set_title(ttl, fontsize=10); a.grid(alpha=0.3); a.legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig("fig_typeB_levels.png", dpi=130); plt.close()
print("\nWrote fig_typeB_levels.png")
