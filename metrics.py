"""
WFFC change-in-energy metric, exactly as defined in IEA Wind Task 44 WP2
(Review and Best Practices for Wind Farm Flow Control Field Assessment), Sec 4.2.1.

Change in AEP / change in energy (their Eq. 12), with 2-D binning by wind
direction (i) and wind speed (j) and occurrence weights (their Eq. 13):

    dAEP = Σ_ij w_ij ( Pbar_WFFC_ij − Pbar_Base_ij ) / Σ_ij w_ij Pbar_Base_ij
    w_ij = (N_Base_ij + N_WFFC_ij) / N            (the 1/N cancels in the ratio)

Pbar_*_ij is the mean test-turbine (here the turbine-pair) power in bin (i, j)
during baseline (OFF) and flow-control (WFFC, ON) periods. We report dAEP in
percent. Using observed bin weights gives the change in energy for the campaign's
own conditions; long-term wind-rose weights would give the change in annual AEP.
"""
import numpy as np

WD_EDGES = np.array([268.0, 270.0, 272.0])      # 2-deg wind-direction bins in the 266-274 sector
WS_EDGES = np.arange(4.0, 25.0, 1.0)            # 1-m/s wind-speed bins
N_WS = len(WS_EDGES) + 1
NB = (len(WD_EDGES) + 1) * N_WS
MIN_PER_BIN = 3


def _bin(wd, ws):
    return np.digitize(wd, WD_EDGES) * N_WS + np.digitize(ws, WS_EDGES)


def delta_aep(p, on, wd, ws, bid=None):
    """IEA Eq. 12 change in energy [%] from toggle data.

    p   : observed turbine-pair power per sample (ON samples carry control power,
          OFF samples carry baseline power)
    on  : boolean toggle mask (True = flow control, False = baseline)
    """
    if bid is None:
        bid = _bin(wd, ws)
    off = ~on
    sW = np.bincount(bid[on], p[on], NB); nW = np.bincount(bid[on], None, NB)
    sB = np.bincount(bid[off], p[off], NB); nB = np.bincount(bid[off], None, NB)
    v = (nW >= MIN_PER_BIN) & (nB >= MIN_PER_BIN)
    PW = sW / np.where(nW > 0, nW, 1)           # mean WFFC power per bin
    PB = sB / np.where(nB > 0, nB, 1)           # mean baseline power per bin
    w = (nW + nB) * v                            # occurrence weights (Eq. 13)
    num = (w * (PW - PB)).sum(); den = (w * PB).sum()
    return 100.0 * num / den if den > 0 else np.nan


def block_bootstrap(p, on, wd, ws, t, n_boot=200, block_days=2.0, rng=None):
    """Block bootstrap of dAEP over ~`block_days` calendar blocks of `t`."""
    blk = (t / block_days).astype(int); ub = np.unique(blk)
    mem = [np.where(blk == b)[0] for b in ub]; nbk = len(ub)
    bid = _bin(wd, ws); est = np.empty(n_boot)
    for i in range(n_boot):
        sel = np.concatenate([mem[j] for j in rng.integers(0, nbk, nbk)])
        est[i] = delta_aep(p[sel], on[sel], None, None, bid=bid[sel])
    return est


def boot_halfwidth(p, on, wd, ws, t, n_boot=200, block_days=2.0, rng=None):
    e = block_bootstrap(p, on, wd, ws, t, n_boot, block_days, rng)
    lo, hi = np.nanpercentile(e, [2.5, 97.5])
    return (hi - lo) / 2
