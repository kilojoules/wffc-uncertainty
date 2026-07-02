"""
WFFC change-in-energy metric, exactly as defined in IEA Wind Task 44 WP2, Sec 4.2.1:

    dAEP = Σ_ij w_ij ( Pbar_WFFC_ij − Pbar_Base_ij ) / Σ_ij w_ij Pbar_Base_ij   (Eq. 12)
    w_ij = (N_Base_ij + N_WFFC_ij) / N                                          (Eq. 13)

2-D binning by wind direction (i) and wind speed (j). Reported in percent.

Validation-design fixes (adversarial review C1 + M3):
  * `delta_aep` accepts an explicit valid-bin mask so a campaign's corrupted
    estimate and its own clean truth are always evaluated on IDENTICAL bins,
    samples and weights — eliminating the estimand drift caused by the
    min-count bin mask growing with campaign length.
  * one shared, vectorised block bootstrap with n_boot=1000 (percentile CI),
    implemented via multinomial block counts + matmul so B=1000 is cheap.
"""
import numpy as np

WD_EDGES = np.array([268.0, 270.0, 272.0])      # 2-deg wd bins within the 266-274 sector
WS_EDGES = np.arange(4.0, 25.0, 1.0)            # 1-m/s ws bins
N_WS = len(WS_EDGES) + 1
NB = (len(WD_EDGES) + 1) * N_WS
MIN_PER_BIN = 3


def bin_index(wd, ws):
    return np.digitize(wd, WD_EDGES) * N_WS + np.digitize(ws, WS_EDGES)


def bin_sums(p, on, bid):
    """Per-bin sums/counts for each toggle state -> (sW, nW, sB, nB)."""
    off = ~on
    sW = np.bincount(bid[on], p[on], NB); nW = np.bincount(bid[on], None, NB)
    sB = np.bincount(bid[off], p[off], NB); nB = np.bincount(bid[off], None, NB)
    return sW, nW, sB, nB


def valid_mask(nW, nB):
    return (nW >= MIN_PER_BIN) & (nB >= MIN_PER_BIN)


def daep_from_sums(sW, nW, sB, nB, valid):
    PW = sW / np.where(nW > 0, nW, 1)
    PB = sB / np.where(nB > 0, nB, 1)
    w = (nW + nB) * valid
    num = (w * (PW - PB)).sum(); den = (w * PB).sum()
    return 100.0 * num / den if den > 0 else np.nan


def delta_aep(p, on, wd=None, ws=None, bid=None, valid=None):
    """IEA Eq. 12 change in energy [%]. Pass `valid` to pin the bin mask."""
    if bid is None:
        bid = bin_index(wd, ws)
    sW, nW, sB, nB = bin_sums(p, on, bid)
    if valid is None:
        valid = valid_mask(nW, nB)
    return daep_from_sums(sW, nW, sB, nB, valid)


def campaign_estimates(p_clean, p_obs, on, bid, fixed=None):
    """(clean dAEP, observed dAEP, valid mask) on IDENTICAL bins/samples/weights.

    `fixed` (optional) caps the valid-bin set at a length-independent mask so
    the estimand cannot drift as campaigns grow (review fix C1). Scoring
    est_obs − est_clean isolates the measurement-error contribution.
    """
    sW, nW, sB, nB = bin_sums(p_obs, on, bid)
    valid = valid_mask(nW, nB)
    if fixed is not None:
        valid &= fixed
    est_obs = daep_from_sums(sW, nW, sB, nB, valid)
    cW, _, cB, _ = bin_sums(p_clean, on, bid)
    est_clean = daep_from_sums(cW, nW, cB, nB, valid)
    return est_clean, est_obs, valid


def deterministic_truth(pon, poff, bid, fixed):
    """The estimand: clean dAEP using ALL samples in both states (no toggle
    noise) on the fixed bin mask. This is the length-independent true value
    the campaign estimates and the coverage target (review fix C1)."""
    n = np.bincount(bid, None, NB)
    sW = np.bincount(bid, pon, NB)
    sB = np.bincount(bid, poff, NB)
    return daep_from_sums(sW, n, sB, n, fixed & (n > 0))


def block_boot_halfwidth(p, on, bid, block_ids, n_boot=1000, rng=None, fixed=None):
    """Vectorised block bootstrap 95% half-width of dAEP.

    Precomputes per-block per-bin sums, then draws multinomial block counts
    (equivalent to resampling nbk blocks with replacement) and aggregates via
    matmul — so n_boot=1000 costs a few matrix products.
    """
    ub, inv = np.unique(block_ids, return_inverse=True)
    nbk = len(ub)
    flat = inv * NB + bid
    NF = nbk * NB
    off = ~on
    bsW = np.bincount(flat[on], p[on], NF).reshape(nbk, NB)
    bnW = np.bincount(flat[on], None, NF).reshape(nbk, NB)
    bsB = np.bincount(flat[off], p[off], NF).reshape(nbk, NB)
    bnB = np.bincount(flat[off], None, NF).reshape(nbk, NB)

    C = rng.multinomial(nbk, np.full(nbk, 1.0 / nbk), size=n_boot).astype(float)
    sW = C @ bsW; nW = C @ bnW; sB = C @ bsB; nB = C @ bnB
    valid = (nW >= MIN_PER_BIN) & (nB >= MIN_PER_BIN)
    if fixed is not None:
        valid &= fixed[None, :]
    PW = sW / np.where(nW > 0, nW, 1)
    PB = sB / np.where(nB > 0, nB, 1)
    w = (nW + nB) * valid
    num = (w * (PW - PB)).sum(axis=1); den = (w * PB).sum(axis=1)
    est = np.where(den > 0, 100.0 * num / den, np.nan)
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return (hi - lo) / 2


def wilson(k, n, z=1.959964):
    """Wilson 95% interval for a proportion, in percent."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))
