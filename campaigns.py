"""
Shared campaign simulator for all experiments (review fixes M3 + M1 + M2).

  * One year of real 10-min Risø inflow, aligned waked sector 266-274 deg.
  * Campaigns of length L years are built by resampling 2-day blocks with
    replacement (disclosed idealisation: no interannual variability).
  * Collision-free seeds: every (experiment, campaign length, replicate) gets
    its own SeedSequence spawn — no shared weather draws across lengths.
  * Errors are injected END-TO-END into the observed power (no analytic
    decoupling): Type A = i.i.d. per-sample noise; Type B = a systematic gain
    drawn once per campaign, applied to ON only, to BOTH states (common-mode
    control), or to the ON-period *measured wind speed* (bin-migration carrier).
  * Truth handling: each campaign is scored against its own clean estimate on
    identical bins/samples/weights (metrics.campaign_estimates).
"""
import numpy as np
from pywake_model import build_lookup, K_FIX
import metrics as M

_F_OFF, _F_ON, _, _ = build_lookup()

from py_wake.examples.data import example_data_path
_d = np.load(example_data_path + "/time_series.npz")
WD_ALL, WS_ALL = _d["wd"], _d["ws"]
DT_DAYS = 1.0 / 6 / 24
T_YEAR = len(WD_ALL) * DT_DAYS

_sec = (WD_ALL >= 266) & (WD_ALL <= 274) & (WS_ALL >= 3) & (WS_ALL <= 25)
IDX = np.where(_sec)[0]
WD_S, WS_S, T_S = WD_ALL[IDX], WS_ALL[IDX], IDX * DT_DAYS
N_S = len(IDX)

_blk = (T_S / 2.0).astype(int)
BLOCKS = [np.where(_blk == b)[0] for b in np.unique(_blk)]
NBLK = len(BLOCKS)

TOGGLE_LEN = 7          # toggle flips every 7 waked-sector samples (~hours-days calendar)
SIG_A = 0.02            # Type A: 2% i.i.d. per-sample power noise


def farm_power(wd, ws, control, k=K_FIX):
    wd = np.atleast_1d(wd); ws = np.clip(np.atleast_1d(ws), 3.0, 25.0)
    pts = np.column_stack([wd, ws, np.full(np.shape(wd), k)])
    return (_F_ON if control else _F_OFF)(pts)


# clean deterministic powers for the base-year sector samples, at the fixed k
PON = farm_power(WD_S, WS_S, True)
POFF = farm_power(WD_S, WS_S, False)

# Length-independent bin mask (the base-year toggle-valid set) and the
# deterministic estimand on it (review fix C1: no mask drift, no toggle noise).
_BID0 = M.bin_index(WD_S, WS_S)
_ON0 = ((np.arange(N_S) // TOGGLE_LEN) % 2 == 1)
_sW0, _nW0, _sB0, _nB0 = M.bin_sums(np.where(_ON0, PON, POFF), _ON0, _BID0)
FIXED_MASK = M.valid_mask(_nW0, _nB0)
TRUE_DAEP = M.deterministic_truth(PON, POFF, _BID0, FIXED_MASK)      # real controller
TRUE_DAEP_PLACEBO = 0.0                                              # exact by construction


def rng_for(experiment_id, L_index, rep):
    """Collision-free RNG per (experiment, campaign length, replicate)."""
    return np.random.default_rng(np.random.SeedSequence([experiment_id, L_index, rep]))


def make_campaign(L_years, rng):
    """Block-resampled campaign -> (sample idx, per-sample block id, toggle mask)."""
    nblk = max(2, round(L_years * NBLK))
    pick = rng.integers(0, NBLK, nblk)
    members = [BLOCKS[p] for p in pick]
    idx = np.concatenate(members)
    block_ids = np.repeat(np.arange(nblk), [len(m) for m in members])
    on = ((np.arange(len(idx)) // TOGGLE_LEN) % 2 == 1)
    return idx, block_ids, on


def observe(idx, on, rng, sigma_b=0.0, mode="on_only", placebo=False,
            ws_bias_frac=0.0, sig_a=SIG_A):
    """Simulate what the analyst records. END-TO-END error injection.

    Returns (p_clean, p_obs, bid, b) where bid is the (wd,ws) bin index built
    from the MEASURED wind speed (so a ws-carrier systematic migrates bins).

      mode='on_only'     : ON power x (1+b)              (differential systematic)
      mode='common'      : ON and OFF power x (1+b)      (common-mode control)
      mode='ws_carrier'  : ON-period measured ws x (1+b) (bin-migration carrier)
      placebo=True       : controller does nothing (ON physics = OFF physics),
                           so the true dAEP is exactly 0 by construction.
    """
    pon = (POFF if placebo else PON)[idx]
    poff = POFF[idx]
    p_clean = np.where(on, pon, poff)
    b = rng.normal(0.0, sigma_b) if sigma_b > 0 else 0.0

    gain_on, gain_off, ws_meas = 1.0, 1.0, WS_S[idx]
    if mode == "on_only":
        gain_on = 1.0 + b
    elif mode == "common":
        gain_on = gain_off = 1.0 + b
    elif mode == "ws_carrier":
        ws_meas = np.where(on, WS_S[idx] * (1.0 + b + ws_bias_frac), WS_S[idx])
    else:
        raise ValueError(mode)

    noise = 1.0 + sig_a * rng.standard_normal(len(idx))
    p_obs = np.where(on, pon * gain_on, poff * gain_off) * noise
    bid = M.bin_index(WD_S[idx], ws_meas)
    return p_clean, p_obs, bid, b


def true_daep(placebo=False):
    """Clean full-record dAEP at the fixed k (for reporting, not scoring)."""
    on = ((np.arange(N_S) // TOGGLE_LEN) % 2 == 1)
    pon = POFF if placebo else PON
    p = np.where(on, pon, POFF)
    return M.delta_aep(p, on, WD_S, WS_S)
