"""
PyWake wake-steering lookup for the WFFC uncertainty experiments.

Two V80 turbines in a row (5D spacing). Control = wake steering by the upstream
turbine with a REALISTIC schedule (fix for review finding M4):

    gamma(wd, ws) = GAMMA0 * sign(wd - 270) * taper(ws)

  * direction-signed: deflect the wake AWAY from the downstream rotor on each
    side of the aligned direction (sign verified against PyWake: +25 deg helps
    for wd > 270, -25 deg for wd < 270);
  * tapered: full yaw below WS_TAPER_LO, linear to zero at WS_TAPER_HI, zero
    above (real controllers stop steering as the downstream turbine approaches
    rated power).

The wake-expansion coefficient is FIXED at the PyWake default (K_FIX) for every
experiment — no per-experiment tuning (review M4). The k dimension is kept in
the lookup only for sensitivity checks.

We pre-compute farm power on a (wd, ws, k) grid for baseline (yaw=0) and
control (gamma schedule). Everything downstream is fast numpy interp.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.examples.data.hornsrev1 import V80
from py_wake.site import UniformSite
from py_wake.deflection_models.jimenez import JimenezWakeDeflection

# ---- geometry / control -----------------------------------------------------
WT = V80()
D = WT.diameter()                 # 80 m
X = np.array([0.0, 5 * D])        # WT0 upstream, WT1 downstream (5D east)
Y = np.array([0.0, 0.0])
GAMMA0 = 25.0                     # max control yaw [deg]
WS_TAPER_LO, WS_TAPER_HI = 11.0, 13.0   # taper yaw to 0 approaching rated
K_FIX = 0.0324555                 # PyWake default Bastankhah k — fixed everywhere

# ---- lookup grid ------------------------------------------------------------
WD_GRID = np.arange(250.0, 290.0 + 1e-6, 1.0)
WS_GRID = np.arange(3.0, 25.0 + 1e-6, 1.0)
K_GRID = np.linspace(0.018, 0.052, 18)


def yaw_schedule(wd, ws):
    """Controller yaw [deg] as a function of wind direction and speed.

    Note: np.sign(0) = 0, so exactly at wd = 270.0 (the aligned direction, which
    is also a bin edge) the schedule commands 0 deg and is discontinuous there.
    This is a measure-zero event for the continuous Riso inflow and, where it
    does land, DEFLATES the benefit (no steering) — it never inflates ΔAEP.
    """
    taper = np.clip((WS_TAPER_HI - np.asarray(ws, float)) / (WS_TAPER_HI - WS_TAPER_LO), 0.0, 1.0)
    return GAMMA0 * np.sign(np.asarray(wd, float) - 270.0) * taper


def _farm_power_grid(kval, control):
    """Farm power [kW] over the (wd, ws) grid for constant k; control toggles the schedule."""
    site = UniformSite(p_wd=[1], ti=0.07)
    wfm = Bastankhah_PorteAgel_2014(site, WT, k=kval,
                                    deflectionModel=JimenezWakeDeflection())
    nwd, nws = len(WD_GRID), len(WS_GRID)
    yaw = np.zeros((2, nwd, nws))
    if control:
        wdm, wsm = np.meshgrid(WD_GRID, WS_GRID, indexing="ij")
        yaw[0] = yaw_schedule(wdm, wsm)
    sr = wfm(X, Y, wd=WD_GRID, ws=WS_GRID, yaw=yaw, tilt=0, n_cpu=1)
    return sr.Power.values.sum(axis=0) / 1e3


def build_lookup():
    """Return interpolators P_off(wd,ws,k), P_on(wd,ws,k) for farm power [kW]."""
    nwd, nws, nk = len(WD_GRID), len(WS_GRID), len(K_GRID)
    P_off = np.empty((nwd, nws, nk))
    P_on = np.empty((nwd, nws, nk))
    for ik, kval in enumerate(K_GRID):
        P_off[:, :, ik] = _farm_power_grid(kval, control=False)
        P_on[:, :, ik] = _farm_power_grid(kval, control=True)
    pts = (WD_GRID, WS_GRID, K_GRID)
    f_off = RegularGridInterpolator(pts, P_off, bounds_error=False, fill_value=None)
    f_on = RegularGridInterpolator(pts, P_on, bounds_error=False, fill_value=None)
    return f_off, f_on, P_off, P_on


if __name__ == "__main__":
    f_off, f_on, P_off, P_on = build_lookup()
    ik = int(np.argmin(np.abs(K_GRID - K_FIX)))
    print("lookup:", P_off.shape, f"k_fix={K_FIX}")
    print(f"{'wd':>5} {'gain@9m/s [%]':>14} {'gain@12m/s [%]':>15}")
    for wd in [264, 267, 270, 273, 276]:
        iw = int(np.where(WD_GRID == wd)[0][0])
        g9 = 100 * (P_on[iw, 6, ik] - P_off[iw, 6, ik]) / P_off[iw, 6, ik]
        g12 = 100 * (P_on[iw, 9, ik] - P_off[iw, 9, ik]) / P_off[iw, 9, ik]
        print(f"{wd:>5} {g9:14.2f} {g12:15.2f}")
