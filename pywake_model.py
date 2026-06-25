"""
PyWake wake-steering model lookup for the WFFC uncertainty experiment.

Two V80 turbines in a row (5D spacing). The upstream turbine can be yawed
(wake steering). The wake-expansion coefficient `k` of the Bastankhah-Porte-Agel
Gaussian model is treated as THE atmospheric parameter:

  * it drifts slowly over the campaign  (aleatoric non-stationarity = "weather")
  * the analyst does not know it exactly (epistemic / GUM Type B uncertainty)

We pre-compute the farm power on a regular grid (wd, ws, k) for baseline
(yaw=0) and control (yaw=gamma). Everything downstream is fast numpy/interp on
this lookup, so PyWake is only called ~2*len(k_grid) times. RAM stays tiny.
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
GAMMA = 25.0                      # control yaw on the upstream turbine [deg]

# ---- lookup grid ------------------------------------------------------------
WD_GRID = np.arange(250.0, 290.0 + 1e-6, 1.0)   # waked sector around 270 deg
WS_GRID = np.arange(3.0, 25.0 + 1e-6, 1.0)      # operating wind speeds [m/s]
K_GRID  = np.linspace(0.018, 0.052, 18)         # wake-expansion coefficient


def _farm_power_grid(kval, yaw_deg):
    """Farm power [kW] over the (wd, ws) grid for a constant k and upstream yaw."""
    site = UniformSite(p_wd=[1], ti=0.07)        # ti unused by fixed-k BPA wake
    wfm = Bastankhah_PorteAgel_2014(site, WT, k=kval,
                                    deflectionModel=JimenezWakeDeflection())
    nwd, nws = len(WD_GRID), len(WS_GRID)
    yaw = np.zeros((2, nwd, nws))
    yaw[0] = yaw_deg                              # only the upstream turbine yaws
    sr = wfm(X, Y, wd=WD_GRID, ws=WS_GRID, yaw=yaw, tilt=0, n_cpu=1)
    # Power dims: (wt, wd, ws) -> sum over turbines -> (wd, ws), in kW
    return sr.Power.values.sum(axis=0) / 1e3


def build_lookup():
    """Return interpolators P_off(wd,ws,k), P_on(wd,ws,k) for farm power [kW]."""
    nwd, nws, nk = len(WD_GRID), len(WS_GRID), len(K_GRID)
    P_off = np.empty((nwd, nws, nk))
    P_on  = np.empty((nwd, nws, nk))
    for ik, kval in enumerate(K_GRID):
        P_off[:, :, ik] = _farm_power_grid(kval, 0.0)
        P_on[:, :, ik]  = _farm_power_grid(kval, GAMMA)
    pts = (WD_GRID, WS_GRID, K_GRID)
    f_off = RegularGridInterpolator(pts, P_off, bounds_error=False, fill_value=None)
    f_on  = RegularGridInterpolator(pts, P_on,  bounds_error=False, fill_value=None)
    return f_off, f_on, P_off, P_on


if __name__ == "__main__":
    f_off, f_on, P_off, P_on = build_lookup()
    print("lookup grids:", P_off.shape, "(wd,ws,k)")
    print(f"RAM of lookup arrays: {(P_off.nbytes + P_on.nbytes)/1e6:.3f} MB")

    # Sanity: campaign-average true uplift vs atmospheric k, at a typical
    # sector wind-speed distribution (weight ws by a Rayleigh-ish profile).
    wd0 = 270.0
    ws = WS_GRID
    w_ws = np.exp(-((ws - 9.0) ** 2) / (2 * 4.0 ** 2)); w_ws /= w_ws.sum()
    print("\n  k       P_off[kW]  P_on[kW]   uplift[%]")
    for kval in [0.020, 0.030, 0.040, 0.050]:
        poff = np.array([f_off([wd0, w, kval])[0] for w in ws])
        pon  = np.array([f_on([wd0, w, kval])[0] for w in ws])
        Eoff = (w_ws * poff).sum(); Eon = (w_ws * pon).sum()
        print(f"  {kval:.3f}   {Eoff:8.1f}  {Eon:8.1f}   {100*(Eon-Eoff)/Eoff:+6.2f}")
