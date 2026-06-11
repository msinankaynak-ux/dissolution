"""Permeation / IVPT metrics — steady-state flux (Jss), lag-time and permeability
coefficient (Kp) from a cumulative amount-per-area vs time profile (Franz cell /
in vitro permeation). Standard Fick's-law analysis (textbook; not proprietary).
"""
import numpy as np


def permeation_metrics(time, cum, donor_conc=None, lag_frac=0.4):
    """Jss = slope of the terminal linear region; lag = x-intercept; Kp = Jss/Cd.
    Units follow the input (e.g. cum µg/cm², time h → Jss µg/cm²/h)."""
    t = np.asarray(time, float)
    y = np.asarray(cum, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size < 3:
        return {"jss": None, "lag": None, "kp": None, "r2": None, "n_points": int(t.size)}
    thr = t.min() + lag_frac * (t.max() - t.min())
    sel = t >= thr
    if sel.sum() < 2:
        sel = np.ones_like(t, bool)
    ts, ys = t[sel], y[sel]
    A = np.vstack([ts, np.ones_like(ts)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    yhat = slope * ts + intercept
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else None
    jss = float(slope)
    lag = float(-intercept / slope) if abs(slope) > 1e-12 else None
    kp = float(jss / donor_conc) if (donor_conc and donor_conc > 0) else None
    return {"jss": jss, "lag": lag, "kp": kp, "r2": r2, "n_points": int(sel.sum())}
