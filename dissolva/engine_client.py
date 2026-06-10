"""Engine client — routes heavy analysis to the DissolvA backend API when a
backend URL is configured, with a transparent fallback to the local engine
(``dissolva.models``) so the app keeps working offline / before deploy.

Configure the backend with either:
- ``st.secrets["backend"]["url"]``  (Streamlit Cloud / secrets.toml), or
- env var ``BACKEND_URL``.
Unset → local mode (uses the in-repo engine).

This is the seam for IP protection: once every heavy op (fit / f2 / bootstrap)
goes through the API and the API returns everything the UI needs, the engine
can be removed from the public frontend repo.
"""
import os
import numpy as np
import streamlit as st

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from dissolva import models as _engine


def backend_url():
    """Configured backend base URL, or None for local mode."""
    u = None
    try:
        u = (st.secrets.get("backend") or {}).get("url")
    except Exception:
        u = None
    u = u or os.getenv("BACKEND_URL") or ""
    return u.rstrip("/") or None


def using_backend():
    return backend_url() is not None and requests is not None


def _api_key():
    try:
        return (st.secrets.get("backend") or {}).get("api_key") or os.getenv("BACKEND_API_KEY") or ""
    except Exception:
        return os.getenv("BACKEND_API_KEY") or ""


def _post(path, payload, timeout=120):
    url = backend_url()
    headers = {}
    key = _api_key()
    if key:
        headers["X-API-Key"] = key
    resp = requests.post(url + path, json=payload, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp.json()


def _floats(a):
    return [float(x) for x in a]


# ── f1 / f2 similarity ───────────────────────────────────────────────────────
def similarity(ref_time, ref_release, test_time, test_release):
    """Return (f1, f2, n_points_used). Backend if configured, else local engine."""
    if using_backend():
        try:
            d = _post("/api/f2", {
                "ref_time": _floats(ref_time), "ref_release": _floats(ref_release),
                "test_time": _floats(test_time), "test_release": _floats(test_release),
            })
            return d["f1"], d["f2"], d["n_points_used"]
        except Exception as e:
            st.caption(f"⚠️ Backend unavailable ({type(e).__name__}); computed locally.")

    # local fallback
    t_ref = np.asarray(ref_time, float); r_ref = np.asarray(ref_release, float)
    t_tst = np.asarray(test_time, float); r_tst = np.asarray(test_release, float)
    common = np.intersect1d(t_ref, t_tst)
    rr = np.array([r_ref[np.where(t_ref == ti)[0][0]] for ti in common])
    rt = np.array([r_tst[np.where(t_tst == ti)[0][0]] for ti in common])
    mask = _engine.fda_f2_mask(rr)
    return (float(_engine.f1_score(rr[mask], rt[mask])),
            float(_engine.f2_score(rr[mask], rt[mask])),
            int(mask.sum()))


# ── Kinetic model fitting ────────────────────────────────────────────────────
def fit_models(time, release, names, include_curves=True, curve_n=400,
               weight_scheme="none", sd=None):
    """Fit `names` to the profile. Returns (results_by_name, best_by_aicc).

    Each result is a dict with the same keys the UI expects (name, category,
    success, params, r2, r2adj, rmse, aic, aicc, bic, msc, n_params, equation,
    reference, error) plus optional curve_t/curve_y for plotting (backend mode).

    `weight_scheme` selects weighted least squares (none|1/y|1/y2|1/sd); `sd` is
    the per-point standard deviation list required when weight_scheme='1/sd'.
    """
    if using_backend():
        try:
            payload = {
                "time": _floats(time), "release": _floats(release),
                "models": list(names),
                "include_curves": bool(include_curves), "curve_n": int(curve_n),
                "weight_scheme": weight_scheme,
            }
            if sd is not None:
                payload["sd"] = _floats(sd)
            d = _post("/api/fit", payload)
            return {r["name"]: r for r in d["results"]}, d.get("best_by_aicc")
        except Exception as e:
            st.caption(f"⚠️ Backend unavailable ({type(e).__name__}); computed locally.")

    t = np.asarray(time, float); y = np.asarray(release, float)
    out = {n: _engine.fit_model(t, y, n, weight_scheme=weight_scheme, sd=sd)
           for n in names}
    ok = [(n, r) for n, r in out.items() if r.get("success") and r.get("aicc") is not None]
    best = min(ok, key=lambda kv: kv[1]["aicc"])[0] if ok else None
    return out, best


# ── Vessel-level bootstrap f2 ────────────────────────────────────────────────
def bootstrap(ref_time, ref_raw, test_time, test_raw, method="nonparametric",
              iterations=5000, seed=42, lower_pctile=5.0, progress=None):
    """Return a dict with f2_observed/f2_lower/f2_upper/f2_mean/f2_median/f2_sd/
    n_iter/similar/distribution. Backend if configured, else local engine."""
    if using_backend():
        try:
            return _post("/api/bootstrap-f2", {
                "ref_time": _floats(ref_time), "ref_raw": [_floats(r) for r in ref_raw],
                "test_time": _floats(test_time), "test_raw": [_floats(r) for r in test_raw],
                "method": method, "iterations": int(iterations), "seed": int(seed),
                "lower_pctile": float(lower_pctile), "include_distribution": True,
            }, timeout=300)
        except Exception as e:
            st.caption(f"⚠️ Backend unavailable ({type(e).__name__}); computed locally.")

    # local fallback — mirror the engine pipeline
    t_ref = np.asarray(ref_time, float); raw_ref = np.asarray(ref_raw, float)
    t_tst = np.asarray(test_time, float); raw_tst = np.asarray(test_raw, float)
    common = np.intersect1d(t_ref, t_tst)
    ref_idx = [np.where(t_ref == ti)[0][0] for ti in common]
    tst_idx = [np.where(t_tst == ti)[0][0] for ti in common]
    rr_c = raw_ref[ref_idx, :]; rt_c = raw_tst[tst_idx, :]
    rr_obs = rr_c.mean(axis=1); rt_obs = rt_c.mean(axis=1)
    mask = _engine.fda_f2_mask(rr_obs)
    f2_obs = float(_engine.f2_score(rr_obs[mask], rt_obs[mask]))
    dist = _engine.bootstrap_f2(rr_c, rt_c, mask, iterations=int(iterations),
                                seed=int(seed), method=method, progress=progress)
    dist = dist[np.isfinite(dist)]
    f2_lower = float(np.percentile(dist, lower_pctile))
    return {
        "f2_observed": f2_obs, "f2_lower": f2_lower,
        "f2_upper": float(np.percentile(dist, 100 - lower_pctile)),
        "f2_mean": float(dist.mean()), "f2_median": float(np.median(dist)),
        "f2_sd": float(dist.std(ddof=1)), "n_iter": int(dist.size),
        "similar": bool(f2_lower >= 50), "distribution": [float(x) for x in dist],
    }


# ── Membership & usage analytics (best-effort; never block/raise to the UI) ──
def _admin_key():
    try:
        return (st.secrets.get("backend") or {}).get("admin_key") or os.getenv("BACKEND_ADMIN_KEY") or ""
    except Exception:
        return os.getenv("BACKEND_ADMIN_KEY") or ""


def upsert_member(email, name=""):
    """Register/refresh the signed-in user as a free 'core' member. Silent no-op
    if the backend or email is missing."""
    if not (using_backend() and email):
        return
    try:
        _post("/api/members/upsert", {"email": email, "name": name or ""}, timeout=15)
    except Exception:
        pass


def log_event(feature, email=""):
    """Fire-and-forget usage event (feature name only — no scientific data)."""
    if not using_backend():
        return
    try:
        _post("/api/events", {"feature": feature, "email": email or ""}, timeout=8)
    except Exception:
        pass


def _admin_get(path, timeout=20):
    url = backend_url()
    if not url:
        raise RuntimeError("Backend URL not configured.")
    headers = {}
    if _api_key():
        headers["X-API-Key"] = _api_key()
    if _admin_key():
        headers["X-Admin-Key"] = _admin_key()
    resp = requests.get(url + path, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def admin_members():
    return _admin_get("/api/admin/members")


def admin_stats():
    return _admin_get("/api/admin/stats")
