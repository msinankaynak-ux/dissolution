"""DissolvA page module: Kinetic Model Fitting. Extracted from app.py (Phase 3b modularization)."""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import warnings
try:
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False
from scipy.optimize import curve_fit, root
from scipy.stats import norm as sp_norm
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from dissolva.theme import OXFORD, AMBER, PALETTE, style_ax
from dissolva.models import (MODEL_DEFS, CATEGORIES, fit_model, compute_mdt,
    compute_de, r2s, r2adj, aic_fn, msc_fn, _nz)
from dissolva.state import (current_tier, require_tier, _safe_profile_names,
    _get_index, _rename_profile, _clear_all)
from dissolva.content import show_literature, show_all_references, analyze_profile_shape
from dissolva import engine_client


def render():
    cfg = st.session_state.method_cfg
    time_unit = cfg["time_unit"]
    st.header("Kinetic Model Fitting")

    # ── Active Substance Info Card ───────────────────────────────────────────────
    _as_km = st.session_state.get("active_substance", {})
    if _as_km.get("fetch_done") and _as_km.get("name"):
        _pc_km  = _as_km.get("pubchem") or {}
        _bcs_km = _as_km.get("bcs_class") or {}
        _sel_km = _as_km.get("selected_method")
        _fda_km = _as_km.get("fda_methods", [])
        _sel_fda = _fda_km[_sel_km] if (_sel_km is not None and _sel_km < len(_fda_km)) else None

        st.markdown(
            f'<div style="background:linear-gradient(90deg,#002147,#003a7a);'
            f'border-radius:8px;padding:10px 18px;margin-bottom:14px;'
            f'display:flex;align-items:center;gap:16px;">'
            f'<div style="font-size:28px;">🔬</div>'
            f'<div style="flex:1;">'
            f'<span style="font-size:15px;font-weight:700;color:white;">{_as_km["name"]}</span>'
            f'<span style="font-size:11px;color:rgba(255,255,255,0.6);margin-left:10px;">'
            f'{_pc_km.get("formula","")} | MW: {_pc_km.get("mw","")} g/mol | '
            f'LogP: {_pc_km.get("xlogp","N/A")}</span>'
            + (f'<div style="font-size:10px;color:rgba(255,255,255,0.5);margin-top:2px;">'
               f'FDA Method: {_sel_fda["apparatus"]} | {_sel_fda["speed_rpm"]} rpm | '
               f'{_sel_fda["medium"][:50]}</div>' if _sel_fda else '') +
            f'</div></div>',
            unsafe_allow_html=True
        )
    if not st.session_state.profiles:
        st.warning("Please load at least one dissolution profile in Data Input first.")
        st.stop()

    _km_names = list(st.session_state.profiles.keys())
    pname = st.selectbox("Select Profile", _km_names,
        index=_get_index(_km_names, st.session_state.selected_ref_id, 0))
    d = st.session_state.profiles[pname]
    t_arr = np.array(d["time"], dtype=float)
    r_arr = np.array(d["release"], dtype=float)

    # Smart model recommendation
    shape_info = analyze_profile_shape(t_arr, r_arr)
    top_models_valid = [m for m in shape_info['top_models'] if m in MODEL_DEFS]
    st.markdown(
        f'<div style="background:rgba(255,191,0,0.08);border-left:4px solid #FFBF00;'
        f'border-radius:0 6px 6px 0;padding:14px 18px;margin:12px 0;">'
        f'<div style="font-size:0.85rem;font-weight:700;color:#e6ac00;margin-bottom:6px;">'
        f'{shape_info["icon"]} Smart Model Recommendation — {shape_info["shape"]} Profile</div>'
        f'<div style="font-size:0.82rem;color:#666;">{shape_info["reason"]}<br>'
        f'<strong>Recommended:</strong> {", ".join(top_models_valid)}</div></div>',
        unsafe_allow_html=True
    )
    use_smart = st.checkbox(
        f"Use only recommended {len(top_models_valid)} models",
        value=True, key=f"use_smart_models_{pname}"
    )

    st.markdown("#### Select Models by Category")
    tab_list = st.tabs(CATEGORIES)
    selected_models = []
    for tab, cat in zip(tab_list, CATEGORIES):
        with tab:
            cat_models = [k for k,v in MODEL_DEFS.items() if v[5]==cat]
            if use_smart:
                default_sel = [m for m in top_models_valid if m in cat_models]
            else:
                default_sel = cat_models[:4] if cat=="Basic" else []
            chosen = st.multiselect(f"{cat} models", cat_models,
                                    default=default_sel,
                                    key=f"ms_{cat}_{pname}")
            selected_models.extend(chosen)

    if selected_models:
        with st.expander("Equation Reference for Selected Models"):
            for mn in selected_models:
                _,_,pnames,eq,ref,cat = MODEL_DEFS[mn]
                st.markdown(
                    f"**{mn}** ({ref})<br>"
                    f"<div class='eq-box'>{eq} | params: {', '.join(pnames)}</div>",
                    unsafe_allow_html=True
                )

    # ── Weighting scheme (weighted least squares) ────────────────────────────
    _sd_list = d.get("sd")
    _sd_arr = np.array(_sd_list, dtype=float) if _sd_list is not None else None
    _has_sd = _sd_arr is not None and _sd_arr.size == len(r_arr) and bool(np.any(_sd_arr > 0))
    _w_labels = ["None (ordinary least squares)", "1/y", "1/y²"]
    _w_codes = ["none", "1/y", "1/y2"]
    if _has_sd:
        _w_labels.append("1/SD² (inverse variance)")
        _w_codes.append("1/sd")
    _w_label = st.selectbox(
        "Weighting scheme", _w_labels, index=0, key=f"weight_scheme_{pname}",
        help="Weighted least squares down-weights points with larger variance. "
             "1/y and 1/y² are relative schemes; 1/SD² uses per-point SD from Data Input."
    )
    _weight_scheme = _w_codes[_w_labels.index(_w_label)]
    st.caption(
        "Weighting helps when later / high-release points carry larger variance "
        "than early points, so they don't dominate the unweighted fit."
    )
    if not _has_sd:
        st.caption("SD weighting needs per-point SD from Data Input.")

    if st.button("Run Model Fitting", type="primary") and selected_models:
        _sd_for_fit = _sd_list if _weight_scheme == "1/sd" else None
        with st.spinner(f"Fitting {len(selected_models)} model(s)…"):
            _results, _best = engine_client.fit_models(
                t_arr, r_arr, selected_models,
                weight_scheme=_weight_scheme, sd=_sd_for_fit)
        st.session_state.fit_results = _results
        st.success(f"Fitting complete - {len(selected_models)} models processed.")

    if st.session_state.fit_results:
        res_ok   = {k:v for k,v in st.session_state.fit_results.items() if v["success"]}
        res_fail = {k:v for k,v in st.session_state.fit_results.items() if not v["success"]}

        # Korsmeyer-Peppas validity advisory (Peppas 1985): the release exponent n
        # is only interpretable over the first ~60% of release.
        _kp_like = [m for m in res_ok if ("Korsmeyer" in m or m.startswith("KP"))]
        if _kp_like and float(np.nanmax(r_arr)) > 60.0:
            st.info(
                "ℹ️ **Korsmeyer-Peppas note:** the release exponent *n* is interpretable only "
                "for the **first ~60% of release** (Peppas, 1985). Above 60%, *n* and its "
                "transport-mechanism classification become unreliable — read the "
                f"{', '.join(_kp_like)} fit with caution."
            )

        # Korsmeyer-Peppas transport-mechanism interpretation (Ritger-Peppas 1987,
        # cylindrical geometry). UI-only: reads the already-fitted exponent n.
        for _kpm in _kp_like:
            _kp_params = (res_ok[_kpm].get("params") or {})
            _n = _kp_params.get("n")
            if _n is None:
                continue
            try:
                _nf = float(_n)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(_nf):
                continue
            # Ritger-Peppas (cylinder): explicit half-open bands so the n≈0.89
            # Case-II window is deterministic (not float-edge dependent).
            if _nf <= 0.45:
                _mech = "Fickian diffusion"
            elif _nf < 0.885:
                _mech = "Anomalous (non-Fickian) transport"
            elif _nf <= 0.895:
                _mech = "Case II transport (zero-order)"
            else:
                _mech = "Super Case II transport"
            st.caption(f"{_kpm}: n = {_nf:.2f} → {_mech} (cylindrical geometry).")

        st.subheader("Model Ranking")
        # Sıralama ölçütü — AICc/BIC küçük=iyi, R2adj/MSC büyük=iyi
        _RANK_META = {
            "AICc (recommended)": ("aicc", True),
            "BIC":                ("bic",  True),
            "AIC":                ("aic",  True),
            "RMSE":               ("rmse", True),
            "R2adj":              ("r2adj", False),
            "MSC":                ("msc",  False),
        }
        rank_choice = st.selectbox(
            "Rank models by", list(_RANK_META.keys()), index=0,
            help="AICc = small-sample-corrected AIC; recommended for dissolution with few time points. "
                 "AIC/AICc/BIC/RMSE: lower is better. R2adj/MSC: higher is better."
        )
        _rank_key, _rank_asc = _RANK_META[rank_choice]
        # Safe numeric round — None/NaN/inf metrics (possible for marginal fits via
        # the backend) become np.nan so pandas sort + background_gradient keep working.
        def _rn(x, n):
            try:
                xf = float(x)
                return round(xf, n) if (xf == xf and xf not in (float("inf"), float("-inf"))) else np.nan
            except (TypeError, ValueError):
                return np.nan
        rows=[{"Model":v.get("name",""),"Category":v.get("category",""),
               "R2":_rn(v.get("r2"),4),"R2adj":_rn(v.get("r2adj"),4),
               "RMSE":_rn(v.get("rmse"),3),
               "AIC":_rn(v.get("aic"),2),"AICc":_rn(v.get("aicc"),2),"BIC":_rn(v.get("bic"),2),
               "MSC":_rn(v.get("msc"),3),
               "Params":v.get("n_params",0),"Reference":v.get("reference","")}
              for v in res_ok.values()]
        if rows:
            _sort_col = {"aicc":"AICc","bic":"BIC","aic":"AIC","rmse":"RMSE",
                         "r2adj":"R2adj","msc":"MSC"}[_rank_key]
            df_r=pd.DataFrame(rows).sort_values(_sort_col,ascending=_rank_asc).reset_index(drop=True)
            df_r.index+=1
            _grad_cmap = "YlGn_r" if _rank_asc else "YlGn"  # küçük=iyi ise ters renk skalası
            st.dataframe(df_r.style.background_gradient(subset=[_sort_col],cmap=_grad_cmap),use_container_width=True)
            best=df_r.iloc[0]
            st.markdown(
                f"<div class='info-banner'>Best fit (by {rank_choice}): <strong>{best['Model']}</strong> "
                f"— R2adj={best['R2adj']}, RMSE={best['RMSE']}, AICc={best['AICc']}, BIC={best['BIC']}</div>",
                unsafe_allow_html=True
            )

        # ── Release Time Points (Tx%) ────────────────────────────────────────
        # Time for each fitted model to reach 25/50/80/90% release, interpolated
        # within the measured time range (T50/T80 are regulatory-standard).
        with st.expander("Release Time Points (Tx%)"):
            _tx_rows = []
            for v in res_ok.values():
                _tx = v.get("tx") or {}
                _tx_rows.append({
                    "Model": v.get("name", ""),
                    f"T25 ({time_unit})": _rn(_tx.get("25"), 2) if _tx.get("25") is not None else "n/a",
                    f"T50 ({time_unit})": _rn(_tx.get("50"), 2) if _tx.get("50") is not None else "n/a",
                    f"T80 ({time_unit})": _rn(_tx.get("80"), 2) if _tx.get("80") is not None else "n/a",
                    f"T90 ({time_unit})": _rn(_tx.get("90"), 2) if _tx.get("90") is not None else "n/a",
                })
            if _tx_rows:
                _df_tx = pd.DataFrame(_tx_rows).reset_index(drop=True)
                _df_tx.index += 1
                st.dataframe(_df_tx, use_container_width=True)
            st.caption(
                "Time for the fitted model to reach each % release, interpolated "
                "within the measured time range; 'n/a' = not reached in range."
            )

        st.subheader("Dissolution Curves with Model Fits")
        fig,ax=plt.subplots(figsize=(10,5.5)); style_ax(fig,ax)
        ax.scatter(t_arr,r_arr,color=OXFORD,s=65,zorder=5,edgecolors="white",lw=0.8,label="Experimental")
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        t_sm=np.linspace(t_arr.min(),t_arr.max(),400)
        _curve_skip=[]
        for i,(mn,v) in enumerate(res_ok.items()):
            try:
                ct, cy = v.get("curve_t"), v.get("curve_y")
                if ct and cy:
                    xs, ys = ct, cy                              # backend-provided curve
                else:                                            # local fallback: recompute
                    _pv = list((v.get("params") or {}).values())
                    if not _pv or any((p is None or p != p) for p in _pv):
                        _curve_skip.append(mn); continue
                    xs, ys = t_sm, MODEL_DEFS[mn][0](t_sm, *_pv)
                _r2a = v.get("r2adj")
                _lbl = f"{mn} (R2adj={_r2a:.3f})" if isinstance(_r2a,(int,float)) and _r2a==_r2a else mn
                ax.plot(xs,ys,color=PALETTE[i%len(PALETTE)],lw=1.6,alpha=0.85,label=_lbl)
            except Exception:
                _curve_skip.append(mn)
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel("Cumulative Drug Released (%)")
        ax.set_xlim(left=0)
        ax.set_title(f"Kinetic Model Fitting — {pname}"); ax.set_ylim(0,112)
        # Legend outside the axes (right) so dense curve sets stay readable
        _ncol = 1 if (len(res_ok) + 1) <= 14 else 2
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  ncol=_ncol, framealpha=0.9, borderaxespad=0.0)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        if _curve_skip:
            st.caption(f"Curves not shown for: {', '.join(_curve_skip)} (non-finite parameters).")

        # Safe display formatter — None/NaN/inf → "N/A" (never crashes f-strings).
        def _fmt(x, n, g=False):
            try:
                xf = float(x)
                if xf != xf or xf in (float("inf"), float("-inf")):
                    return "N/A"
                return f"{xf:.{n}g}" if g else f"{xf:.{n}f}"
            except (TypeError, ValueError):
                return "N/A"

        st.subheader("Fitted Parameters")
        for mn,v in res_ok.items():
            with st.expander(f"{mn}  |  R2adj={_fmt(v.get('r2adj'),4)}  |  "
                             f"RMSE={_fmt(v.get('rmse'),3)}  |  AICc={_fmt(v.get('aicc'),2)}"):
                st.markdown(f"<div class='eq-box'>{v.get('equation','')}</div>",unsafe_allow_html=True)
                _eqf = v.get("equation_fitted")
                if _eqf:
                    st.caption(f"Fitted: {_eqf}")
                if v.get("bounds_enforced") is False:
                    st.caption("⚠️ Unconstrained fit — physical bounds could not be applied; "
                               "interpret parameters with caution.")
                _nanf = v.get("nan_fraction") or 0
                if _nanf > 0.10:
                    st.caption(f"⚠️ {_nanf:.0%} of model predictions are undefined — "
                               "fit may be unreliable.")
                _params = v.get("params") or {}
                _pci = v.get("param_ci") or {}
                cols=st.columns(min(4,max(1,len(_params))))
                for j,(pn,pv) in enumerate(_params.items()):
                    col = cols[j%4]
                    col.metric(pn, _fmt(pv,5,g=True))
                    ci = _pci.get(pn) or {}
                    lo, hi = ci.get("ci_low"), ci.get("ci_high")
                    if lo is not None and hi is not None:
                        col.caption(f"95% CI: {_fmt(lo,4,g=True)}–{_fmt(hi,4,g=True)}")
        if res_fail:
            st.warning(f"Did not converge: {', '.join(res_fail.keys())}")

        # ── Residual Diagnostics ─────────────────────────────────────────────
        # How well does the fit behave? Residual-vs-fitted + Q-Q, plus Shapiro-Wilk
        # (normality) and Wald-Wolfowitz runs (randomness) p-values from the engine.
        _diag_models = [m for m, v in res_ok.items()
                        if (v.get("diagnostics") or {}).get("residuals")]
        if _diag_models:
            with st.expander("Residual Diagnostics"):
                _default_idx = (_diag_models.index(_best)
                                if (_best in _diag_models) else 0)
                _dm = st.selectbox(
                    "Model", _diag_models, index=_default_idx,
                    key=f"resid_diag_model_{pname}",
                    help="Residuals = observed − fitted at each measured time point. "
                         "Good fits scatter randomly around 0 and look normal."
                )
                diag = res_ok[_dm].get("diagnostics") or {}
                _resid = diag.get("residuals") or []
                _fitted = diag.get("fitted") or []
                if _resid and _fitted and len(_resid) == len(_fitted):
                    _resid_a = np.asarray(_resid, dtype=float)
                    _fit_a = np.asarray(_fitted, dtype=float)

                    c1, c2 = st.columns(2)
                    with c1:
                        fig1, ax1 = plt.subplots(figsize=(5, 4)); style_ax(fig1, ax1)
                        ax1.axhline(0, color=AMBER, ls="--", lw=1.2, zorder=1)
                        ax1.scatter(_fit_a, _resid_a, color=OXFORD, s=45,
                                    edgecolors="white", lw=0.6, zorder=3)
                        ax1.set_xlabel("Fitted value (%)")
                        ax1.set_ylabel("Residual (%)")
                        ax1.set_title("Residuals vs Fitted")
                        fig1.tight_layout()
                        st.pyplot(fig1); plt.close(fig1)
                    with c2:
                        fig2, ax2 = plt.subplots(figsize=(5, 4)); style_ax(fig2, ax2)
                        try:
                            from scipy import stats as _sps
                            _sps.probplot(_resid_a, dist="norm", plot=ax2)
                        except Exception:
                            ax2.text(0.5, 0.5, "Q-Q plot unavailable",
                                     ha="center", va="center", transform=ax2.transAxes)
                        ax2.set_title("Normal Q-Q")
                        fig2.tight_layout()
                        st.pyplot(fig2); plt.close(fig2)

                    st.caption(
                        "Left: points scattered randomly around the dashed zero line "
                        "indicate a good fit; a curved/funnel pattern suggests the model "
                        "is mis-specified. Right: points near the diagonal mean the "
                        "residuals are approximately normal."
                    )

                    def _verdict(p, good_msg, warn_msg):
                        if p is None:
                            return "n/a (too few points)"
                        try:
                            pf = float(p)
                        except (TypeError, ValueError):
                            return "n/a (too few points)"
                        if pf != pf:
                            return "n/a (too few points)"
                        return (f"✅ {good_msg}" if pf > 0.05 else f"⚠️ {warn_msg}")

                    def _pv(p):
                        try:
                            pf = float(p)
                            return f"{pf:.3f}" if pf == pf else "n/a"
                        except (TypeError, ValueError):
                            return "n/a"

                    m1, m2 = st.columns(2)
                    _sp = diag.get("shapiro_p")
                    _rp = diag.get("runs_p")
                    m1.metric("Shapiro-Wilk p", _pv(_sp))
                    m1.caption("**Normality of residuals:** "
                               + _verdict(_sp, "residuals look normal",
                                          "residuals deviate from normal → check the model"))
                    m2.metric("Runs test p", _pv(_rp))
                    m2.caption("**Randomness of residuals:** "
                               + _verdict(_rp, "residuals look random",
                                          "residuals show structure → model may be mis-specified"))
                    st.caption(
                        f"Based on {diag.get('n_resid', len(_resid))} residual points. "
                        "p > 0.05 is good (no evidence of a problem); "
                        "p ≤ 0.05 is a warning."
                    )

    show_literature("Kinetic Model Fitting")

# ===========================================================================
# PAGE: STATISTICAL ANALYSIS
# ===========================================================================
