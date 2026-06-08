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

    if st.button("Run Model Fitting", type="primary") and selected_models:
        st.session_state.fit_results = {}
        prog = st.progress(0)
        for i,mn in enumerate(selected_models):
            st.session_state.fit_results[mn] = fit_model(t_arr, r_arr, mn)
            prog.progress((i+1)/len(selected_models))
        st.success(f"Fitting complete - {len(selected_models)} models processed.")

    if st.session_state.fit_results:
        res_ok   = {k:v for k,v in st.session_state.fit_results.items() if v["success"]}
        res_fail = {k:v for k,v in st.session_state.fit_results.items() if not v["success"]}

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
        rows=[{"Model":v["name"],"Category":v["category"],
               "R2":round(v["r2"],4),"R2adj":round(v["r2adj"],4),
               "RMSE":round(v["rmse"],3),
               "AIC":round(v["aic"],2),"AICc":round(v["aicc"],2),"BIC":round(v["bic"],2),
               "MSC":round(v["msc"],3),
               "Params":v["n_params"],"Reference":v["reference"]}
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

        st.subheader("Dissolution Curves with Model Fits")
        fig,ax=plt.subplots(figsize=(10,5.5)); style_ax(fig,ax)
        ax.scatter(t_arr,r_arr,color=OXFORD,s=65,zorder=5,edgecolors="white",lw=0.8,label="Experimental")
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        t_sm=np.linspace(t_arr.min(),t_arr.max(),400)
        for i,(mn,v) in enumerate(res_ok.items()):
            func=MODEL_DEFS[mn][0]; popt=list(v["params"].values())
            try:
                ys=func(t_sm,*popt)
                ax.plot(t_sm,ys,color=PALETTE[i%len(PALETTE)],lw=1.6,alpha=0.85,
                        label=f"{mn} (R2adj={v['r2adj']:.3f})")
            except: pass
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel("Cumulative Drug Released (%)")
        ax.set_xlim(left=0)
        ax.set_title(f"Kinetic Model Fitting — {pname}"); ax.set_ylim(0,112)
        ax.legend(fontsize=6.5,ncol=2,loc="lower right")
        st.pyplot(fig); plt.close()

        st.subheader("Fitted Parameters")
        for mn,v in res_ok.items():
            with st.expander(f"{mn}  |  R2adj={v['r2adj']:.4f}  |  RMSE={v['rmse']:.3f}  |  AICc={v['aicc']:.2f}"):
                st.markdown(f"<div class='eq-box'>{v['equation']}</div>",unsafe_allow_html=True)
                cols=st.columns(min(4,max(1,len(v["params"]))))
                for j,(pn,pv) in enumerate(v["params"].items()):
                    cols[j%4].metric(pn,f"{pv:.5g}")
        if res_fail:
            st.warning(f"Did not converge: {', '.join(res_fail.keys())}")

    show_literature("Kinetic Model Fitting")

# ===========================================================================
# PAGE: STATISTICAL ANALYSIS
# ===========================================================================
