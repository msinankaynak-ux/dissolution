"""DissolvA page module: Method Settings. Extracted from app.py (Phase 3b modularization)."""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
try:
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False
from dissolva.theme import OXFORD, AMBER, PALETTE, style_ax
from dissolva.models import (MODEL_DEFS, CATEGORIES, compute_mdt,
    compute_de, r2s, r2adj, aic_fn, msc_fn, _nz)
from dissolva.state import (current_tier, require_tier, _safe_profile_names,
    _get_index, _rename_profile, _clear_all)
from dissolva.content import show_literature, show_all_references, analyze_profile_shape


def render():
    cfg = st.session_state.method_cfg
    time_unit = cfg["time_unit"]
    conc_unit = cfg["conc_unit"]
    dose_mg = cfg["dose_mg"]
    q_time = cfg["q_time"]
    q_limit = cfg["q_limit"]
    cfg = st.session_state.method_cfg
    st.markdown(
        "<h2 style='color:#FFFFFF;margin:0 0 4px;'>Method & Parameter Settings</h2>"
        "<p style='color:#9fb0d0;font-size:0.88rem;margin:0 0 20px;'>"
        "General parameters, dissolution apparatus and medium conditions. "
        "All settings are saved automatically and included in the Excel report.</p>",
        unsafe_allow_html=True
    )

    tab_gen, tab_q, tab_app = st.tabs(
        ["⚙️ General", "✅ Acceptance (Q)", "\U0001F9EA Apparatus & Medium"])

    # ── TAB 1: General Parameters ────────────────────────────────────────────
    with tab_gen:
        c1, c2, c3 = st.columns(3)
        with c1:
            cfg["time_unit"] = st.selectbox("Time Unit",
                ["minutes", "hours"], index=["minutes","hours"].index(cfg["time_unit"]))
        with c2:
            cfg["conc_unit"] = st.selectbox("Concentration Unit",
                ["mg/mL", "ug/mL", "mg/L"], index=["mg/mL","ug/mL","mg/L"].index(cfg["conc_unit"]))
        with c3:
            cfg["dose_mg"] = st.number_input("Dose (mg)", value=float(cfg["dose_mg"]), min_value=0.1)

    # ── TAB 2: FDA/USP Acceptance Criterion (Q) ──────────────────────────────
    with tab_q:
        c4, c5 = st.columns(2)
        with c4:
            cfg["q_time"] = st.number_input(
                "Q Time Point", value=float(cfg["q_time"]), min_value=0.0,
                help="Time point for Q criterion evaluation (e.g. 45 min for IR)")
        with c5:
            cfg["q_limit"] = st.number_input(
                "Q Value (%)", value=float(cfg["q_limit"]), min_value=0.0, max_value=100.0,
                help="Minimum % dissolved at Q-time (default 80% per USP <711>)")
        ql = cfg['q_limit']; qt = cfg['q_time']; tu = cfg['time_unit']
        st.markdown(f'<div class="info-banner">NLT <strong>{ql:.0f}%</strong> dissolved at <strong>{qt:.0f} {tu}</strong> &nbsp;|&nbsp; USP &lt;711&gt; / FDA 1997</div>', unsafe_allow_html=True)

        # ── Internal Spec (In-house Criterion) ──
        with st.expander(
            "⚙️ Internal Acceptance Criterion *(optional — click to expand/collapse)*",
            expanded=cfg.get("internal_spec_enabled", False)
        ):
            cfg["internal_spec_enabled"] = st.toggle(
                "Enable Internal Spec",
                value=cfg.get("internal_spec_enabled", False),
                key="is_toggle_method",
                help="Company internal criterion — advisory only, not a regulatory requirement."
            )
            if cfg["internal_spec_enabled"]:
                is_c1, is_c2, is_c3 = st.columns(3)
                with is_c1:
                    cfg["internal_spec_name"] = st.text_input(
                        "Criterion Name",
                        value=cfg.get("internal_spec_name", "Internal Spec"),
                        help="e.g. 'Internal Spec', 'Company Q', 'Release Limit'",
                        key="is_name_method")
                with is_c2:
                    cfg["internal_spec_time"] = st.number_input(
                        "Time Point",
                        value=float(cfg.get("internal_spec_time", 45.0)),
                        min_value=0.0, key="is_time_method")
                with is_c3:
                    cfg["internal_spec_limit"] = st.number_input(
                        "Limit (%)",
                        value=float(cfg.get("internal_spec_limit", 85.0)),
                        min_value=0.0, max_value=100.0, key="is_limit_method")
                isl = cfg["internal_spec_limit"]
                ist = cfg["internal_spec_time"]
                isn = cfg["internal_spec_name"]
                st.markdown(
                    f'<div style="background:rgba(148,103,189,0.08);border-left:3px solid #9467bd;' +
                    f'border-radius:0 6px 6px 0;padding:10px 14px;font-size:0.83rem;margin-top:8px;">' +
                    f'<strong style="color:#9467bd;">ℹ️ Advisory only</strong> — ' +
                    f'{isn}: NLT <strong>{isl:.0f}%</strong> dissolved at ' +
                    f'<strong>{ist:.0f} {tu}</strong>. ' +
                    f'Non-compliance triggers a warning — not a regulatory finding.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.caption("Toggle on to define a company internal release limit. "
                           "Shown as a purple dashed line on dissolution charts — advisory only.")

    # ── TAB 3: Dissolution Apparatus & Medium ────────────────────────────────
    with tab_app:
        # ── Study system + release/diffusion method (literature-based) ─────
        sm1, sm2 = st.columns(2)
        with sm1:
            system_opts = [
                "Solid oral (tablet/capsule)", "Powder / granule / pellet",
                "Suspension", "Modified / extended release",
                "Transdermal / patch", "Topical semisolid (cream/gel/ointment)",
                "Nanoparticle / liposome / micelle", "Implant / depot", "Other"
            ]
            cur_sys = cfg.get("system", system_opts[0])
            if cur_sys not in system_opts:
                cur_sys = "Other"
            cfg["system"] = st.selectbox(
                "Study System / Dosage Form", system_opts,
                index=system_opts.index(cur_sys),
                help="The kind of system studied. Guides which release/diffusion methods are typical.")
        with sm2:
            apparatus_opts = [
                "USP I (Basket)", "USP II (Paddle)",
                "USP III (Reciprocating Cylinder)",
                "USP IV (Flow-Through Cell)",
                "USP V (Paddle over Disk)", "USP VI (Cylinder)",
                "USP VII (Reciprocating Holder)",
                "Franz Diffusion Cell (IVRT/IVPT)",
                "Dialysis Membrane (bag/sac)", "Reverse Dialysis",
                "Sample-and-Separate",
                "Continuous Flow / Flow-Through (non-USP)", "Other"
            ]
            cur_app = cfg.get("apparatus", "USP II (Paddle)")
            if cur_app not in apparatus_opts:
                cur_app = "Other"
            cfg["apparatus"] = st.selectbox(
                "Release / Diffusion Method", apparatus_opts,
                index=apparatus_opts.index(cur_app),
                help="USP apparatus or membrane/diffusion method. Pick 'Other' to type your own.")

        if cfg["apparatus"] == "Other":
            cfg["apparatus_custom"] = st.text_input(
                "Specify method", value=cfg.get("apparatus_custom", ""),
                placeholder="e.g. adapted USP-IV, custom diffusion cell, in situ method...")

        # ── Method-specific parameters (shown only when relevant) ──
        _app = cfg["apparatus"]
        if "Franz" in _app:
            st.markdown("**Franz Diffusion Cell Parameters**")
            fz1, fz2, fz3 = st.columns(3)
            with fz1:
                cfg["franz_area_cm2"] = st.number_input(
                    "Diffusion Area (cm\u00b2)", value=float(cfg.get("franz_area_cm2", 1.0)),
                    min_value=0.0, step=0.1)
            with fz2:
                cfg["franz_receptor_ml"] = st.number_input(
                    "Receptor Volume (mL)", value=float(cfg.get("franz_receptor_ml", 12.0)),
                    min_value=0.0, step=0.5)
            with fz3:
                cfg["membrane"] = st.text_input(
                    "Membrane", value=cfg.get("membrane", ""),
                    placeholder="e.g. synthetic / skin / epidermis")
            st.caption("Permeation data are usually cumulative amount per area (\u00b5g/cm\u00b2); "
                       "steady-state flux (Jss), permeability (Kp) and lag-time can be derived. "
                       "Amount-based input units are coming to Data Input.")
        elif "Dialysis" in _app:
            st.markdown("**Dialysis Parameters**")
            dz1, dz2 = st.columns(2)
            with dz1:
                cfg["mwco_kda"] = st.number_input(
                    "Membrane MWCO (kDa)", value=float(cfg.get("mwco_kda", 12.0)),
                    min_value=0.0, step=0.5)
            with dz2:
                cfg["membrane"] = st.text_input(
                    "Membrane / bag", value=cfg.get("membrane", ""),
                    placeholder="e.g. cellulose, Float-A-Lyzer")
        elif ("Flow-Through" in _app) or ("Continuous Flow" in _app):
            cfg["flow_rate_ml_min"] = st.number_input(
                "Flow Rate (mL/min)", value=float(cfg.get("flow_rate_ml_min", 8.0)),
                min_value=0.0, step=0.5)

        st.markdown("---")
        medium_opts = [
            "0.1N HCl (pH 1.2)", "Acetate Buffer (pH 4.5)",
            "Phosphate Buffer (pH 6.8)", "Phosphate Buffer (pH 7.4)",
            "Purified Water", "SGF (Simulated Gastric Fluid)",
            "SIF (Simulated Intestinal Fluid)", "FaSSIF", "FeSSIF", "Other"
        ]
        cur_med = cfg.get("medium", "0.1N HCl (pH 1.2)")
        if cur_med not in medium_opts:
            cur_med = "Other"
        cfg["medium"] = st.selectbox("Dissolution / Receptor Medium",
            medium_opts, index=medium_opts.index(cur_med))

        if cfg["medium"] == "Other":
            cfg["medium_custom"] = st.text_input(
                "Specify Medium", value=cfg.get("medium_custom", ""),
                placeholder="e.g. Phosphate Buffer pH 7.2")

        st.markdown("**Additional Dissolution Agent (Surfactant etc.)**")
        ca1, ca2, ca3 = st.columns(3)
        with ca1:
            surfactant_opts = [
                "None", "SLS (Sodium Lauryl Sulfate)",
                "Tween 80", "Poloxamer 188", "CTAB", "Other"
            ]
            cur_surf = cfg.get("surfactant", "None")
            if cur_surf not in surfactant_opts:
                cur_surf = "Other"
            cfg["surfactant"] = st.selectbox(
                "Agent", surfactant_opts,
                index=surfactant_opts.index(cur_surf))
        with ca2:
            cfg["surfactant_conc"] = st.number_input(
                "Concentration (%)", value=float(cfg.get("surfactant_conc", 0.0)),
                min_value=0.0, max_value=5.0, step=0.05,
                help="e.g. 0.5% SLS, 1% Tween 80")
        with ca3:
            if cfg["surfactant"] == "Other":
                cfg["surfactant_custom"] = st.text_input(
                    "Specify Agent", value=cfg.get("surfactant_custom", ""))

        c3, c4, c5 = st.columns(3)
        with c3:
            cfg["rpm"] = st.number_input(
                "Rotation Speed (rpm)", value=int(cfg.get("rpm", 50)),
                min_value=1, max_value=300, step=5)
        with c4:
            cfg["volume_ml"] = st.number_input(
                "Medium Volume (mL)", value=int(cfg.get("volume_ml", 900)),
                min_value=100, max_value=4000, step=100)
        with c5:
            cfg["temp_c"] = st.number_input(
                "Temperature (°C)", value=float(cfg.get("temp_c", 37.0)),
                min_value=20.0, max_value=50.0, step=0.5)

        cfg["notes"] = st.text_area(
            "Additional Method Notes",
            value=cfg.get("notes", ""), height=100,
            placeholder="e.g. Sinker used, sampling times, filter type...")

    st.session_state.method_cfg = cfg
    time_unit = cfg["time_unit"]
    conc_unit = cfg["conc_unit"]
    dose_mg   = cfg["dose_mg"]
    q_time    = cfg["q_time"]
    q_limit   = cfg["q_limit"]
    st.success("Settings saved automatically.")

# ===========================================================================
# PAGE: ANALYTICAL SETTINGS
# ===========================================================================
