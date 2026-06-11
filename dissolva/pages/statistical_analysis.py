"""DissolvA page module: Statistical Analysis. Extracted from app.py (Phase 3b modularization)."""
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
from dissolva.theme import OXFORD, AMBER, PALETTE, style_ax, FIGSIZE
from dissolva.models import (MODEL_DEFS, CATEGORIES, compute_mdt,
    compute_de, r2s, r2adj, aic_fn, msc_fn, _nz)
from dissolva.state import (current_tier, require_tier, _safe_profile_names,
    _get_index, _rename_profile, _clear_all)
from dissolva.content import show_literature, show_all_references, analyze_profile_shape


def render():
    cfg = st.session_state.method_cfg
    time_unit = cfg["time_unit"]
    q_time = cfg["q_time"]
    q_limit = cfg["q_limit"]
    st.header("Statistical Analysis")
    if not st.session_state.profiles:
        st.warning("No profiles loaded."); st.stop()

    names = list(st.session_state.profiles.keys())
    st.subheader("MDT and DE per Profile")
    rows=[]
    for nm in names:
        ta=np.array(st.session_state.profiles[nm]["time"])
        ra=np.array(st.session_state.profiles[nm]["release"])
        rows.append({"Profile":nm,
                     f"MDT ({time_unit})":round(compute_mdt(ta,ra),3),
                     "DE (%)":round(compute_de(ta,ra),3)})
    _df_mdt = pd.DataFrame(rows)
    _df_mdt.index = range(1, len(_df_mdt)+1)
    st.dataframe(_df_mdt, use_container_width=True)

    # ── Permeation metrics (Franz / IVPT) — shown for permeation methods or µg/cm² data ──
    _app = cfg.get("apparatus", "")
    _du = cfg.get("data_unit", "% released")
    if ("Franz" in _app) or (_du == "µg/cm²"):
        from dissolva.permeation import permeation_metrics
        st.subheader("Permeation Metrics — Jss · lag-time · Kp")
        _dc = float(cfg.get("franz_donor_conc", 0.0)) or None
        _tu = "h" if cfg.get("time_unit") == "hours" else "min"
        _unit = _du if _du != "% released" else "%"
        prows = []
        for nm in names:
            ta = np.array(st.session_state.profiles[nm]["time"], float)
            ra = np.array(st.session_state.profiles[nm]["release"], float)
            pm = permeation_metrics(ta, ra, donor_conc=_dc)
            prows.append({
                "Profile": nm,
                f"Jss ({_unit}/{_tu})": round(pm["jss"], 4) if pm["jss"] is not None else "—",
                f"Lag ({_tu})": round(pm["lag"], 3) if pm["lag"] is not None else "—",
                "Kp": round(pm["kp"], 5) if pm["kp"] is not None else "—",
                "R² (steady-state)": round(pm["r2"], 4) if pm["r2"] is not None else "—",
                "n pts": pm["n_points"],
            })
        _dfp = pd.DataFrame(prows)
        _dfp.index = range(1, len(_dfp) + 1)
        st.dataframe(_dfp, use_container_width=True)
        st.caption("Steady-state flux Jss = slope of the terminal linear region; lag-time = x-intercept; "
                   "Kp = Jss / donor concentration (set it in Method Settings → Apparatus). "
                   "Standard Fick's-law / Franz-cell analysis. Set Data Unit to µg/cm² for proper flux units.")

    st.subheader("Individual Profile Plots")
    # Visual options
    _is_cfg_s  = st.session_state.method_cfg
    _is_on_s   = _is_cfg_s.get("internal_spec_enabled", False)
    _isn_s     = _is_cfg_s.get("internal_spec_name",  "Internal Spec")
    _isl_s     = float(_is_cfg_s.get("internal_spec_limit", 85.0))

    # 4 columns if Internal Spec active, otherwise 3
    if _is_on_s:
        sp_c1, sp_c2, sp_c3, sp_c4 = st.columns(4)
    else:
        sp_c1, sp_c2, sp_c3 = st.columns(3)
        sp_c4 = None

    with sp_c1:
        show_q_line  = st.radio("Q Value Line", ["Show","Hide"], horizontal=True, key="stat_qline") == "Show"
    with sp_c2:
        show_qt_line = st.radio("Q Time Marker", ["Show","Hide"], horizontal=True, key="stat_qtline") == "Show"
    with sp_c3:
        show_eb      = st.radio("Error Bars (SD)", ["Show","Hide"], horizontal=True, key="stat_eb") == "Show"
    if _is_on_s and sp_c4:
        with sp_c4:
            show_is_stat = st.radio(
                f"{_isn_s} Line",
                ["Show", "Hide"], horizontal=True, key="stat_is_line"
            ) == "Show"
    else:
        show_is_stat = False
    _ist_s     = float(_is_cfg_s.get("internal_spec_time",  45.0))

    ncols=min(2,len(names)); cols=st.columns(ncols)
    for i,nm in enumerate(names):
        ta  = np.array(st.session_state.profiles[nm]["time"])
        ra  = np.array(st.session_state.profiles[nm]["release"])
        sda = np.array(st.session_state.profiles[nm].get("sd") or [0.0]*len(ta))
        has_sd = not np.all(sda == 0)
        fig,ax=plt.subplots(figsize=(5.5,3.8)); style_ax(fig,ax)
        ax.fill_between(ta, np.clip(ra-sda,0,None), ra+sda, alpha=0.10, color=PALETTE[i%len(PALETTE)])
        if has_sd and show_eb:
            ax.errorbar(ta, ra, yerr=sda, fmt="o-", color=PALETTE[i%len(PALETTE)],
                        lw=2, ms=6, capsize=4, capthick=1.5, elinewidth=1.2)
        else:
            ax.plot(ta, ra, "o-", color=PALETTE[i%len(PALETTE)], lw=2, ms=6)
        if show_q_line:
            ax.axhline(q_limit, color=AMBER, lw=1.3, ls="--", alpha=0.85,
                       label=f"Q = {q_limit:.0f}% (FDA/USP)")
        if show_qt_line:
            ax.axvline(q_time, color="#e74c3c", lw=1.2, ls=":", alpha=0.75,
                       label=f"Q-time = {q_time:.0f} {time_unit}")
        # Internal Spec line - on Statistics page
        if _is_on_s and show_is_stat:
            ax.axhline(_isl_s, color="#9467bd", lw=1.3, ls=(0,(5,3)), alpha=0.85,
                       label=f"{_isn_s} = {_isl_s:.0f}%")
            ax.axvline(_ist_s, color="#9467bd", lw=1.1, ls=(0,(3,3)), alpha=0.7,
                       label=f"{_isn_s} t = {_ist_s:.0f} {time_unit}")
        ax.set_title(nm, fontsize=11)
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel("Cumulative Drug Released (%)")
        ax.set_xlim(left=0, right=ta.max()*1.05)
        ax.set_ylim(bottom=0, top=112)
        if show_q_line or show_qt_line or (show_is_stat and _is_on_s):
            ax.legend(fontsize=7.5)
        cols[i%ncols].pyplot(fig); plt.close()

        # Internal Spec per-profile evaluation
        if _is_on_s and show_is_stat:
            _is_idx_s = np.where(np.isclose(ta, _ist_s))[0]
            if len(_is_idx_s) > 0:
                _is_val_s = ra[_is_idx_s[0]]
                if _is_val_s >= _isl_s:
                    cols[i%ncols].success(
                        f"✅ **{_isn_s} — PASS** · {nm}: "
                        f"{_is_val_s:.1f}% ≥ {_isl_s:.0f}% @ {_ist_s:.0f} {time_unit} "
                        f"*(Advisory — not a regulatory criterion)*"
                    )
                else:
                    cols[i%ncols].warning(
                        f"⚠️ **{_isn_s} — BELOW LIMIT** · {nm}: "
                        f"{_is_val_s:.1f}% < {_isl_s:.0f}% @ {_ist_s:.0f} {time_unit} "
                        f"*(Advisory only — not a regulatory finding)*"
                    )
            else:
                cols[i%ncols].info(
                    f"ℹ️ {_isn_s}: No data point at t={_ist_s:.0f} {time_unit} for {nm}."
                )

    # ── Dissolution Rate (dF/dt) — model-independent, observed data only ───
    st.markdown("---")
    st.subheader("Dissolution Rate (dF/dt)")
    fig_dr, ax_dr = plt.subplots(figsize=FIGSIZE); style_ax(fig_dr, ax_dr)
    _dr_plotted = 0
    for i, nm in enumerate(names):
        ta = np.array(st.session_state.profiles[nm]["time"], dtype=float)
        ra = np.array(st.session_state.profiles[nm]["release"], dtype=float)
        if len(ta) < 2:  # np.gradient needs >= 2 points
            continue
        rate = np.gradient(ra, ta)
        ax_dr.plot(ta, rate, "o-", color=PALETTE[i % len(PALETTE)],
                   lw=2, ms=5, label=nm)
        _dr_plotted += 1
    if _dr_plotted > 0:
        ax_dr.set_xlabel(f"Time ({time_unit})")
        ax_dr.set_ylabel(f"Release rate (%/{time_unit})")
        ax_dr.set_xlim(left=0)
        ax_dr.legend(fontsize=8)
        st.pyplot(fig_dr)
    plt.close(fig_dr)
    st.caption(
        "Model-independent dissolution rate (numerical derivative of the observed "
        "profile); peaks show the fastest-release phase."
    )

    # ── Per-Profile Statistics (moved from Data Input) ─────────────────────
    st.markdown("---")
    st.subheader("Per-Profile Statistics")
    st.caption(
        "Per-profile vessel-based statistics. "
        "SD and RSD calculated from raw vessel data."
    )

    for nm, d in st.session_state.profiles.items():
        with st.expander(
            f"{nm}  |  n={d.get('n',1)} vessels  |  {len(d['time'])} time points",
            expanded=True
        ):
            t_v   = np.array(d["time"])
            r_v   = np.array(d["release"])
            sd_v  = np.array(d["sd"])  if d.get("sd")  is not None else np.zeros(len(t_v))
            rsd_v = np.array(d["rsd"]) if d.get("rsd") is not None else np.zeros(len(t_v))

            df_show = pd.DataFrame({
                f"Time ({time_unit})": t_v.round(2),
                "Mean (%)":  r_v.round(2),
                "SD":        sd_v.round(3),
                "RSD (%)":   rsd_v.round(2),
                "CV (%)":    rsd_v.round(2),
                "n":         d.get("n", 1),
            })
            st.dataframe(
                df_show.style.background_gradient(subset=["RSD (%)"], cmap="RdYlGn_r"),
                use_container_width=True
            )

            # RSD-based Bootstrap assessment
            # ─────────────────────────────────────────────────────────────────
            # GUIDELINE DEFINITION (approved 14-04-2026):
            #   FDA 1997  → "earlier time points (e.g. 15 min)" → t ≤ 15 min: CV ≤ 20%
            #              → other points (t > 15 min)         : CV ≤ 10%
            #   EMA 2010  → "first time point" → unofficial: t ≤ 10 min: CV ≤ 20%
            #   EMA RP2017→ "> 20% RSD at time-points ≤ 10 min" (numerical clarification)
            # OLD (incorrect): ref<85% → early, ref≥85% → late
            # CORRECT: TIME-BASED threshold
            # ─────────────────────────────────────────────────────────────────
            has_sd = not np.all(sd_v == 0)
            if has_sd and len(rsd_v) > 0:
                # FDA: %20 erken noktalarda (≤15 dk) VE en erken örnekleme
                # noktasında (ilk nokta t>15 olsa bile); diğerlerinde %10.
                early_mask = (t_v <= 15.0) | (t_v == t_v.min())
                late_mask  = ~early_mask
                rsd_early_arr = rsd_v[early_mask]
                rsd_late_arr  = rsd_v[late_mask]
                cv_early_max = float(np.max(rsd_early_arr)) if len(rsd_early_arr) > 0 else 0.0
                cv_late_max  = float(np.max(rsd_late_arr))  if len(rsd_late_arr)  > 0 else 0.0

                # FDA criteria: early point CV < 20%, others < 10%
                fda_ok_early = cv_early_max <= 20.0
                fda_ok_late  = cv_late_max  <= 10.0
                fda_ok       = fda_ok_early and fda_ok_late
                n_vessels    = d.get("n", 1)

                st.markdown("##### 📊 FDA CV Criteria Assessment (f2 Eligibility)")
                cv_c1, cv_c2, cv_c3 = st.columns(3)
                cv_c1.metric(
                    "Max CV% — Early points (≤15 min | FDA)",
                    f"{cv_early_max:.1f}%",
                    "✓ ≤ 20% (FDA)" if fda_ok_early else "✗ > 20% (FDA)",
                    delta_color="normal" if fda_ok_early else "inverse"
                )
                cv_c2.metric(
                    "Max CV% — Late points (>15 min | FDA)",
                    f"{cv_late_max:.1f}%",
                    "✓ ≤ 10% (FDA)" if fda_ok_late else "✗ > 10% (FDA)",
                    delta_color="normal" if fda_ok_late else "inverse"
                )
                cv_c3.metric("Vessels (n)", n_vessels,
                    "✓ ≥ 12 (FDA ideal)" if n_vessels >= 12 else f"⚠️ < 12")

                if fda_ok and n_vessels >= 6:
                    st.success(
                        f"**✅ {nm} — Standard f2 Test is Sufficient.**\n\n"
                        f"CV% values meet FDA (1997) criteria "
                        f"(FDA: t ≤ 15 min → CV ≤ 20%; t > 15 min → CV ≤ 10%). "
                        f"Bootstrap f2 analysis is not required; "
                        f"the **single-point f2 test** is conclusive."
                    )
                else:
                    _reasons = []
                    if not fda_ok_early:
                        _reasons.append(f"Early-point CV% = {cv_early_max:.1f}% > 20%")
                    if not fda_ok_late:
                        _reasons.append(f"Late-point CV% = {cv_late_max:.1f}% > 10%")
                    if n_vessels < 6:
                        _reasons.append(f"n = {n_vessels} vessel < 6")
                    st.warning(
                        f"**⚠️ {nm} — Bootstrap f2 Assessment Recommended.**\n\n"
                        f"Reason: {'; '.join(_reasons)}.\n\n"
                        f"FDA (1997): When CV% criteria are exceeded, "
                        f"**Bootstrap f2** or **Multivariate Confidence Region** analysis is recommended."
                    )

            # Raw vessel data
            if d.get("raw") and d.get("vessels"):
                raw_df = pd.DataFrame(d["raw"], columns=d["vessels"])
                raw_df.insert(0, f"Time ({time_unit})", t_v)
                with st.expander("Raw vessel data"):
                    st.dataframe(raw_df.style.format(precision=2), use_container_width=True)

    show_literature("Statistical Analysis")

# ===========================================================================
# PAGE: f1 & f2
# ===========================================================================
