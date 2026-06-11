"""DissolvA page module: Bootstrap f2 Analysis. Extracted from app.py (Phase 3b modularization)."""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import warnings
try:
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
from scipy.optimize import curve_fit, root
from scipy.stats import norm as sp_norm
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from dissolva.theme import OXFORD, AMBER, PALETTE, style_ax
from dissolva.models import (MODEL_DEFS, CATEGORIES, compute_mdt,
    compute_de, r2s, r2adj, aic_fn, msc_fn, _nz, fda_f2_mask, f2_score)
from dissolva import engine_client
from dissolva.state import (current_tier, require_tier, _safe_profile_names,
    _get_index, _rename_profile, _clear_all)
from dissolva.content import (show_literature, show_all_references,
    analyze_profile_shape, bootstrap_recommendation)


def render():
    _PLOTLY_OK = _PLOTLY_AVAILABLE
    cfg = st.session_state.method_cfg
    time_unit = cfg["time_unit"]

    # ---- Page header --------------------------------------------------------
    st.markdown(
        "<h2 style='color:#FFFFFF;margin:0 0 4px;'>Bootstrap f2 Analysis</h2>"
        "<p style='color:#9fb0d0;font-size:0.88rem;margin:0 0 12px;'>"
        "FDA-compliant bootstrap simulation for the f2 similarity factor. "
        "Reports the <strong>5th percentile</strong> (lower bound of 90% CI) "
        "as required by FDA. Requires raw vessel data (Excel upload mode).</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='step-box'>"
        "<strong>FDA Bootstrap Criterion (Shah et al. 1998):</strong><br>"
        "1. Resample with replacement from each formulation's individual vessel data.<br>"
        "2. Calculate f2 for each bootstrap iteration.<br>"
        "3. Report the <strong>5th percentile</strong> of the f2 distribution "
        "(lower bound of 90% CI).<br>"
        "4. If this value is <strong>≥ 50</strong>, profiles are considered similar."
        "</div>",
        unsafe_allow_html=True
    )

    # ---- Guard: need ≥2 profiles with raw data ------------------------------
    if len(st.session_state.profiles) < 2:
        st.warning(
            "At least 2 profiles with raw vessel data are required. "
            "Upload data via 'Excel / CSV Upload (Raw Vessel Data)' in Data Input."
        )
        st.stop()

    profiles_with_raw = {
        nm: d for nm, d in st.session_state.profiles.items()
        if d.get("raw") and d.get("vessels") and len(d.get("vessels", [])) >= 2
    }
    if len(profiles_with_raw) < 2:
        st.warning(
            "Bootstrap analysis requires **raw vessel-level data** for at least 2 profiles. "
            f"Profiles with raw data: **{list(profiles_with_raw.keys()) or 'None'}**. "
            "Please upload data using 'Excel / CSV Upload (Raw Vessel Data)' mode."
        )
        st.stop()

    names_raw = list(profiles_with_raw.keys())

    # ── Bootstrap method selection ──────────────────────────────────────────────
    bs_method = st.radio(
        "Bootstrap Method",
        ["Parametric (standard)", "Nonparametric (Shah 1998 — recommended for CV > 15%)"],
        horizontal=True,
        key="bs_method_radio",
        help=(
            "Parametric: Current standard method — resampling with distribution assumption. "
            "Nonparametric: No distribution assumption; Shah 1998 original recommendation. "
            "For CV > 15%, nonparametric provides more reliable CI."
        )
    )
    if "Nonparametric" in bs_method:
        st.info(
            "ℹ️ **Nonparametric Bootstrap (Shah 1998):** Vessel data were resampled without "
            "any distribution assumption. "
            "May yield different CI than parametric method when CV > 15%. "
            "*Shah VP et al. Pharm Res. 1998;15(6):889-896*"
        )

    # ---- Profile selection --------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Reference Profile** *(innovator / originator)*")
        def _on_ref_bs(): st.session_state.selected_ref_id = st.session_state._bs_ref_sel
        def _on_test_bs(): st.session_state.selected_test_id = st.session_state._bs_test_sel

        ref_bs = st.selectbox("Reference", names_raw,
            index=_get_index(names_raw, st.session_state.selected_ref_id, 0),
            key="_bs_ref_sel", on_change=_on_ref_bs,
            label_visibility="collapsed")
        n_ref = st.session_state.profiles[ref_bs].get("n", 0)
        st.caption(f"{ref_bs} — {n_ref} vessels")
    with col2:
        st.markdown("**Test Profile** *(your formulation)*")
        test_options = [nm for nm in names_raw if nm != ref_bs]
        if not test_options:
            st.error("Need at least 2 profiles with raw data."); st.stop()
        test_bs = st.selectbox("Test", test_options,
            index=_get_index(test_options, st.session_state.selected_test_id, 0),
            key="_bs_test_sel", on_change=_on_test_bs,
            label_visibility="collapsed")
        n_test = st.session_state.profiles[test_bs].get("n", 0)
        st.caption(f"{test_bs} — {n_test} vessels")

    # ---- Pre-run diagnostic: is bootstrap warranted for THIS data? ----------
    _rec = bootstrap_recommendation(st.session_state.profiles, ref_bs, test_bs)
    if not _rec["has_cv"]:
        st.info(
            "ℹ️ **No CV/RSD data** — these profiles have no vessel-level variability (CV%), so the "
            "FDA CV criteria cannot be checked automatically. Bootstrap still runs from the raw "
            "vessel data; assess variability manually when interpreting the result."
        )
    elif _rec["needs_boot"]:
        st.warning(
            "⚠️ **Bootstrap f2 is RECOMMENDED for your data** — the standard single-point f2 test is insufficient:\n\n"
            + "\n".join(f"- {r}" for r in _rec["reasons"])
            + f"\n\n**Recommended method: {_rec['recommended_method']} Bootstrap.** "
            + ("Because CV% > 15, Nonparametric gives a more reliable CI. "
               if _rec["cv_max"] > 15 else "CV% ≤ 15, so Parametric is sufficient. ")
            + "📌 *Shah VP et al. Pharm Res. 1998;15(6):889-896 | FDA Guidance 1997*"
        )
    else:
        _f2txt = f"f2 = {_rec['f2']:.2f} (outside the boundary zone), " if _rec["f2"] is not None else ""
        st.success(
            "✅ **Bootstrap not required — the standard f2 test is sufficient.** FDA criteria are met: "
            f"{_f2txt}n = {_rec['n_vessels']} vessels, "
            f"early CV% ≤ {_rec['cv_early_max']:.1f} (≤20 ✓), late CV% ≤ {_rec['cv_late_max']:.1f} (≤10 ✓). "
            "You may still run bootstrap for confirmation."
        )

    st.markdown("---")

    # ---- Parameters ---------------------------------------------------------
    col3, col4, col5 = st.columns(3)
    with col3:
        n_iter = st.number_input(
            "Bootstrap Iterations", min_value=1000, max_value=50000,
            value=5000, step=500,
            help="FDA recommends ≥ 2000; 5000 is standard practice."
        )
    with col4:
        lower_pctile = st.selectbox(
            "CI Lower Bound Percentile",
            [5.0, 2.5, 0.5],
            index=0,
            format_func=lambda x: f"{x}th pctl (→ {int(100-x*2)}% CI)" if x != 5.0 else "5th pctl (→ 90% CI) — FDA",
            help="FDA standard is 5th percentile (90% CI)."
        )
    with col5:
        seed_val = st.number_input(
            "Random Seed", min_value=0, max_value=99999, value=42,
            help="0 = different result each run; any other integer = reproducible."
        )

    # ---- Run button ---------------------------------------------------------
    if st.button("▶  Run Bootstrap Simulation", type="primary"):

        d_ref  = st.session_state.profiles[ref_bs]
        d_test = st.session_state.profiles[test_bs]

        t_ref_arr  = np.array(d_ref["time"],  dtype=float)
        t_test_arr = np.array(d_test["time"], dtype=float)
        raw_ref    = np.array(d_ref["raw"],   dtype=float)   # (n_tp_ref,  n_v_ref)
        raw_test   = np.array(d_test["raw"],  dtype=float)   # (n_tp_test, n_v_test)

        t_common = np.intersect1d(t_ref_arr, t_test_arr)
        if len(t_common) == 0:
            st.error("No common time points between the two profiles."); st.stop()

        # Observed f2 (mean profiles) — FDA 85% rule (first point exceeding 85% included)
        rr_obs = np.array([raw_ref[np.where(t_ref_arr == ti)[0][0], :].mean()
                           for ti in t_common])
        rt_obs = np.array([raw_test[np.where(t_test_arr == ti)[0][0], :].mean()
                           for ti in t_common])
        mask_obs = fda_f2_mask(rr_obs)
        if not mask_obs.any():
            st.error("No valid time points where reference release ≤ 85%."); st.stop()
        f2_obs = f2_score(rr_obs[mask_obs], rt_obs[mask_obs])

        # Subset raw matrices to common time points
        ref_idx  = [np.where(t_ref_arr  == ti)[0][0] for ti in t_common]
        test_idx = [np.where(t_test_arr == ti)[0][0] for ti in t_common]
        raw_ref_common  = raw_ref[ref_idx,  :]
        raw_test_common = raw_test[test_idx, :]

        # Run bootstrap over FIXED observed evaluation points (mask_obs) — gözlem
        # ile bootstrap aynı noktaları kullanır (Shah 1998 / EMA yöntemi).
        _method = "nonparametric" if "Nonparametric" in bs_method else "parametric"
        prog = st.progress(0, text="Running bootstrap simulation…")
        def _prog(frac, i):
            prog.progress(frac, text=f"Iteration {i:,} / {int(n_iter):,}…")
        _bs = engine_client.bootstrap(
            d_ref["time"], d_ref["raw"], d_test["time"], d_test["raw"],
            method=_method, iterations=int(n_iter), seed=int(seed_val),
            lower_pctile=float(lower_pctile), progress=_prog,
        )
        prog.empty()

        _dist = _bs.get("distribution") or []
        f2_boot = np.asarray(_dist, dtype=float)
        f2_boot = f2_boot[np.isfinite(f2_boot)] if f2_boot.ndim else np.array([])
        if f2_boot.size == 0:
            st.error("Bootstrap produced no valid iterations for these profiles. "
                     "Check that both profiles have raw vessel data and overlapping time points.")
            st.stop()
        f2_lower   = _bs.get("f2_lower")
        f2_upper   = _bs.get("f2_upper")
        f2_mean    = _bs.get("f2_mean")
        f2_med     = _bs.get("f2_median")
        f2_sd      = _bs.get("f2_sd")
        is_similar = bool(_bs.get("similar"))

        # ---- Key metric card (big) ------------------------------------------
        verdict_color = "#c6efce" if is_similar else "#ffc7ce"
        verdict_icon  = "✅ SIMILAR" if is_similar else "❌ NOT SIMILAR"
        st.markdown(
            f"<div style='background:{verdict_color};border-radius:8px;"
            f"padding:16px 22px;font-size:1.15rem;font-weight:700;margin:14px 0;"
            f"border-left:6px solid {'#27ae60' if is_similar else '#e74c3c'};'>"
            f"{verdict_icon} &nbsp;|&nbsp; "
            f"FDA Decision Point — Lower {lower_pctile:.0f}th Percentile f2 = "
            f"<span style='font-size:1.4rem;'><strong>{f2_lower:.2f}</strong></span> "
            f"({'≥' if is_similar else '<'} 50)</div>",
            unsafe_allow_html=True
        )

        # ---- Metric row -----------------------------------------------------
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Observed f2",                  f"{f2_obs:.2f}")
        mc2.metric("Bootstrap Mean f2",             f"{f2_mean:.2f}")
        mc3.metric("Bootstrap Median f2",           f"{f2_med:.2f}")
        mc4.metric(f"{lower_pctile:.0f}th Pctl (Lower CI ← FDA)", f"{f2_lower:.2f}")
        mc5.metric("Bootstrap SD",                  f"{f2_sd:.2f}")

        # ---- Summary table --------------------------------------------------
        st.markdown("#### Bootstrap Summary Statistics")
        df_summary = pd.DataFrame({
            "Statistic": [
                "Observed f2 (mean profiles)",
                "Bootstrap Mean f2",
                "Bootstrap Median f2",
                "Bootstrap SD",
                f"Lower {lower_pctile:.0f}th Percentile  ← FDA Decision Point",
                f"Upper {100-lower_pctile:.0f}th Percentile",
                "Valid iterations (n)",
                "FDA Similarity Verdict",
            ],
            "Value": [
                f"{f2_obs:.4f}",
                f"{f2_mean:.4f}",
                f"{f2_med:.4f}",
                f"{f2_sd:.4f}",
                f"{f2_lower:.4f}",
                f"{f2_upper:.4f}",
                f"{len(f2_boot):,}",
                verdict_icon,
            ]
        })
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        # ---- Interactive Plotly Distplot (Histogram + KDE) ------------------
        st.markdown("#### f2 Bootstrap Distribution")

        if _PLOTLY_OK:
            try:
                # ff.create_distplot: histogram + KDE overlay
                fig_dist = ff.create_distplot(
                    [f2_boot.tolist()],
                    group_labels=["Bootstrap f2"],
                    bin_size=max(0.5, (f2_boot.max() - f2_boot.min()) / 60),
                    colors=[OXFORD],
                    show_rug=False,
                )

                # Threshold line at 50
                fig_dist.add_vline(
                    x=50, line_width=2.5, line_dash="dash", line_color=AMBER,
                    annotation_text="f2 = 50 (FDA Threshold)",
                    annotation_position="top right",
                    annotation_font_color=AMBER,
                )
                # Lower CI line
                fig_dist.add_vline(
                    x=f2_lower, line_width=2, line_dash="dot",
                    line_color="#e74c3c",
                    annotation_text=f"{lower_pctile:.0f}th Pctl = {f2_lower:.2f}",
                    annotation_position="top left",
                    annotation_font_color="#e74c3c",
                )
                # Observed f2 line
                fig_dist.add_vline(
                    x=f2_obs, line_width=2, line_dash="solid",
                    line_color="#27ae60",
                    annotation_text=f"Observed f2 = {f2_obs:.2f}",
                    annotation_position="top right",
                    annotation_font_color="#27ae60",
                    annotation_yshift=30,
                )
                # Shade "fail" zone
                x_min_h = float(f2_boot.min())
                fig_dist.add_vrect(
                    x0=x_min_h, x1=min(50.0, float(f2_boot.max())),
                    fillcolor="#e74c3c", opacity=0.07,
                    layer="below", line_width=0,
                )

                fig_dist.update_layout(
                    title=dict(
                        text=(
                            f"Bootstrap f2 Distribution — {ref_bs} vs {test_bs}<br>"
                            f"<sup>{len(f2_boot):,} iterations | "
                            f"Lower {lower_pctile:.0f}th Pctl = {f2_lower:.2f} | "
                            f"{'SIMILAR ✓' if is_similar else 'NOT SIMILAR ✗'}</sup>"
                        ),
                        font=dict(color=OXFORD, size=15),
                    ),
                    xaxis_title="f2 Value",
                    yaxis_title="Density / Frequency",
                    plot_bgcolor="#F8F4EC",
                    paper_bgcolor="#FDFAF5",
                    font=dict(family="EB Garamond, Georgia, serif", color=OXFORD),
                    legend=dict(bgcolor="rgba(255,255,255,0.8)"),
                    xaxis=dict(gridcolor="#e0dbd0"),
                    yaxis=dict(gridcolor="#e0dbd0"),
                    height=460,
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            except Exception as _plot_err:
                st.warning(f"Plotly distplot could not render ({_plot_err}). "
                           "Falling back to matplotlib histogram.")
                _PLOTLY_OK = False   # fall through to matplotlib

        if not _PLOTLY_OK:
            # Matplotlib fallback
            fig_fb, ax_fb = plt.subplots(figsize=(10, 4.5))
            style_ax(fig_fb, ax_fb)
            ax_fb.hist(f2_boot, bins=60, color=OXFORD, alpha=0.78, edgecolor="white",
                       lw=0.4, label="Bootstrap f2")
            ax_fb.axvline(50,        color=AMBER,    lw=2.2, ls="--", label="f2=50 (FDA)")
            ax_fb.axvline(f2_lower,  color="#e74c3c", lw=1.8, ls=":",
                          label=f"{lower_pctile:.0f}th Pctl = {f2_lower:.2f}")
            ax_fb.axvline(f2_obs,    color="#27ae60", lw=1.8, ls="-",
                          label=f"Observed f2 = {f2_obs:.2f}")
            ax_fb.set_xlabel("f2 Value"); ax_fb.set_ylabel("Frequency")
            ax_fb.set_title(f"Bootstrap f2 Distribution — {ref_bs} vs {test_bs}")
            ax_fb.legend(fontsize=9)
            st.pyplot(fig_fb); plt.close()

        # ---- Point-by-point table -------------------------------------------
        st.markdown("#### Point-by-Point Comparison (Mean Values)")
        df_pts = pd.DataFrame({
            f"Time ({time_unit})":          t_common,
            f"Reference Mean % ({ref_bs})": rr_obs.round(2),
            f"Test Mean % ({test_bs})":     rt_obs.round(2),
            "|Diff| (%)":                   np.abs(rr_obs - rt_obs).round(2),
            "Used in f2 (ref ≤ 85%)":      ["Yes" if r <= 85 else "No"
                                              for r in rr_obs],
        })
        st.dataframe(df_pts, use_container_width=True, hide_index=True)

        # Save bootstrap results to session_state (for Excel report)
        st.session_state["bootstrap_results"] = {
            "f2_obs":   f2_obs,
            "f2_mean":  f2_mean,
            "f2_median":f2_med,
            "f2_sd":    f2_sd,
            "ci_lower": f2_lower,
            "ci_upper": f2_upper,
            "n_iter":   len(f2_boot),
            "ref":      ref_bs,
            "test":     test_bs,
            "verdict":  verdict_icon,
            "f2_dist":  f2_boot.tolist(),
            "method":   bs_method,
        }

        # ---- Reference citation ---------------------------------------------
        st.markdown(
            "<div class='info-banner' style='font-size:0.83rem;'>"
            "<strong>Reference:</strong> Shah VP, Tsong Y, Sathe P, Liu JP. "
            "<em>In vitro dissolution profile comparison — statistics and analysis "
            "of the similarity factor, f2.</em> Pharm Res. 1998;15(6):889-896."
            " &nbsp;|&nbsp; FDA Guidance for Industry: Dissolution Testing of "
            "Immediate Release Solid Oral Dosage Forms (1997)."
            "</div>",
            unsafe_allow_html=True
        )



# ===========================================================================
# PAGE: IVIVC
# ===========================================================================
