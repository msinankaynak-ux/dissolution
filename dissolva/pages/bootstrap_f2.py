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
from dissolva.models import (MODEL_DEFS, CATEGORIES, fit_model, compute_mdt,
    compute_de, r2s, r2adj, aic_fn, msc_fn, _nz)
from dissolva.state import (current_tier, require_tier, _safe_profile_names,
    _get_index, _rename_profile, _clear_all)
from dissolva.content import show_literature, show_all_references, analyze_profile_shape


def render():
    _PLOTLY_OK = _PLOTLY_AVAILABLE
    cfg = st.session_state.method_cfg
    time_unit = cfg["time_unit"]

    # ---- Bootstrap engine (uses only numpy — no plotly import needed here) --
    def run_bootstrap_f2(ref_raw, test_raw, iterations=5000, seed=42):
        """
        ref_raw / test_raw : 2-D np.array  (n_timepoints  x  n_vessels)
        Returns: (f2_array, f2_lower_90pct)
        """
        rng = np.random.default_rng(seed if seed > 0 else None)
        n_v_ref  = ref_raw.shape[1]
        n_v_test = test_raw.shape[1]
        f2_results = []
        for _ in range(iterations):
            ref_sample  = ref_raw[:,  rng.integers(0, n_v_ref,  size=n_v_ref)]
            test_sample = test_raw[:, rng.integers(0, n_v_test, size=n_v_test)]
            R_bar = np.mean(ref_sample,  axis=1)
            T_bar = np.mean(test_sample, axis=1)
            # FDA 85% rule: all ref<85 + first point exceeding 85
            _below_b = R_bar <= 85
            _above_b = R_bar > 85
            mask = _below_b.copy()
            if np.any(_above_b):
                mask[np.where(_above_b)[0][0]] = True
            if np.any(mask):
                R_f, T_f = R_bar[mask], T_bar[mask]
                mse = np.mean((R_f - T_f) ** 2)
                f2  = 50 * np.log10(100 / np.sqrt(1 + mse))
                f2_results.append(f2)
        f2_arr = np.array(f2_results)
        f2_low90 = float(np.percentile(f2_arr, 5))
        return f2_arr, f2_low90

    # ── Nonparametric bootstrap function ───────────────────────────────────
    def run_nonparametric_bootstrap_f2(ref_raw, test_raw, iterations=5000, seed=42):
        """Shah 1998 original — nonparametric bootstrap. No distribution assumption."""
        rng = np.random.default_rng(seed if seed > 0 else None)
        n_v_ref  = ref_raw.shape[1]
        n_v_test = test_raw.shape[1]
        results = []
        chunk = max(1, iterations // 100)
        for i in range(iterations):
            idx_ref  = rng.integers(0, n_v_ref,  size=n_v_ref)
            idx_test = rng.integers(0, n_v_test, size=n_v_test)
            ref_s  = ref_raw[:,  idx_ref]
            test_s = test_raw[:, idx_test]
            R_bar = np.mean(ref_s,  axis=1)
            T_bar = np.mean(test_s, axis=1)
            # FDA 85% rule: all ref<85 + first point exceeding 85
            _below_b = R_bar <= 85
            _above_b = R_bar > 85
            mask = _below_b.copy()
            if np.any(_above_b):
                mask[np.where(_above_b)[0][0]] = True
            if np.any(mask):
                mse = np.mean((R_bar[mask] - T_bar[mask]) ** 2)
                results.append(50 * np.log10(100 / np.sqrt(1 + mse)))
        return np.array(results)

    # ---- Page header --------------------------------------------------------
    st.markdown(
        "<h2 style='color:#002147;margin:0 0 4px;'>Bootstrap f2 Analysis</h2>"
        "<p style='color:#888;font-size:0.88rem;margin:0 0 12px;'>"
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

        # Observed f2 (mean profiles)
        rr_obs = np.array([raw_ref[np.where(t_ref_arr == ti)[0][0], :].mean()
                           for ti in t_common])
        rt_obs = np.array([raw_test[np.where(t_test_arr == ti)[0][0], :].mean()
                           for ti in t_common])
        mask_obs = rr_obs <= 85
        _above_obs = np.where(rr_obs > 85)[0]
        if len(_above_obs) > 0:
            mask_obs[_above_obs[0]] = True  # First point exceeding 85% included
        if not np.any(mask_obs):
            st.error("No valid time points where reference release ≤ 85%."); st.stop()

        f2_obs = float(50 * np.log10(
            100 / np.sqrt(1 + np.mean((rr_obs[mask_obs] - rt_obs[mask_obs]) ** 2))
        ))

        # Subset raw matrices to common time points
        ref_idx  = [np.where(t_ref_arr  == ti)[0][0] for ti in t_common]
        test_idx = [np.where(t_test_arr == ti)[0][0] for ti in t_common]
        raw_ref_common  = raw_ref[ref_idx,  :]
        raw_test_common = raw_test[test_idx, :]

        # Run bootstrap
        prog = st.progress(0, text="Running bootstrap simulation…")

        def _bootstrap_with_progress(ref_raw, test_raw, iterations, seed):
            rng = np.random.default_rng(seed if seed > 0 else None)
            n_v_ref  = ref_raw.shape[1]
            n_v_test = test_raw.shape[1]
            results = []
            chunk = max(1, iterations // 100)
            for i in range(iterations):
                ref_s  = ref_raw[:,  rng.integers(0, n_v_ref,  size=n_v_ref)]
                test_s = test_raw[:, rng.integers(0, n_v_test, size=n_v_test)]
                R_bar  = np.mean(ref_s,  axis=1)
                T_bar  = np.mean(test_s, axis=1)
                mask   = R_bar <= 85
                if np.any(mask):
                    mse = np.mean((R_bar[mask] - T_bar[mask]) ** 2)
                    results.append(50 * np.log10(100 / np.sqrt(1 + mse)))
                if (i + 1) % chunk == 0:
                    prog.progress((i + 1) / iterations,
                                  text=f"Iteration {i+1:,} / {iterations:,}…")
            return np.array(results)

        if "Nonparametric" in bs_method:
            prog2 = st.progress(0, text="Running nonparametric bootstrap…")
            def _nonparam_progress(ref_raw, test_raw, iterations, seed):
                rng2 = np.random.default_rng(seed if seed > 0 else None)
                n_v_r = ref_raw.shape[1]; n_v_t = test_raw.shape[1]
                res2 = []; chunk2 = max(1, iterations // 100)
                for ii in range(iterations):
                    ir = rng2.integers(0, n_v_r, size=n_v_r)
                    it = rng2.integers(0, n_v_t, size=n_v_t)
                    R2 = np.mean(ref_raw[:, ir], axis=1)
                    T2 = np.mean(test_raw[:, it], axis=1)
                    m2 = R2 <= 85
                    if np.any(m2):
                        mse2 = np.mean((R2[m2]-T2[m2])**2)
                        res2.append(50*np.log10(100/np.sqrt(1+mse2)))
                    if (ii+1) % chunk2 == 0:
                        prog2.progress((ii+1)/iterations, text=f"Iteration {ii+1:,}/{iterations:,}…")
                return np.array(res2)
            f2_boot = _nonparam_progress(raw_ref_common, raw_test_common, int(n_iter), int(seed_val))
            prog2.empty()
        else:
            f2_boot = _bootstrap_with_progress(
                raw_ref_common, raw_test_common, int(n_iter), int(seed_val)
            )
        prog.empty()

        f2_boot    = f2_boot[~np.isnan(f2_boot)]
        f2_lower   = float(np.percentile(f2_boot, lower_pctile))
        f2_upper   = float(np.percentile(f2_boot, 100 - lower_pctile))
        f2_mean    = float(np.mean(f2_boot))
        f2_med     = float(np.median(f2_boot))
        f2_sd      = float(np.std(f2_boot, ddof=1))
        is_similar = f2_lower >= 50

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
