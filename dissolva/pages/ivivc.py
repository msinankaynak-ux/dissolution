"""DissolvA page module: IVIVC Analysis. Extracted from app.py (Phase 3b modularization)."""
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
    dose_mg = cfg["dose_mg"]
    from scipy import stats as _scipy_stats
    from scipy.interpolate import interp1d as _interp1d

    # -- Title ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
        '<h2 style="margin:0;color:#002147;">IVIVC Analysis</h2>'
        '<span style="background:#FFBF00;color:#002147;font-size:0.62rem;font-weight:700;'
        'letter-spacing:1.5px;text-transform:uppercase;padding:3px 8px;border-radius:3px;">β BETA VERSION</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="background:#c0392b;border:1px solid #a93226;'
        'border-left:4px solid #922b21;border-radius:4px;padding:10px 14px;margin-bottom:12px;color:white;">'
        '⚠️ <strong>Beta Feature — FDA/EMA 5-Level IVIVC Engine:</strong> '
        'Results must be validated against approved IVIVC software (DDSolver, WinNonlin). '
        'Level A and Multiple Level C are marked as Professional features.'
        '</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.profiles:
        st.warning("⚠️ No dissolution profiles loaded. Please go to Data Input first.")
        st.stop()

    # ── Session state init ────────────────────────────────────────────────────
    if "ivivc_input_data" not in st.session_state:
        st.session_state.ivivc_input_data = {}
    if "selected_level" not in st.session_state:
        st.session_state.selected_level = "Level A"

    # ── IVIVC Level Selection ────────────────────────────────────────────────────
    IVIVC_LEVELS = {
        "Level A": ("🏆 Level A — Point-to-Point", "Wagner-Nelson deconvolution; highest regulatory value (FDA Preferred). [Professional]"),
        "Level B": ("📊 Level B — Statistical Moments", "In vitro MDT vs in vivo MRT; moment analysis."),
        "Level C": ("📈 Level C — Single Point", "Single PK parameter vs single dissolution time point."),
        "Multiple Level C": ("🔬 Multiple Level C — Multi-Point", "≥3 time points across dissolution profile; near Level A reliability. [Professional]"),
        "Level D": ("👁️ Level D — Qualitative/Visual", "Visual trend confirmation; rank ordering only."),
    }

    level_col, info_col = st.columns([1, 2])
    with level_col:
        selected_level = st.radio(
            "Select IVIVC Level",
            list(IVIVC_LEVELS.keys()),
            index=list(IVIVC_LEVELS.keys()).index(st.session_state.selected_level),
            key="_ivivc_level_radio",
        )
        st.session_state.selected_level = selected_level
    with info_col:
        label, desc = IVIVC_LEVELS[selected_level]
        professional = "[Professional]" in desc
        badge_color  = "#002147" if professional else "#27ae60"
        badge_text   = "🔒 Professional" if professional else "✅ All Plans"
        st.markdown(
            f'<div style="background:#f0ece0;border-left:4px solid {badge_color};'
            f'border-radius:4px;padding:12px 16px;margin-top:8px;">'
            f'<strong>{label}</strong><br>'
            f'<span style="font-size:0.88rem;color:#555;">{desc.replace(" [Professional]","")}</span><br><br>'
            f'<span style="background:{badge_color};color:white;font-size:0.7rem;'
            f'padding:2px 8px;border-radius:3px;">{badge_text}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Profile Selection ─────────────────────────────────────────────────────────
    _ivivc_names = list(st.session_state.profiles.keys())
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        iv_profile = st.selectbox(
            "In Vitro Dissolution Profile",
            _ivivc_names,
            index=_get_index(_ivivc_names, st.session_state.selected_ref_id, 0),
            help="Select the dissolution profile to be used as in vitro input."
        )
    d_iv = st.session_state.profiles[iv_profile]
    t_iv = np.array(d_iv["time"], dtype=float)
    r_iv = np.array(d_iv["release"], dtype=float)

    # ── IVIVC Helper Functions ───────────────────────────────────────────
    def _nca(t_arr, cp_arr):
        """Non-Compartmental Analysis: Cmax, Tmax, AUC, ke, MRT."""
        t_arr = np.array(t_arr, dtype=float)
        cp_arr = np.array(cp_arr, dtype=float)
        cmax_idx = np.argmax(cp_arr)
        Cmax = float(cp_arr[cmax_idx])
        Tmax = float(t_arr[cmax_idx])
        AUC  = float(trapezoid(cp_arr, t_arr))
        # Terminal ke via log-linear regression on last ≥3 declining points
        decline = cp_arr[cmax_idx:]
        t_dec   = t_arr[cmax_idx:]
        ke = np.nan
        if len(decline) >= 3:
            mask = decline > 0
            if mask.sum() >= 3:
                slope, _, r, _, _ = _scipy_stats.linregress(t_dec[mask], np.log(decline[mask]))
                ke = float(-slope) if slope < 0 else np.nan
        MRT = float(trapezoid(t_arr * cp_arr, t_arr) / AUC) if AUC > 0 else np.nan
        return {"Cmax": Cmax, "Tmax": Tmax, "AUC": AUC, "ke": ke, "MRT": MRT}

    def _wagner_nelson(t_iv, r_iv, ke):
        """Wagner-Nelson deconvolution → Fraction Absorbed."""
        Ct = r_iv / 100.0 * dose_mg
        AUC_t = np.array([trapezoid(Ct[:i+1], t_iv[:i+1]) for i in range(len(t_iv))])
        AUC_inf = trapezoid(Ct, t_iv) + Ct[-1] / ke
        Fa = np.clip((Ct + ke * AUC_t) / (ke * AUC_inf) * 100, 0, 100)
        return Fa

    def _regression_plotly(x, y, x_label, y_label, title, color=OXFORD):
        """Plotly regression + 95% CI chart."""
        import plotly.graph_objects as _go
        slope, intercept, r, p, se = _scipy_stats.linregress(x, y)
        r2 = r**2
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept
        n = len(x)
        t_crit = _scipy_stats.t.ppf(0.975, df=n-2)
        x_mean = x.mean()
        se_fit = se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
        ci_upper = y_line + t_crit * se_fit
        ci_lower = y_line - t_crit * se_fit

        fig = _go.Figure()
        fig.add_trace(_go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([ci_upper, ci_lower[::-1]]),
            fill='toself', fillcolor='rgba(0,33,71,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name='95% CI', hoverinfo='skip'
        ))
        fig.add_trace(_go.Scatter(
            x=x_line, y=y_line,
            mode='lines', line=dict(color=AMBER, width=2.5, dash='dash'),
            name=f'Regression (R²={r2:.4f})'
        ))
        fig.add_trace(_go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(color=color, size=10, line=dict(color='white', width=1.5)),
            name='Data'
        ))
        eq_text = f"y = {slope:.4f}x + {intercept:.4f}<br>R² = {r2:.4f}  |  p = {p:.4f}"
        fig.add_annotation(
            xref='paper', yref='paper', x=0.05, y=0.95,
            text=eq_text, showarrow=False,
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor=OXFORD, borderwidth=1,
            font=dict(size=11, color=OXFORD)
        )
        fig.update_layout(
            title=dict(text=title, font=dict(color=OXFORD, size=14)),
            xaxis_title=x_label, yaxis_title=y_label,
            plot_bgcolor='#F8F4EC', paper_bgcolor='#FDFAF5',
            font=dict(family='EB Garamond, Georgia, serif', color=OXFORD),
            height=420, margin=dict(t=60),
        )
        return fig, r2, slope, intercept, p

    def _pe_badge(pe):
        """Prediction Error assessment."""
        if pe < 10:
            return "✅ Model Validated (FDA Grade)", "#c6efce", "#1a5c2e"
        elif pe < 20:
            return "⚠️ Marginal (Supportive data may be required)", "#fff3cd", "#856404"
        else:
            return "❌ Model Invalid (Poor correlation)", "#ffc7ce", "#7b1a1a"

    # ══════════════════════════════════════════════════════════════════════════
    # LEVEL A — Point-to-Point (Wagner-Nelson)
    # ══════════════════════════════════════════════════════════════════════════
    if selected_level == "Level A":
        st.markdown("### Level A — Point-to-Point IVIVC (Wagner-Nelson Deconvolution)")
        st.markdown(
            '<div class="info-banner">Level A is the highest level of correlation recognized by the FDA. '
            'Wagner-Nelson deconvolution converts in vitro dissolution to in vivo fraction absorbed (Fa) '
            'at each time point. Requires plasma concentration-time data for the same formulation.</div>',
            unsafe_allow_html=True
        )
        with sel_col2:
            ke_input = st.number_input(
                "Elimination rate constant kₑ (h⁻¹ or min⁻¹)",
                value=st.session_state.ivivc_input_data.get("ke", 0.1),
                min_value=1e-5, format="%.5f",
                help="Obtain from single IV dose PK study or NCA of oral PK data."
            )
            st.session_state.ivivc_input_data["ke"] = ke_input

        st.markdown("#### 📥 In Vivo Plasma Concentration Data (Optional — for %PE validation)")
        st.markdown(
            '<div class="step-box">Enter observed Cp–t data for %PE calculation. '
            'If omitted, only the Wagner-Nelson deconvolution will be shown.</div>',
            unsafe_allow_html=True
        )
        pkcol1, pkcol2 = st.columns(2)
        with pkcol1:
            pk_t_str  = st.text_area("Time points (comma-separated)",
                value=st.session_state.ivivc_input_data.get("pk_t",""),
                height=80, key="lva_pk_t",
                placeholder="e.g. 0,0.5,1,2,4,6,8,12,24")
        with pkcol2:
            pk_cp_str = st.text_area("Cp values (comma-separated)",
                value=st.session_state.ivivc_input_data.get("pk_cp",""),
                height=80, key="lva_pk_cp",
                placeholder="e.g. 0,120,210,185,140,95,60,30,8")

        # Deconvolution
        Fa = _wagner_nelson(t_iv, r_iv, ke_input)

        # Metrics
        m1, m2, m3 = st.columns(3)
        r_corr = float(np.corrcoef(r_iv, Fa)[0,1])
        m1.metric("Max Fa (%)", f"{Fa[-1]:.1f}")
        m2.metric("IVIVC r", f"{r_corr:.4f}")
        m3.metric("R²", f"{r_corr**2:.4f}")

        # Main chart - plotly
        if _PLOTLY_OK:
            import plotly.graph_objects as _go
            fig_lva = _go.Figure()
            fig_lva.add_trace(_go.Scatter(x=t_iv, y=r_iv, mode='lines+markers',
                name='In Vitro Fd (%)', line=dict(color=OXFORD, width=2.5),
                marker=dict(size=8)))
            fig_lva.add_trace(_go.Scatter(x=t_iv, y=Fa, mode='lines+markers',
                name='In Vivo Fa (%) [Wagner-Nelson]',
                line=dict(color='#c0392b', width=2.5, dash='dash'),
                marker=dict(symbol='square', size=8)))
            fig_lva.update_layout(
                title="Level A: In Vitro Fd vs In Vivo Fa (Wagner-Nelson)",
                xaxis_title=f"Time ({time_unit})",
                yaxis_title="Cumulative Drug Released / Absorbed (%)",
                xaxis=dict(rangemode='tozero'),
                yaxis=dict(rangemode='tozero', range=[0,112]),
                plot_bgcolor='#F8F4EC', paper_bgcolor='#FDFAF5',
                font=dict(family='EB Garamond, Georgia, serif'),
                height=420,
            )
            st.plotly_chart(fig_lva, use_container_width=True)
        else:
            fig, axes = plt.subplots(1,2,figsize=(11,4.5))
            for ax in axes: style_ax(fig, ax)
            axes[0].plot(t_iv, r_iv, "o-", color=OXFORD, lw=2, ms=5, label="In Vitro Fd")
            axes[0].plot(t_iv, Fa, "s--", color="#c0392b", lw=2, ms=5, label="Fa (Wagner-Nelson)")
            axes[0].set_xlabel(f"Time ({time_unit})")
            axes[0].set_ylabel("Cumulative (%)")
            axes[0].set_xlim(left=0); axes[0].set_ylim(0,112)
            axes[0].legend(fontsize=8); axes[0].set_title("Fd vs Fa")
            axes[1].scatter(r_iv, Fa, color=OXFORD, s=60, edgecolors=AMBER, lw=1)
            m,b = np.polyfit(r_iv, Fa, 1)
            xl = np.linspace(r_iv.min(), r_iv.max(), 100)
            axes[1].plot(xl, m*xl+b, "--", color=AMBER, lw=2)
            axes[1].set_xlabel("In Vitro Fd (%)"); axes[1].set_ylabel("Fa (%)")
            axes[1].set_title(f"r = {r_corr:.4f}")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Regression Fd vs Fa
        if _PLOTLY_OK:
            fig_reg, r2_reg, slope_reg, intc_reg, p_reg = _regression_plotly(
                r_iv, Fa, "In Vitro Fd (%)", "In Vivo Fa (%)",
                "Level A Regression: Fd vs Fa"
            )
            st.plotly_chart(fig_reg, use_container_width=True)

        # %PE calculation (if PK data entered)
        if pk_t_str.strip() and pk_cp_str.strip():
            try:
                pk_t  = np.array([float(x) for x in pk_t_str.split(",")])
                pk_cp = np.array([float(x) for x in pk_cp_str.split(",")])
                st.session_state.ivivc_input_data.update({"pk_t": pk_t_str, "pk_cp": pk_cp_str})
                nca_res = _nca(pk_t, pk_cp)

                # Predicted AUC from model: integral of Fa curve scaled
                pred_auc = float(trapezoid(Fa/100.0 * dose_mg / ke_input, t_iv))
                obs_auc  = nca_res["AUC"]
                pe_auc   = abs(obs_auc - pred_auc) / obs_auc * 100 if obs_auc > 0 else np.nan

                st.markdown("#### 📋 NCA Results & Prediction Error (%PE)")
                nc1, nc2, nc3, nc4, nc5 = st.columns(5)
                nc1.metric("Cmax",   f"{nca_res['Cmax']:.2f}")
                nc2.metric("Tmax",   f"{nca_res['Tmax']:.2f}")
                nc3.metric("AUC",    f"{nca_res['AUC']:.2f}")
                nc4.metric("kₑ",     f"{nca_res['ke']:.4f}" if not np.isnan(nca_res['ke']) else "N/A")
                nc5.metric("MRT",    f"{nca_res['MRT']:.2f}" if not np.isnan(nca_res['MRT']) else "N/A")

                if not np.isnan(pe_auc):
                    verdict, bg, fg = _pe_badge(pe_auc)
                    st.markdown(
                        f'<div style="background:{bg};border-radius:6px;padding:12px 18px;'
                        f'border-left:5px solid {fg};margin:12px 0;">'
                        f'<strong style="color:{fg};font-size:1rem;">%PE (AUC) = {pe_auc:.2f}%</strong><br>'
                        f'<span style="color:{fg};">{verdict}</span>'
                        f'</div>', unsafe_allow_html=True
                    )
            except Exception as _e:
                st.warning(f"PK data parse error: {_e}")

        st.dataframe(pd.DataFrame({
            f"Time ({time_unit})": t_iv,
            "In Vitro Fd (%)": r_iv.round(2),
            "In Vivo Fa (%) [W-N]": Fa.round(2),
            "ΔFd - Fa": (r_iv - Fa).round(2),
        }), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # LEVEL B — Statistical Moments
    # ══════════════════════════════════════════════════════════════════════════
    elif selected_level == "Level B":
        st.markdown("### Level B — Statistical Moment Analysis (MDT vs MRT)")
        st.markdown(
            '<div class="info-banner">Level B correlates the in vitro Mean Dissolution Time (MDT) '
            'with the in vivo Mean Residence Time (MRT). Uses all of the in vitro data but not a '
            'point-to-point correlation — lower regulatory value than Level A.</div>',
            unsafe_allow_html=True
        )

        # Compute in vitro MDT
        mdt_iv = compute_mdt(t_iv, r_iv)
        st.metric(f"In Vitro MDT ({time_unit})", f"{mdt_iv:.3f}" if not np.isnan(mdt_iv) else "N/A")

        st.markdown("#### 📥 In Vivo MRT Data Entry (Multiple Formulations)")
        st.markdown(
            '<div class="step-box">Enter in vivo MRT values for each formulation. '
            'You can enter data from multiple formulations to build the MDT–MRT regression.</div>',
            unsafe_allow_html=True
        )

        n_forms_b = st.number_input("Number of formulations", min_value=2, max_value=10,
                                     value=st.session_state.ivivc_input_data.get("lvb_n", 3),
                                     key="lvb_n_input")
        st.session_state.ivivc_input_data["lvb_n"] = n_forms_b

        b_data = []
        bcols = st.columns(min(int(n_forms_b), 4))
        for i in range(int(n_forms_b)):
            with bcols[i % 4]:
                mdt_i = st.number_input(f"MDT F{i+1} ({time_unit})",
                    value=float(st.session_state.ivivc_input_data.get(f"lvb_mdt_{i}", mdt_iv if i==0 else 0.0)),
                    min_value=0.0, key=f"lvb_mdt_{i}")
                mrt_i = st.number_input(f"MRT F{i+1} ({time_unit})",
                    value=float(st.session_state.ivivc_input_data.get(f"lvb_mrt_{i}", 0.0)),
                    min_value=0.0, key=f"lvb_mrt_{i}")
                st.session_state.ivivc_input_data.update({f"lvb_mdt_{i}": mdt_i, f"lvb_mrt_{i}": mrt_i})
                b_data.append((mdt_i, mrt_i))

        if all(mrt > 0 for _, mrt in b_data) and len(b_data) >= 2:
            mdt_arr = np.array([d[0] for d in b_data])
            mrt_arr = np.array([d[1] for d in b_data])

            if _PLOTLY_OK:
                fig_b, r2_b, sl_b, ic_b, p_b = _regression_plotly(
                    mdt_arr, mrt_arr,
                    f"In Vitro MDT ({time_unit})", f"In Vivo MRT ({time_unit})",
                    "Level B: MDT vs MRT Regression"
                )
                st.plotly_chart(fig_b, use_container_width=True)
            else:
                r2_b = np.corrcoef(mdt_arr, mrt_arr)[0,1]**2
                sl_b, ic_b = np.polyfit(mdt_arr, mrt_arr, 1)

            m1b, m2b, m3b = st.columns(3)
            m1b.metric("R²", f"{r2_b:.4f}")
            m2b.metric("Slope", f"{sl_b:.4f}")
            m3b.metric("Intercept", f"{ic_b:.4f}")

            eq_str = f"MRT = {sl_b:.4f} × MDT + {ic_b:.4f}"
            st.markdown(
                f'<div class="eq-box">{eq_str}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("Enter MRT values for at least 2 formulations to generate the regression.")

    # ══════════════════════════════════════════════════════════════════════════
    # LEVEL C — Single Point
    # ══════════════════════════════════════════════════════════════════════════
    elif selected_level == "Level C":
        st.markdown("### Level C — Single Point Correlation")
        st.markdown(
            '<div class="info-banner">Level C establishes a single-point relationship between a '
            'PK parameter (Cmax, AUC, Tmax) and a dissolution parameter (t50%, t80%, MDT). '
            'Lower regulatory value than Level A but useful for formulation screening.</div>',
            unsafe_allow_html=True
        )

        with sel_col2:
            pk_param_c = st.selectbox("PK Parameter",
                ["Cmax", "AUC(0-inf)", "Tmax", "MRT"],
                index=st.session_state.ivivc_input_data.get("lvc_pk_idx", 0),
                key="lvc_pk_param")
            st.session_state.ivivc_input_data["lvc_pk_idx"] = ["Cmax","AUC(0-inf)","Tmax","MRT"].index(pk_param_c)

            diss_param_c = st.selectbox("Dissolution Parameter",
                ["MDT", "t50% (TD50)", "t80% (TD80)", "DE (%)"],
                index=st.session_state.ivivc_input_data.get("lvc_diss_idx", 0),
                key="lvc_diss_param")
            st.session_state.ivivc_input_data["lvc_diss_idx"] = ["MDT","t50% (TD50)","t80% (TD80)","DE (%)"].index(diss_param_c)

        # Auto-compute dissolution parameter for current profile
        mdt_auto = compute_mdt(t_iv, r_iv)
        de_auto  = compute_de(t_iv, r_iv)
        f_interp = _interp1d(r_iv, t_iv, bounds_error=False, fill_value=np.nan) if len(np.unique(r_iv))>1 else None
        t50_auto = float(f_interp(50)) if f_interp and 50 <= r_iv.max() else np.nan
        t80_auto = float(f_interp(80)) if f_interp and 80 <= r_iv.max() else np.nan

        auto_map = {"MDT": mdt_auto, "t50% (TD50)": t50_auto,
                    "t80% (TD80)": t80_auto, "DE (%)": de_auto}
        auto_val = auto_map.get(diss_param_c, np.nan)
        st.info(f"Auto-computed **{diss_param_c}** for *{iv_profile}*: "
                f"**{auto_val:.3f} {time_unit}**" if not np.isnan(auto_val) else
                f"Could not compute {diss_param_c} — dissolution may not reach required level.")

        st.markdown("#### 📥 Multi-Formulation Data Entry")
        n_forms_c = st.number_input("Number of formulations", min_value=2, max_value=12,
                                     value=st.session_state.ivivc_input_data.get("lvc_n", 3),
                                     key="lvc_n_input")
        st.session_state.ivivc_input_data["lvc_n"] = n_forms_c

        c_data = []
        ccols_h = st.columns(min(int(n_forms_c), 4))
        for i in range(int(n_forms_c)):
            with ccols_h[i % 4]:
                diss_i = st.number_input(f"{diss_param_c} F{i+1}",
                    value=float(st.session_state.ivivc_input_data.get(f"lvc_diss_{i}", auto_val if i==0 and not np.isnan(auto_val) else 0.0)),
                    min_value=0.0, key=f"lvc_diss_{i}")
                pk_i = st.number_input(f"{pk_param_c} F{i+1}",
                    value=float(st.session_state.ivivc_input_data.get(f"lvc_pk_{i}", 0.0)),
                    min_value=0.0, key=f"lvc_pk_{i}")
                st.session_state.ivivc_input_data.update({f"lvc_diss_{i}": diss_i, f"lvc_pk_{i}": pk_i})
                c_data.append((diss_i, pk_i))

        if all(v > 0 for _, v in c_data) and len(c_data) >= 2:
            diss_arr_c = np.array([d[0] for d in c_data])
            pk_arr_c   = np.array([d[1] for d in c_data])

            if _PLOTLY_OK:
                fig_c, r2_c, sl_c, ic_c, p_c = _regression_plotly(
                    diss_arr_c, pk_arr_c,
                    diss_param_c, pk_param_c,
                    f"Level C: {diss_param_c} vs {pk_param_c}"
                )
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                r2_c = np.corrcoef(diss_arr_c, pk_arr_c)[0,1]**2
                sl_c, ic_c = np.polyfit(diss_arr_c, pk_arr_c, 1)

            m1c, m2c, m3c = st.columns(3)
            m1c.metric("R²", f"{r2_c:.4f}")
            m2c.metric("Slope", f"{sl_c:.4f}")
            m3c.metric("Intercept", f"{ic_c:.4f}")
            st.markdown(f'<div class="eq-box">{pk_param_c} = {sl_c:.4f} × {diss_param_c} + {ic_c:.4f}</div>',
                        unsafe_allow_html=True)
        else:
            st.info(f"Enter {pk_param_c} and {diss_param_c} values for at least 2 formulations.")

    # ══════════════════════════════════════════════════════════════════════════
    # MULTIPLE LEVEL C — Multi-Point
    # ══════════════════════════════════════════════════════════════════════════
    elif selected_level == "Multiple Level C":
        st.markdown("### Multiple Level C — Multi-Point IVIVC")
        st.markdown(
            '<div class="info-banner">Multiple Level C uses dissolution values at ≥3 time points '
            '(early ~20%, mid ~50%, late ~80%) each correlated with the same PK parameter. '
            'If all R² > 0.90, it approaches Level A reliability (FDA Guidance, 1997).</div>',
            unsafe_allow_html=True
        )

        # Auto-compute dissolution timepoints
        f_diss = _interp1d(r_iv, t_iv, bounds_error=False, fill_value=np.nan) if len(np.unique(r_iv))>1 else None
        t20 = float(f_diss(20)) if f_diss and 20 <= r_iv.max() else np.nan
        t50 = float(f_diss(50)) if f_diss and 50 <= r_iv.max() else np.nan
        t80 = float(f_diss(80)) if f_diss and 80 <= r_iv.max() else np.nan

        st.markdown("#### Auto-Detected Dissolution Time Points")
        adc1, adc2, adc3 = st.columns(3)
        adc1.metric("t₂₀% (Early Phase)", f"{t20:.2f} {time_unit}" if not np.isnan(t20) else "N/A")
        adc2.metric("t₅₀% (Mid Phase)",   f"{t50:.2f} {time_unit}" if not np.isnan(t50) else "N/A")
        adc3.metric("t₈₀% (Late Phase)",  f"{t80:.2f} {time_unit}" if not np.isnan(t80) else "N/A")

        with sel_col2:
            pk_param_mc = st.selectbox("PK Parameter for all phases",
                ["Cmax", "AUC(0-inf)", "Tmax"],
                index=st.session_state.ivivc_input_data.get("lvmc_pk_idx", 0),
                key="lvmc_pk_param")
            st.session_state.ivivc_input_data["lvmc_pk_idx"] = ["Cmax","AUC(0-inf)","Tmax"].index(pk_param_mc)

        st.markdown(f"#### 📥 {pk_param_mc} Data for Each Phase (Multiple Formulations)")
        n_forms_mc = st.number_input("Number of formulations", min_value=3, max_value=12,
                                      value=st.session_state.ivivc_input_data.get("lvmc_n", 4),
                                      key="lvmc_n_input")
        st.session_state.ivivc_input_data["lvmc_n"] = n_forms_mc

        phase_names = ["Early (t20%)", "Mid (t50%)", "Late (t80%)"]
        phase_times = [t20, t50, t80]
        mc_phase_data = {ph: [] for ph in phase_names}

        mc_header = st.columns([2,2,2,2])
        mc_header[0].markdown("**Formulation**")
        mc_header[1].markdown(f"**{phase_names[0]}**")
        mc_header[2].markdown(f"**{phase_names[1]}**")
        mc_header[3].markdown(f"**{phase_names[2]}**")

        for i in range(int(n_forms_mc)):
            row_cols = st.columns([2,2,2,2])
            with row_cols[0]:
                st.markdown(f"*F{i+1}*")
            for j, ph in enumerate(phase_names):
                with row_cols[j+1]:
                    val = st.number_input(
                        f"{ph} F{i+1}",
                        value=float(st.session_state.ivivc_input_data.get(f"lvmc_{j}_{i}", 0.0)),
                        min_value=0.0, key=f"lvmc_{j}_{i}",
                        label_visibility="collapsed"
                    )
                    st.session_state.ivivc_input_data[f"lvmc_{j}_{i}"] = val
                    mc_phase_data[ph].append(val)

        # Dissolution values at each phase (t20, t50, t80)
        mc_diss_vals = [
            np.array([20.0] * int(n_forms_mc)),
            np.array([50.0] * int(n_forms_mc)),
            np.array([80.0] * int(n_forms_mc)),
        ]

        all_r2_pass = True
        r2_results = {}
        if all(any(v > 0 for v in mc_phase_data[ph]) for ph in phase_names):
            phase_colors = [OXFORD, "#27ae60", "#c0392b"]
            mc_tabs = st.tabs(phase_names)
            for ti, (ph, diss_x, ph_color) in enumerate(zip(phase_names, mc_diss_vals, phase_colors)):
                with mc_tabs[ti]:
                    pk_vals = np.array(mc_phase_data[ph])
                    if all(pk_vals > 0):
                        if _PLOTLY_OK:
                            fig_mc, r2_mc, sl_mc, ic_mc, p_mc = _regression_plotly(
                                diss_x, pk_vals,
                                f"Dissolution at {ph} (%)",
                                pk_param_mc,
                                f"{ph}: Dissolution vs {pk_param_mc}",
                                color=ph_color
                            )
                            st.plotly_chart(fig_mc, use_container_width=True)
                        else:
                            r2_mc = np.corrcoef(diss_x, pk_vals)[0,1]**2
                        r2_results[ph] = r2_mc
                        color_r2 = "#c6efce" if r2_mc >= 0.9 else "#ffc7ce"
                        st.markdown(
                            f'<div style="background:{color_r2};border-radius:4px;'
                            f'padding:8px 14px;font-weight:600;">'
                            f'R² = {r2_mc:.4f} {"✅ R² ≥ 0.90" if r2_mc>=0.9 else "❌ R² < 0.90"}</div>',
                            unsafe_allow_html=True
                        )
                        if r2_mc < 0.9:
                            all_r2_pass = False
                    else:
                        st.info(f"Enter {pk_param_mc} values for all formulations in {ph} phase.")
                        all_r2_pass = False

            if r2_results:
                st.markdown("#### 🎯 Multiple Level C — Overall Assessment")
                if all_r2_pass:
                    st.success(
                        "✅ All three phases achieve R² ≥ 0.90 — This correlation approaches "
                        "**Level A reliability** as per FDA Guidance for Industry (1997). "
                        "Consider proceeding to a formal Level A validation study."
                    )
                else:
                    st.warning(
                        "⚠️ One or more phases show R² < 0.90. The correlation does not meet "
                        "the threshold for Level A-equivalent confidence. Review formulation "
                        "design or extend the dissolution time points."
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # LEVEL D — Qualitative / Visual
    # ══════════════════════════════════════════════════════════════════════════
    elif selected_level == "Level D":
        st.markdown("### Level D — Qualitative / Visual Trend Analysis")
        st.markdown(
            '<div class="info-banner">Level D is a qualitative approach providing visual trend '
            'confirmation. In vitro and in vivo profiles are overlaid on the same scale for '
            'rank-ordering assessment. No formal regression is required.</div>',
            unsafe_allow_html=True
        )

        with sel_col2:
            ke_d = st.number_input(
                "Elimination rate constant kₑ (for Fa estimate)",
                value=st.session_state.ivivc_input_data.get("lvd_ke", 0.1),
                min_value=1e-5, format="%.5f", key="lvd_ke_input"
            )
            st.session_state.ivivc_input_data["lvd_ke"] = ke_d

        Fa_d = _wagner_nelson(t_iv, r_iv, ke_d)

        if _PLOTLY_OK:
            import plotly.graph_objects as _go
            import plotly.subplots as _ps
            fig_d = _ps.make_subplots(
                rows=1, cols=2,
                subplot_titles=("In Vitro Dissolution Profile", "Estimated In Vivo Absorption (Fa)"),
                shared_yaxes=True
            )
            fig_d.add_trace(_go.Scatter(
                x=t_iv, y=r_iv, mode='lines+markers',
                name='In Vitro Fd (%)',
                line=dict(color=OXFORD, width=2.5),
                marker=dict(size=8)
            ), row=1, col=1)
            fig_d.add_trace(_go.Scatter(
                x=t_iv, y=Fa_d, mode='lines+markers',
                name='In Vivo Fa (%) [est.]',
                line=dict(color='#c0392b', width=2.5, dash='dash'),
                marker=dict(symbol='square', size=8)
            ), row=1, col=2)
            fig_d.update_yaxes(title_text="Cumulative (%)", range=[0,112], row=1, col=1)
            fig_d.update_xaxes(title_text=f"Time ({time_unit})", rangemode='tozero', row=1, col=1)
            fig_d.update_xaxes(title_text=f"Time ({time_unit})", rangemode='tozero', row=1, col=2)
            fig_d.update_layout(
                title="Level D: Side-by-Side In Vitro vs In Vivo Trend",
                plot_bgcolor='#F8F4EC', paper_bgcolor='#FDFAF5',
                font=dict(family='EB Garamond, Georgia, serif', color=OXFORD),
                height=430,
            )
            st.plotly_chart(fig_d, use_container_width=True)

        # Visual rank ordering table
        r_corr_d = float(np.corrcoef(r_iv, Fa_d)[0,1])
        st.metric("Visual Trend r", f"{r_corr_d:.4f}")
        st.markdown(
            '<div class="info-banner">'
            '<strong>Interpretation:</strong> Level D does not provide a predictive model. '
            'Use to visually confirm that formulations with faster dissolution also show '
            'faster in vivo absorption (rank ordering). A high visual concordance supports '
            'the rationale for proceeding to a formal Level A or B study.</div>',
            unsafe_allow_html=True
        )

        st.dataframe(pd.DataFrame({
            f"Time ({time_unit})": t_iv,
            "In Vitro Fd (%)": r_iv.round(2),
            "Estimated Fa (%) [Level D]": Fa_d.round(2),
        }), use_container_width=True)

    # ── Common Literature ───────────────────────────────────────────────────────
    show_literature("IVIVC Analysis")

# ===========================================================================
# PAGE: EXCEL REPORT
# ===========================================================================

# ===========================================================================
# PAGE: EXCEL REPORT
# ===========================================================================
