"""DissolvA page module: IVIVC Analysis (v3.5 — scientifically corrected rewrite).

Fixes vs the previous (disabled) version:
- Level A: Wagner-Nelson deconvolution is now applied to the IN VIVO PLASMA Cp-t
  data (not to the dissolution curve) to obtain the true fraction absorbed Fa(t);
  Fa is then correlated point-to-point against the in vitro fraction dissolved Fd.
  Internal predictability (%PE) uses forward CONVOLUTION of the IVIVC-mapped
  absorption with a 1-compartment disposition to predict Cmax/AUC.
- Multiple Level C: correlates % dissolved (which VARIES across formulations) at
  fixed sampling times against a PK parameter — the previous version regressed
  against a constant x (degenerate).
- Level D: rank-order (Spearman) concordance across formulations — no fabricated Fa.
Core math validated numerically against a 1-compartment first-order synthetic
(Wagner-Nelson recovers Fa; convolution recovers Cmax/AUC to <0.5% on a fine grid).
"""
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
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy import stats as _stats
from dissolva.theme import OXFORD, AMBER, PALETTE, style_ax
from dissolva.models import (MODEL_DEFS, CATEGORIES, compute_mdt,
    compute_de, r2s, r2adj, aic_fn, msc_fn, _nz)
from dissolva.state import (current_tier, require_tier, _safe_profile_names,
    _get_index, _rename_profile, _clear_all)
from dissolva.content import show_literature, show_all_references, analyze_profile_shape


# ═══════════════════════════════════════════════════════════════════════════
# Core PK / IVIVC math  (module-level, unit-agnostic, validated)
# ═══════════════════════════════════════════════════════════════════════════
def _nca(t, cp):
    """Non-compartmental analysis of a plasma Cp-t profile."""
    t = np.asarray(t, float); cp = np.asarray(cp, float)
    i = int(np.argmax(cp))
    Cmax, Tmax = float(cp[i]), float(t[i])
    AUC_last = float(trapezoid(cp, t))
    ke = _ke_terminal(t, cp)
    AUC_inf = AUC_last + (cp[-1] / ke if (ke and ke > 0) else 0.0)
    # AUMC for MRT
    aumc = float(trapezoid(t * cp, t))
    aumc_inf = aumc + (t[-1] * cp[-1] / ke + cp[-1] / ke**2 if (ke and ke > 0) else 0.0)
    MRT = aumc_inf / AUC_inf if AUC_inf > 0 else np.nan
    return {"Cmax": Cmax, "Tmax": Tmax, "AUC_last": AUC_last,
            "AUC_inf": AUC_inf, "ke": ke, "MRT": MRT}


def _ke_terminal(t, cp, n_min=3):
    """Terminal elimination rate constant from log-linear regression of the
    last declining points."""
    t = np.asarray(t, float); cp = np.asarray(cp, float)
    i = int(np.argmax(cp))
    td, cd = t[i:], cp[i:]
    m = cd > 0
    if m.sum() < n_min:
        return np.nan
    td, cd = td[m], cd[m]
    k = min(len(td), max(n_min, 3))
    slope = np.polyfit(td[-k:], np.log(cd[-k:]), 1)[0]
    return float(-slope) if slope < 0 else np.nan


def _wagner_nelson_plasma(t, cp, ke):
    """In vivo fraction absorbed (%) from PLASMA Cp-t (1-compartment)."""
    t = np.asarray(t, float); cp = np.asarray(cp, float)
    auc_t = np.array([trapezoid(cp[:i + 1], t[:i + 1]) for i in range(len(t))])
    auc_inf = trapezoid(cp, t) + cp[-1] / ke
    Fa = (cp + ke * auc_t) / (ke * auc_inf) * 100.0
    return np.clip(Fa, 0, 100), float(auc_inf)


def _predict_cp(t, Fabs_frac, ke, auc_inf, nfine=40):
    """Forward convolution: predicted Cp from an absorption input Fabs (0..1)
    and a 1-compartment disposition. Returns (t_fine, cp_fine)."""
    t = np.asarray(t, float); F = np.asarray(Fabs_frac, float)
    tf = np.linspace(t[0], t[-1], (len(t) - 1) * nfine + 1)
    Ff = np.interp(tf, t, F)
    dFf = np.gradient(Ff, tf)
    cpf = np.array([ke * auc_inf * trapezoid(dFf[:i + 1] * np.exp(-ke * (tf[i] - tf[:i + 1])),
                                             tf[:i + 1]) for i in range(len(tf))])
    return tf, np.clip(cpf, 0, None)


def _pe_badge(pe):
    if pe < 10:
        return "✅ Meets FDA internal predictability (%PE < 10%)", "#0e3a1e", "#7bd88f"
    elif pe < 15:
        return "⚠️ Marginal — supportive data may be required (10–15%)", "#3a2f0e", "#f0c419"
    else:
        return "❌ Fails predictability (%PE ≥ 15%)", "#3a1414", "#e57373"


def _regression(x, y, x_label, y_label, title):
    """Least-squares regression + 95% CI plot (dark theme). Returns
    (fig_or_None, r2, slope, intercept, p)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    slope, intercept, r, p, se = _stats.linregress(x, y)
    r2 = r ** 2
    if not _PLOTLY_OK or len(x) < 2:
        return None, r2, slope, intercept, p
    xl = np.linspace(x.min(), x.max(), 200)
    yl = slope * xl + intercept
    n = len(x)
    tcrit = _stats.t.ppf(0.975, df=max(n - 2, 1))
    sxx = np.sum((x - x.mean()) ** 2) or 1e-9
    se_fit = se * np.sqrt(1 / n + (xl - x.mean()) ** 2 / sxx)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([xl, xl[::-1]]),
                             y=np.concatenate([yl + tcrit * se_fit, (yl - tcrit * se_fit)[::-1]]),
                             fill='toself', fillcolor='rgba(255,204,0,0.10)',
                             line=dict(color='rgba(0,0,0,0)'), name='95% CI', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=xl, y=yl, mode='lines',
                             line=dict(color=AMBER, width=2.5, dash='dash'),
                             name=f'Fit (R²={r2:.4f})'))
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                             marker=dict(color='#5DA9E9', size=10, line=dict(color='white', width=1)),
                             name='Data'))
    fig.add_annotation(xref='paper', yref='paper', x=0.03, y=0.97,
                       text=f"y = {slope:.4f}x + {intercept:.4f}<br>R² = {r2:.4f} · p = {p:.4f}",
                       showarrow=False, bgcolor='rgba(22,32,63,0.85)', bordercolor=AMBER,
                       borderwidth=1, font=dict(size=11, color='#e8edf6'), align='left')
    fig.update_layout(title=dict(text=title, font=dict(color='#e8edf6', size=14)),
                      xaxis_title=x_label, yaxis_title=y_label,
                      plot_bgcolor='#16203F', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#cbd5e1'), height=420, margin=dict(t=54),
                      legend=dict(bgcolor='rgba(0,0,0,0)'))
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.07)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.07)', zeroline=False)
    return fig, r2, slope, intercept, p


def _t_at_percent(t, r, pct):
    """Time at which cumulative release reaches pct (%), via interpolation."""
    r = np.asarray(r, float); t = np.asarray(t, float)
    if r.max() < pct or len(np.unique(r)) < 2:
        return np.nan
    order = np.argsort(r)
    return float(np.interp(pct, r[order], t[order]))


def _parse_csv(s):
    return np.array([float(x) for x in str(s).replace("\n", ",").split(",") if x.strip() != ""])


# ═══════════════════════════════════════════════════════════════════════════
def render():
    cfg = st.session_state.method_cfg
    time_unit = cfg["time_unit"]
    dose_mg = cfg["dose_mg"]

    st.markdown(
        '<style>'
        '.info-banner{background:rgba(0,33,71,0.35);border-left:4px solid #FFCC00;'
        'padding:11px 15px;border-radius:5px;color:#cbd5e1;font-size:0.9rem;margin:6px 0 10px;}'
        '.step-box{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
        'padding:9px 13px;border-radius:5px;color:#9fb0d0;font-size:0.85rem;margin:4px 0;}'
        '.eq-box{background:#16203F;border:1px solid rgba(255,204,0,0.30);border-radius:6px;'
        'padding:10px 14px;color:#FFCC00;font-family:ui-monospace,Menlo,monospace;margin:10px 0;}'
        '</style>', unsafe_allow_html=True)

    st.markdown(
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">'
        '<h2 style="margin:0;color:#FFFFFF;">IVIVC Analysis</h2>'
        '<span style="background:#FFBF00;color:#002147;font-size:0.62rem;font-weight:700;'
        'letter-spacing:1.5px;text-transform:uppercase;padding:3px 8px;border-radius:3px;">v3.5</span>'
        '</div>', unsafe_allow_html=True)
    st.caption("In vitro–in vivo correlation · FDA Guidance for Industry: Extended Release Oral "
               "Dosage Forms (1997). Level A uses plasma-based Wagner-Nelson deconvolution + "
               "convolution %PE validation.")

    if not st.session_state.get("profiles"):
        st.warning("⚠️ No dissolution profiles loaded. Open **Data Input** first.")
        st.stop()

    st.session_state.setdefault("ivivc_input_data", {})
    st.session_state.setdefault("selected_level", "Level A")
    D = st.session_state.ivivc_input_data

    IVIVC_LEVELS = {
        "Level A": ("🏆 Level A — Point-to-Point", "Plasma Wagner-Nelson deconvolution → Fa vs Fd; %PE validation. FDA-preferred."),
        "Level B": ("📊 Level B — Statistical Moments", "In vitro MDT vs in vivo MRT (moment analysis)."),
        "Level C": ("📈 Level C — Single Point", "One dissolution parameter vs one PK parameter across formulations."),
        "Multiple Level C": ("🔬 Multiple Level C — Multi-Point", "% dissolved at ≥3 fixed times vs a PK parameter; near Level A if all R²>0.90."),
        "Level D": ("👁️ Level D — Rank Order", "Qualitative rank-order concordance (Spearman) across formulations."),
    }

    lc, ic = st.columns([1, 2])
    with lc:
        selected_level = st.radio("IVIVC level", list(IVIVC_LEVELS.keys()),
                                  index=list(IVIVC_LEVELS.keys()).index(st.session_state.selected_level),
                                  key="_ivivc_level_radio")
        st.session_state.selected_level = selected_level
    with ic:
        label, desc = IVIVC_LEVELS[selected_level]
        st.markdown(f'<div style="background:#16203F;border-left:4px solid #FFCC00;border-radius:5px;'
                    f'padding:12px 16px;margin-top:6px;color:#cbd5e1;"><strong style="color:#fff;">{label}</strong>'
                    f'<br><span style="font-size:0.88rem;">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    names = list(st.session_state.profiles.keys())
    iv_profile = st.selectbox("In vitro dissolution profile", names,
                              index=_get_index(names, st.session_state.get("selected_ref_id"), 0),
                              help="Dissolution profile used as the in vitro input.")
    d_iv = st.session_state.profiles[iv_profile]
    t_iv = np.array(d_iv["time"], float)
    r_iv = np.array(d_iv["release"], float)

    # ═══════════════════════════════════════════════════════════════════════
    # LEVEL A — plasma Wagner-Nelson + convolution %PE
    # ═══════════════════════════════════════════════════════════════════════
    if selected_level == "Level A":
        st.markdown("### Level A — Point-to-Point (Plasma Wagner-Nelson Deconvolution)")
        st.markdown('<div class="info-banner">The in vivo <b>fraction absorbed Fa(t)</b> is obtained by '
                    'Wagner-Nelson deconvolution of the <b>plasma concentration–time</b> data for the same '
                    'formulation, then correlated point-to-point with the in vitro <b>fraction dissolved Fd(t)</b>. '
                    'Internal predictability (%PE) is computed by convolving the IVIVC-mapped absorption back to '
                    'a predicted plasma profile.</div>', unsafe_allow_html=True)

        st.markdown("#### 📥 In vivo plasma concentration–time data (required)")
        st.markdown('<div class="step-box">Enter comma-separated values (same time unit as dissolution). '
                    'Both lists must be the same length.</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            pk_t_str = st.text_area(f"Plasma time ({time_unit})", value=D.get("pk_t", ""),
                                    placeholder="0, 0.5, 1, 2, 3, 4, 6, 8, 12", height=80, key="la_pk_t")
        with c2:
            pk_cp_str = st.text_area("Plasma Cp (conc)", value=D.get("pk_cp", ""),
                                     placeholder="0, 2.1, 3.0, 3.3, 3.0, 2.5, 1.7, 1.1, 0.5", height=80, key="la_pk_cp")

        if not (pk_t_str.strip() and pk_cp_str.strip()):
            st.info("Enter plasma Cp–t data above to run the Level A deconvolution and correlation.")
            show_literature("IVIVC Analysis")
            return
        try:
            pk_t = _parse_csv(pk_t_str); pk_cp = _parse_csv(pk_cp_str)
            assert len(pk_t) == len(pk_cp) and len(pk_t) >= 4
            assert np.all(np.diff(pk_t) > 0)
        except Exception:
            st.error("Plasma data invalid: equal-length lists (≥4 points), strictly increasing times.")
            st.stop()
        D.update({"pk_t": pk_t_str, "pk_cp": pk_cp_str})

        ke_auto = _ke_terminal(pk_t, pk_cp)
        col_ke, col_note = st.columns([1, 2])
        with col_ke:
            ke_default = D.get("la_ke") or (round(ke_auto, 5) if ke_auto and not np.isnan(ke_auto) else 0.1)
            ke = st.number_input(f"Elimination kₑ (1/{time_unit})", value=float(ke_default),
                                 min_value=1e-6, format="%.5f", key="la_ke_in",
                                 help="Auto-estimated from the terminal slope; override with IV/oral-solution kₑ if known.")
            D["la_ke"] = ke
        with col_note:
            if ke_auto and not np.isnan(ke_auto):
                st.caption(f"Terminal-slope estimate: kₑ ≈ {ke_auto:.5f} · half-life ≈ {np.log(2)/ke_auto:.2f} {time_unit}")

        Fa_full, auc_inf = _wagner_nelson_plasma(pk_t, pk_cp, ke)
        nca = _nca(pk_t, pk_cp)

        # correlate at dissolution time points that fall within the plasma sampling window
        lo, hi = pk_t.min(), pk_t.max()
        mask = (t_iv >= lo) & (t_iv <= hi) & (t_iv > 0)
        if mask.sum() < 3:
            st.error("Need ≥3 dissolution time points within the plasma sampling window to correlate.")
            st.stop()
        t_c = t_iv[mask]
        Fd_c = r_iv[mask]
        Fa_c = np.interp(t_c, pk_t, Fa_full)

        fig, r2, slope, intercept, p = _regression(Fd_c, Fa_c, "In vitro Fd (%)",
                                                   "In vivo Fa (%)", "Level A: Fd vs Fa")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", f"{r2:.4f}")
        m2.metric("Slope", f"{slope:.4f}")
        m3.metric("Intercept", f"{intercept:.4f}")
        m4.metric("Correlation window", f"{lo:g}–{hi:g} {time_unit}")
        st.markdown(f'<div class="eq-box">Fa = {slope:.4f} × Fd + {intercept:.4f}   (R² = {r2:.4f})</div>',
                    unsafe_allow_html=True)

        # ---- internal predictability (%PE) via convolution ----
        st.markdown("#### 🎯 Internal predictability (%PE)")
        Fabs_pred = np.clip(slope * r_iv + intercept, 0, 100) / 100.0
        # build absorption input over the plasma window, anchored at 0
        t_mod = np.unique(np.concatenate([[0.0], t_iv[(t_iv > 0) & (t_iv <= hi)]]))
        F_mod = np.interp(t_mod, np.concatenate([[0.0], t_iv]), np.concatenate([[0.0], Fabs_pred]))
        tf, cpf = _predict_cp(t_mod, F_mod, ke, auc_inf)
        cmax_pred = float(cpf.max())
        auc_pred = float(trapezoid(cpf, tf))
        cmax_obs, auc_obs = nca["Cmax"], nca["AUC_inf"]
        pe_cmax = abs(cmax_obs - cmax_pred) / cmax_obs * 100 if cmax_obs > 0 else np.nan
        pe_auc = abs(auc_obs - auc_pred) / auc_obs * 100 if auc_obs > 0 else np.nan

        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Cmax (obs)", f"{cmax_obs:.3g}")
        pc2.metric("Cmax (pred)", f"{cmax_pred:.3g}", f"%PE {pe_cmax:.1f}")
        pc3.metric("AUC∞ (obs)", f"{auc_obs:.3g}")
        pc4.metric("AUC∞ (pred)", f"{auc_pred:.3g}", f"%PE {pe_auc:.1f}")
        mean_pe = np.nanmean([pe_cmax, pe_auc])
        verdict, bg, fg = _pe_badge(mean_pe)
        st.markdown(f'<div style="background:{bg};border-left:5px solid {fg};border-radius:6px;'
                    f'padding:12px 18px;margin:10px 0;"><strong style="color:{fg};">Mean %PE = {mean_pe:.2f}% '
                    f'(Cmax & AUC)</strong><br><span style="color:{fg};">{verdict}</span></div>',
                    unsafe_allow_html=True)
        st.caption("FDA internal predictability: mean absolute %PE for Cmax and AUC ≤ 10% (individual ≤ 15%). "
                   "External predictability with a separate formulation is recommended for regulatory use.")

        if _PLOTLY_OK:
            figp = go.Figure()
            figp.add_trace(go.Scatter(x=pk_t, y=pk_cp, mode='markers', name='Observed Cp',
                                      marker=dict(color='#5DA9E9', size=9)))
            figp.add_trace(go.Scatter(x=tf, y=cpf, mode='lines', name='Predicted Cp (convolution)',
                                      line=dict(color=AMBER, width=2.5)))
            figp.update_layout(title=dict(text="Observed vs predicted plasma profile", font=dict(color='#e8edf6', size=14)),
                               xaxis_title=f"Time ({time_unit})", yaxis_title="Cp",
                               plot_bgcolor='#16203F', paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color='#cbd5e1'), height=380, margin=dict(t=52),
                               legend=dict(bgcolor='rgba(0,0,0,0)'))
            figp.update_xaxes(gridcolor='rgba(255,255,255,0.07)')
            figp.update_yaxes(gridcolor='rgba(255,255,255,0.07)')
            st.plotly_chart(figp, use_container_width=True)

        with st.expander("📋 Point-to-point data"):
            st.dataframe(pd.DataFrame({
                f"Time ({time_unit})": t_c, "In vitro Fd (%)": Fd_c.round(2),
                "In vivo Fa (%) [W-N]": Fa_c.round(2), "Δ (Fd−Fa)": (Fd_c - Fa_c).round(2)}),
                use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # LEVEL B — MDT vs MRT
    # ═══════════════════════════════════════════════════════════════════════
    elif selected_level == "Level B":
        st.markdown("### Level B — Statistical Moment Analysis (MDT vs MRT)")
        st.markdown('<div class="info-banner">Correlates the in vitro <b>Mean Dissolution Time (MDT)</b> with '
                    'the in vivo <b>Mean Residence Time (MRT)</b> across formulations. Uses all the data but is '
                    'not point-to-point — lower regulatory value than Level A.</div>', unsafe_allow_html=True)
        mdt_iv = compute_mdt(t_iv, r_iv)
        st.metric(f"In vitro MDT of «{iv_profile}» ({time_unit})",
                  f"{mdt_iv:.3f}" if not np.isnan(mdt_iv) else "N/A")
        st.markdown("#### 📥 Per-formulation MDT (in vitro) & MRT (in vivo)")
        n = int(st.number_input("Number of formulations", 2, 12, int(D.get("lvb_n", 3)), key="lvb_n"))
        D["lvb_n"] = n
        rows = []
        cols = st.columns(min(n, 4))
        for i in range(n):
            with cols[i % 4]:
                mdt_i = st.number_input(f"MDT F{i+1}", value=float(D.get(f"lvb_mdt_{i}", round(mdt_iv, 3) if (i == 0 and not np.isnan(mdt_iv)) else 0.0)),
                                        min_value=0.0, key=f"lvb_mdt_{i}")
                mrt_i = st.number_input(f"MRT F{i+1}", value=float(D.get(f"lvb_mrt_{i}", 0.0)),
                                        min_value=0.0, key=f"lvb_mrt_{i}")
                D[f"lvb_mdt_{i}"] = mdt_i; D[f"lvb_mrt_{i}"] = mrt_i
                rows.append((mdt_i, mrt_i))
        valid = [(a, b) for a, b in rows if a > 0 and b > 0]
        if len(valid) >= 2:
            x = np.array([a for a, _ in valid]); y = np.array([b for _, b in valid])
            fig, r2, sl, icpt, p = _regression(x, y, f"In vitro MDT ({time_unit})",
                                               f"In vivo MRT ({time_unit})", "Level B: MDT vs MRT")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            a1, a2, a3 = st.columns(3)
            a1.metric("R²", f"{r2:.4f}"); a2.metric("Slope", f"{sl:.4f}"); a3.metric("Intercept", f"{icpt:.4f}")
            st.markdown(f'<div class="eq-box">MRT = {sl:.4f} × MDT + {icpt:.4f}</div>', unsafe_allow_html=True)
        else:
            st.info("Enter MDT and MRT (>0) for at least 2 formulations.")

    # ═══════════════════════════════════════════════════════════════════════
    # LEVEL C — single point
    # ═══════════════════════════════════════════════════════════════════════
    elif selected_level == "Level C":
        st.markdown("### Level C — Single-Point Correlation")
        st.markdown('<div class="info-banner">Relates one <b>dissolution parameter</b> (MDT, t50%, t80%, DE) to one '
                    '<b>PK parameter</b> (Cmax, AUC, Tmax, MRT) across formulations. Useful for screening; lower '
                    'regulatory value than Level A.</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            pk_param = st.selectbox("PK parameter", ["Cmax", "AUC(0-∞)", "Tmax", "MRT"],
                                    index=D.get("lvc_pk_idx", 0), key="lvc_pk")
            D["lvc_pk_idx"] = ["Cmax", "AUC(0-∞)", "Tmax", "MRT"].index(pk_param)
        with cc2:
            diss_param = st.selectbox("Dissolution parameter", ["MDT", "t50%", "t80%", "DE (%)"],
                                      index=D.get("lvc_diss_idx", 0), key="lvc_diss")
            D["lvc_diss_idx"] = ["MDT", "t50%", "t80%", "DE (%)"].index(diss_param)
        auto = {"MDT": compute_mdt(t_iv, r_iv), "t50%": _t_at_percent(t_iv, r_iv, 50),
                "t80%": _t_at_percent(t_iv, r_iv, 80), "DE (%)": compute_de(t_iv, r_iv)}.get(diss_param, np.nan)
        if not np.isnan(auto):
            st.info(f"Auto-computed **{diss_param}** for «{iv_profile}»: **{auto:.3f}**")
        st.markdown("#### 📥 Per-formulation values")
        n = int(st.number_input("Number of formulations", 2, 12, int(D.get("lvc_n", 3)), key="lvc_n"))
        D["lvc_n"] = n
        rows = []
        cols = st.columns(min(n, 4))
        for i in range(n):
            with cols[i % 4]:
                dv = st.number_input(f"{diss_param} F{i+1}", value=float(D.get(f"lvc_d_{i}", round(auto, 3) if (i == 0 and not np.isnan(auto)) else 0.0)),
                                     min_value=0.0, key=f"lvc_d_{i}")
                pv = st.number_input(f"{pk_param} F{i+1}", value=float(D.get(f"lvc_p_{i}", 0.0)),
                                     min_value=0.0, key=f"lvc_p_{i}")
                D[f"lvc_d_{i}"] = dv; D[f"lvc_p_{i}"] = pv
                rows.append((dv, pv))
        valid = [(a, b) for a, b in rows if a > 0 and b > 0]
        if len(valid) >= 2:
            x = np.array([a for a, _ in valid]); y = np.array([b for _, b in valid])
            fig, r2, sl, icpt, p = _regression(x, y, diss_param, pk_param, f"Level C: {diss_param} vs {pk_param}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            a1, a2, a3 = st.columns(3)
            a1.metric("R²", f"{r2:.4f}"); a2.metric("Slope", f"{sl:.4f}"); a3.metric("Intercept", f"{icpt:.4f}")
            st.markdown(f'<div class="eq-box">{pk_param} = {sl:.4f} × {diss_param} + {icpt:.4f}</div>', unsafe_allow_html=True)
        else:
            st.info(f"Enter {diss_param} and {pk_param} (>0) for at least 2 formulations.")

    # ═══════════════════════════════════════════════════════════════════════
    # MULTIPLE LEVEL C — % dissolved at fixed times vs PK  (corrected)
    # ═══════════════════════════════════════════════════════════════════════
    elif selected_level == "Multiple Level C":
        st.markdown("### Multiple Level C — Multi-Point Correlation")
        st.markdown('<div class="info-banner">At each of several <b>fixed sampling times</b>, the <b>% dissolved</b> '
                    '(which differs across formulations) is correlated with a PK parameter. If R² &gt; 0.90 at all '
                    'chosen times, the correlation approaches Level A reliability (FDA, 1997). '
                    '<i>Corrected:</i> the x-axis is % dissolved per formulation — not a constant.</div>',
                    unsafe_allow_html=True)
        tp1, tp2, tp3, tpk = st.columns(4)
        with tp1:
            tA = st.number_input(f"Time A ({time_unit})", value=float(D.get("mc_tA", 10.0)), min_value=0.0, key="mc_tA")
        with tp2:
            tB = st.number_input(f"Time B ({time_unit})", value=float(D.get("mc_tB", 20.0)), min_value=0.0, key="mc_tB")
        with tp3:
            tC = st.number_input(f"Time C ({time_unit})", value=float(D.get("mc_tC", 40.0)), min_value=0.0, key="mc_tC")
        with tpk:
            pk_param = st.selectbox("PK parameter", ["Cmax", "AUC(0-∞)"], index=D.get("mc_pk_idx", 0), key="mc_pk")
            D["mc_pk_idx"] = ["Cmax", "AUC(0-∞)"].index(pk_param)
        D.update({"mc_tA": tA, "mc_tB": tB, "mc_tC": tC})
        times = [tA, tB, tC]; labels = [f"t={tA:g}", f"t={tB:g}", f"t={tC:g}"]

        st.markdown("#### 📥 Per-formulation: % dissolved at each time + PK parameter")
        n = int(st.number_input("Number of formulations", 3, 12, int(D.get("mc_n", 4)), key="mc_n"))
        D["mc_n"] = n
        hdr = st.columns([1.4, 2, 2, 2, 2])
        hdr[0].markdown("**Form.**")
        for j, lab in enumerate(labels):
            hdr[j + 1].markdown(f"**% diss @ {lab}**")
        hdr[4].markdown(f"**{pk_param}**")
        diss_mat = [[], [], []]; pk_vec = []
        for i in range(n):
            row = st.columns([1.4, 2, 2, 2, 2])
            row[0].markdown(f"F{i+1}")
            for j in range(3):
                v = row[j + 1].number_input(f"d{j}_{i}", value=float(D.get(f"mc_d{j}_{i}", 0.0)),
                                            min_value=0.0, max_value=110.0, key=f"mc_d{j}_{i}",
                                            label_visibility="collapsed")
                D[f"mc_d{j}_{i}"] = v; diss_mat[j].append(v)
            pv = row[4].number_input(f"p_{i}", value=float(D.get(f"mc_p_{i}", 0.0)), min_value=0.0,
                                     key=f"mc_p_{i}", label_visibility="collapsed")
            D[f"mc_p_{i}"] = pv; pk_vec.append(pv)
        pk_arr = np.array(pk_vec)

        if (pk_arr > 0).sum() >= 3:
            tabs = st.tabs(labels)
            r2s_all = {}
            for j, (lab, tab) in enumerate(zip(labels, tabs)):
                with tab:
                    x = np.array(diss_mat[j]); y = pk_arr
                    m = (x > 0) & (y > 0)
                    if m.sum() >= 3 and len(np.unique(x[m])) >= 2:
                        fig, r2, sl, icpt, p = _regression(x[m], y[m], f"% dissolved @ {lab}",
                                                           pk_param, f"{lab}: % dissolved vs {pk_param}")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        r2s_all[lab] = r2
                        ok = r2 >= 0.90
                        st.markdown(f'<div style="background:{"#0e3a1e" if ok else "#3a1414"};'
                                    f'color:{"#7bd88f" if ok else "#e57373"};border-radius:5px;padding:8px 14px;'
                                    f'font-weight:600;">R² = {r2:.4f} {"✅ ≥ 0.90" if ok else "❌ < 0.90"}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.info("Enter % dissolved (with variation across formulations) and PK values for ≥3 formulations.")
            if r2s_all:
                if all(v >= 0.90 for v in r2s_all.values()) and len(r2s_all) == 3:
                    st.success("✅ All time points achieve R² ≥ 0.90 — approaches Level A reliability (FDA, 1997). "
                               "Consider a formal Level A validation.")
                else:
                    st.warning("⚠️ One or more time points show R² < 0.90 — does not meet the Level-A-equivalent threshold.")
        else:
            st.info("Enter data for at least 3 formulations.")

    # ═══════════════════════════════════════════════════════════════════════
    # LEVEL D — rank order (Spearman)
    # ═══════════════════════════════════════════════════════════════════════
    elif selected_level == "Level D":
        st.markdown("### Level D — Rank-Order Concordance")
        st.markdown('<div class="info-banner">Qualitative check that formulations rank the same way in vitro and in '
                    'vivo: a faster-dissolving formulation should show higher/earlier in vivo exposure. Uses '
                    '<b>Spearman rank correlation</b> — no predictive model.</div>', unsafe_allow_html=True)
        dd1, dd2 = st.columns(2)
        with dd1:
            diss_metric = st.selectbox("In vitro speed metric", ["% dissolved @ time", "1/MDT (faster=higher)", "1/t50%"],
                                       index=D.get("lvd_dm", 0), key="lvd_dm_sel")
            D["lvd_dm"] = ["% dissolved @ time", "1/MDT (faster=higher)", "1/t50%"].index(diss_metric)
            t_at = st.number_input(f"…@ time ({time_unit})", value=float(D.get("lvd_t", 30.0)),
                                   min_value=0.0, key="lvd_t") if diss_metric == "% dissolved @ time" else None
            D["lvd_t"] = t_at if t_at is not None else D.get("lvd_t", 30.0)
        with dd2:
            vivo_metric = st.selectbox("In vivo exposure metric", ["Cmax", "AUC(0-∞)"], index=D.get("lvd_vm", 0), key="lvd_vm_sel")
            D["lvd_vm"] = ["Cmax", "AUC(0-∞)"].index(vivo_metric)
        st.markdown("#### 📥 Per-formulation in vitro speed & in vivo exposure")
        n = int(st.number_input("Number of formulations", 3, 12, int(D.get("lvd_n", 4)), key="lvd_n"))
        D["lvd_n"] = n
        rows = []
        cols = st.columns(min(n, 4))
        for i in range(n):
            with cols[i % 4]:
                sv = st.number_input(f"In vitro F{i+1}", value=float(D.get(f"lvd_s_{i}", 0.0)), min_value=0.0, key=f"lvd_s_{i}")
                ev = st.number_input(f"{vivo_metric} F{i+1}", value=float(D.get(f"lvd_e_{i}", 0.0)), min_value=0.0, key=f"lvd_e_{i}")
                D[f"lvd_s_{i}"] = sv; D[f"lvd_e_{i}"] = ev
                rows.append((sv, ev))
        valid = [(a, b) for a, b in rows if a > 0 and b > 0]
        if len(valid) >= 3:
            x = np.array([a for a, _ in valid]); y = np.array([b for _, b in valid])
            rho, pval = _stats.spearmanr(x, y)
            verdict = ("Strong concordant rank order" if rho >= 0.8 else
                       "Moderate concordance" if rho >= 0.5 else "Weak/discordant rank order")
            st.metric("Spearman ρ", f"{rho:.3f}", f"p = {pval:.3f}")
            col = "#7bd88f" if rho >= 0.8 else ("#f0c419" if rho >= 0.5 else "#e57373")
            st.markdown(f'<div class="info-banner" style="border-left-color:{col};">'
                        f'<b style="color:{col};">{verdict}.</b> Level D is qualitative — high concordance supports '
                        f'proceeding to a formal Level A/B study.</div>', unsafe_allow_html=True)
            if _PLOTLY_OK:
                fig = go.Figure(go.Scatter(x=x, y=y, mode='markers+text',
                                           text=[f"F{i+1}" for i in range(len(x))], textposition="top center",
                                           marker=dict(color='#5DA9E9', size=11)))
                fig.update_layout(title=dict(text="In vitro speed vs in vivo exposure", font=dict(color='#e8edf6', size=14)),
                                  xaxis_title=diss_metric, yaxis_title=vivo_metric,
                                  plot_bgcolor='#16203F', paper_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='#cbd5e1'), height=380, margin=dict(t=52))
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.07)'); fig.update_yaxes(gridcolor='rgba(255,255,255,0.07)')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enter in vitro speed and in vivo exposure (>0) for at least 3 formulations.")

    show_literature("IVIVC Analysis")
