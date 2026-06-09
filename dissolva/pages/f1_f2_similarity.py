"""DissolvA page module: f1 and f2 Similarity. Extracted from app.py (Phase 3b modularization)."""
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
    compute_de, r2s, r2adj, aic_fn, msc_fn, _nz, fda_f2_mask, f1_score, f2_score)
from dissolva import engine_client
from dissolva.state import (current_tier, require_tier, _safe_profile_names,
    _get_index, _rename_profile, _clear_all)
from dissolva.content import (show_literature, show_all_references,
    analyze_profile_shape, bootstrap_recommendation)


def render():
    cfg = st.session_state.method_cfg
    time_unit = cfg["time_unit"]
    q_time = cfg["q_time"]
    q_limit = cfg["q_limit"]
    st.header("f1 and f2 Similarity Factor Analysis")

    st.markdown("""
    <div class="step-box">
    <strong>How to use this page:</strong><br>
    1. Load your dissolution profiles in the <em>Data Input</em> page first.<br>
    2. <strong>Reference Profile</strong>: Select the innovator / originator product (the product you are comparing against).<br>
    3. <strong>Test Profile</strong>: Select your formulation (the product you want to compare).<br>
    4. The calculator uses only time points where the reference release is 85% or less (FDA guidance).<br>
    5. <strong>f2 >= 50</strong> means the profiles are similar. <strong>f1 <= 15</strong> means acceptable difference.
    </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.profiles) < 2:
        st.warning("At least 2 profiles are required. Go to Data Input and load your reference and test profiles.")
        st.stop()

    names = list(st.session_state.profiles.keys())

    def _on_ref_change_f2():
        st.session_state.selected_ref_id = st.session_state._f2_ref_sel
    def _on_test_change_f2():
        st.session_state.selected_test_id = st.session_state._f2_test_sel

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Reference Profile** *(innovator / originator)*")
        ref_nm = st.selectbox(
            "Reference", names,
            index=_get_index(names, st.session_state.selected_ref_id, 0),
            key="_f2_ref_sel",
            on_change=_on_ref_change_f2,
            label_visibility="collapsed"
        )
        st.session_state.selected_ref_id = ref_nm
        st.caption(f"This is the product you compare AGAINST: {ref_nm}")
    with col2:
        st.markdown("**Test Profile** *(your formulation)*")
        test_options_f2 = [n for n in names if n != ref_nm]
        test_nm = st.selectbox(
            "Test", names,
            index=_get_index(names, st.session_state.selected_test_id, min(1,len(names)-1)),
            key="_f2_test_sel",
            on_change=_on_test_change_f2,
            label_visibility="collapsed"
        )
        st.session_state.selected_test_id = test_nm
        st.caption(f"This is the product you want to COMPARE: {test_nm}")

    if ref_nm == test_nm:
        st.error("Reference and Test must be different profiles."); st.stop()

    t_ref=np.array(st.session_state.profiles[ref_nm]["time"])
    r_ref=np.array(st.session_state.profiles[ref_nm]["release"])
    t_tst=np.array(st.session_state.profiles[test_nm]["time"])
    r_tst=np.array(st.session_state.profiles[test_nm]["release"])

    common=np.intersect1d(t_ref,t_tst)
    if len(common)==0:
        st.error("No common time points between the two profiles."); st.stop()

    rr=np.array([r_ref[np.where(t_ref==ti)[0][0]] for ti in common])
    rt=np.array([r_tst[np.where(t_tst==ti)[0][0]] for ti in common])

    # FDA Guidance (1997): ref ≤85% noktaları + 85'i ilk aşan nokta dahil (paylaşılan kural)
    mask = fda_f2_mask(rr)
    rrf, rtf = rr[mask], rt[mask]
    n_points_above85 = int(np.sum(rr > 85))
    n_points_excluded = max(0, n_points_above85 - 1)

    if len(rrf) == 0:
        st.error("No valid time points (reference ≤ 85%)."); st.stop()
    if len(rrf) < 3:
        st.error(
            f"f2 requires at least 3 evaluation points (FDA 1997 Guidance); only "
            f"{len(rrf)} point(s) qualify (reference ≤ 85%, plus the first point above). "
            "Add earlier sampling time points to compute a valid f2."
        ); st.stop()

    # f1/f2 via backend API when configured, else local engine (engine_client)
    f1, f2, _n_used = engine_client.similarity(t_ref, r_ref, t_tst, r_tst)

    mc1,mc2,mc3,mc4,mc5=st.columns(5)
    mc1.metric("f1 (Difference)", f"{f1:.2f}",
        "✓ PASS (≤15)" if f1<=15 else "✗ FAIL (>15)",
        delta_color="normal" if f1<=15 else "inverse")
    mc2.metric("f2 (Similarity)", f"{f2:.2f}",
        "✓ SIMILAR (≥50)" if f2>=50 else "✗ DISSIMILAR (<50)",
        delta_color="normal" if f2>=50 else "inverse")
    mc3.metric("Points used (n)", len(rrf))
    mc4.metric("Max |Delta R| (%)", f"{np.max(np.abs(rrf-rtf)):.2f}")
    mc5.metric("Excluded (>85%)", n_points_excluded,
        help="FDA (1997): After exceeding 85%, only 1 additional point may be included.")
    if n_points_excluded > 0:
        st.info(
            f"ℹ️ **FDA 85% Rule applied:** {n_points_excluded} time point(s) beyond the first "
            f"exceeding 85% have been excluded. The **first point exceeding 85%** is included in "
            f"the f2 calculation; subsequent points are excluded. "
            f"*(FDA Guidance 1997, Section V.B)*"
        )

    # ── Is Bootstrap Required? — paylaşılan şart kontrolü (Bootstrap sayfası ile aynı) ──
    st.markdown("---")
    st.markdown("#### 🔍 Is Bootstrap f2 Required?")
    _rec = bootstrap_recommendation(st.session_state.profiles, ref_nm, test_nm)
    if not _rec["needs_boot"] and _rec["fda_cv_ok"]:
        st.success(
            f"**✅ Bootstrap f2 Analysis Not Required — Standard f2 Test is Sufficient.**\n\n"
            f"FDA criteria are met:\n\n"
            f"- f2 = **{f2:.2f}** — outside the 45–55 boundary zone\n"
            f"- n = **{_rec['n_vessels']}** vessels\n"
            f"- Max CV% (t ≤ 15 min) = **{_rec['cv_early_max']:.1f}%** ≤ 20% ✓\n"
            f"- Max CV% (t > 15 min) = **{_rec['cv_late_max']:.1f}%** ≤ 10% ✓\n\n"
            f"Single-point f2 test is conclusive per FDA (1997) Guidance, Section V."
        )
    else:
        _criteria_text = "\n".join([f"- {c}" for c in _rec["reasons"]]) or "- (criteria)"
        st.warning(
            f"**⚠️ Bootstrap f2 Analysis Recommended**\n\n"
            f"Standard f2 test is insufficient:\n\n{_criteria_text}\n\n"
            f"**Recommended method: {_rec['recommended_method']} Bootstrap.** "
            f"{'Because CV% > 15%, Nonparametric is more reliable. ' if _rec['cv_max'] > 15 else 'CV% ≤ 15%, Parametric is sufficient. '}"
            f"→ Go to the **Bootstrap f2 Analysis** page.\n\n"
            f"📌 *Shah VP et al. Pharm Res. 1998;15(6):889-896 | FDA Guidance 1997*"
        )
    if not _rec["has_cv"]:
        st.caption("Note: these profiles have no CV/RSD data, so the CV criteria could not be checked "
                   "automatically. Upload vessel-level data to enable the full check.")

    vf1="PASS - f1 <= 15: Profiles have acceptable difference" if f1<=15 else "FAIL - f1 > 15: Significant difference detected"
    vf2="SIMILAR - f2 >= 50: Profiles are bioequivalent (FDA)" if f2>=50 else "DISSIMILAR - f2 < 50: Profiles are NOT similar"
    color1="#c6efce" if f1<=15 else "#ffc7ce"
    color2="#c6efce" if f2>=50 else "#ffc7ce"
    st.markdown(
        f'<div style="background:{color1};border-radius:5px;padding:10px 14px;margin:6px 0;"><strong>f1:</strong> {vf1}</div>' +
        f'<div style="background:{color2};border-radius:5px;padding:10px 14px;margin:6px 0;"><strong>f2:</strong> {vf2}</div>',
        unsafe_allow_html=True
    )

    df_cmp=pd.DataFrame({
        f"Time ({time_unit})":common,"Reference: "+ref_nm+" (%)":rr,
        "Test: "+test_nm+" (%)":rt,"|Diff| (%)":np.abs(rr-rt).round(2),
        "Used in f2":["Yes (ref<=85%)" if r<=85 else ("Yes (1st >85%)" if i==int(np.where(rr>85)[0][0]) else "No (ref>85%)") if np.any(rr>85) else "No (ref>85%)" for i,r in enumerate(rr)]
    })
    st.dataframe(df_cmp,use_container_width=True)

    # -- Plot options
    opt_c1, opt_c2, opt_c3, opt_c4 = st.columns(4)
    with opt_c1:
        show_cutoff = st.radio(
            "85% Cutoff Line (FDA)",
            ["Show", "Hide"], horizontal=True, key="f2_cutoff",
            help="Only time points where reference ≤ 85% are used in f2 calculation."
        ) == "Show"
    with opt_c2:
        show_diff_area = st.radio(
            "Difference Area",
            ["Show", "Hide"], horizontal=True, key="f2_area",
            help="Shaded area between reference and test profiles."
        ) == "Show"
    with opt_c3:
        show_q_f2 = st.radio(
            f"Q Value Line ({q_limit:.0f}%)",
            ["Show", "Hide"], horizontal=True, key="f2_qline",
            help=f"FDA/USP acceptance criterion: Q = {q_limit:.0f}%"
        ) == "Show"
    with opt_c4:
        show_qt_f2 = st.radio(
            f"Q Time ({q_time:.0f} {time_unit})",
            ["Show", "Hide"], horizontal=True, key="f2_qtline",
            help=f"Q time point: {q_time:.0f} {time_unit}"
        ) == "Show"

    fig, ax = plt.subplots(figsize=(10, 5)); style_ax(fig, ax)

    # Reference error bars
    sd_ref = np.array(st.session_state.profiles[ref_nm].get("sd") or [0.0]*len(t_ref))
    sd_tst = np.array(st.session_state.profiles[test_nm].get("sd") or [0.0]*len(t_tst))
    has_sd_ref = not np.all(sd_ref == 0)
    has_sd_tst = not np.all(sd_tst == 0)

    if has_sd_ref:
        ax.errorbar(t_ref, r_ref, yerr=sd_ref, fmt="o-", color=OXFORD, lw=2.5,
                    ms=7, capsize=4, capthick=1.5, elinewidth=1.2, alpha=0.9,
                    label=f"Reference: {ref_nm}")
    else:
        ax.plot(t_ref, r_ref, "o-", color=OXFORD, lw=2.5, ms=7,
                label=f"Reference: {ref_nm}")

    if has_sd_tst:
        ax.errorbar(t_tst, r_tst, yerr=sd_tst, fmt="s--", color="#c0392b", lw=2.5,
                    ms=7, capsize=4, capthick=1.5, elinewidth=1.2, alpha=0.9,
                    label=f"Test: {test_nm}")
    else:
        ax.plot(t_tst, r_tst, "s--", color="#c0392b", lw=2.5, ms=7,
                label=f"Test: {test_nm}")

    if show_cutoff:
        ax.axhline(85, color=AMBER, lw=1.2, ls=":", alpha=0.85,
                   label="85% cutoff (FDA f2 criterion)")
    if show_diff_area:
        ax.fill_between(common, rr, rt, alpha=0.12, color="#c0392b",
                        label="Difference area")
    if show_q_f2:
        ax.axhline(q_limit, color=AMBER, lw=1.3, ls="--", alpha=0.85,
                   label=f"Q = {q_limit:.0f}% (USP/FDA)")
    if show_qt_f2:
        ax.axvline(q_time, color="#27ae60", lw=1.2, ls=":", alpha=0.8,
                   label=f"Q-time = {q_time:.0f} {time_unit}")

    # Force origin (0,0)
    t_all_f2 = np.concatenate([t_ref, t_tst])
    ax.set_xlim(left=0, right=t_all_f2.max() * 1.05)
    ax.set_ylim(bottom=0, top=112)

    ax.set_xlabel(f"Time ({time_unit})", fontsize=11)
    ax.set_ylabel("Cumulative Drug Released (%)", fontsize=11)
    ax.set_title(
        f"f1 = {f1:.2f}  |  f2 = {f2:.2f}  |  {ref_nm} vs {test_nm}",
        fontsize=12, color=OXFORD, pad=12
    )
    ax.legend(fontsize=9, framealpha=0.9)
    st.pyplot(fig)
    plt.close()

    # -- Equations in proper format
    st.markdown("**Formulas:**")
    st.latex(r"f_1 = \frac{\sum |R_t - T_t|}{\sum R_t} \times 100")
    st.latex(r"f_2 = 50 \cdot \log_{10}\left(\frac{100}{\sqrt{1 + \frac{1}{n}\sum(R_t-T_t)^2}}\right)")
    st.caption(
        "Rt = reference cumulative release at time t | "
        "Tt = test cumulative release at time t | "
        "n = number of time points used (reference <= 85%)"
    )
    show_literature("f1 and f2 Similarity")

# ===========================================================================
# PAGE: BOOTSTRAP f2 ANALYSIS
# ===========================================================================
