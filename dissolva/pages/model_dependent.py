"""Model-Dependent Similarity (Pro / industry).

FDA (1997) *Dissolution Testing of IR Solid Oral Dosage Forms* §V.B (model-
independent multivariate confidence region, MSD) and §V.C (model-dependent
approach): every unit is fitted with the same kinetic model, the parameter
vectors are compared with Hotelling's T² / Mahalanobis distance and a 90 %
confidence region, and the upper bound of that region is checked against a
similarity limit. An *applicability gate* (ICH M13B §2.4 low-dissolution and
very-rapid provisions, FDA/EMA f2 pre-conditions, parameter identifiability)
runs first so the statistic is never presented without its context.

All computation runs in the DissolvA engine service (backend); this page only
collects inputs and renders results.
"""
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

from dissolva import engine_client, access
from dissolva.theme import AMBER
from dissolva.state import _get_index
from dissolva.content import show_literature

_MODELS = {
    "Weibull (2-p, F∞ = 100) — DDSolver 'Weibull-2'": "weibull",
    "Weibull with Fmax (3-p) — recommended for incomplete release": "weibull_fmax",
    "First order": "first_order",
    "Korsmeyer–Peppas": "korsmeyer_peppas",
    "Logistic": "logistic",
    "Gompertz": "gompertz",
}
_BLUE = "#5DA9E9"
_RED = "#FF6B6B"


def _fmt(x, nd=3):
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def _lock():
    st.info(
        "🔒 **Model-Dependent Similarity is an industry (Pro) feature.** It provides the FDA 1997 "
        "§V.B/V.C multivariate statistics (Hotelling T², Mahalanobis distance, 90 % confidence "
        "region, similarity limit) that regulators sometimes request when f2 cannot be applied. "
        "Ask for access at **msinankaynak@gmail.com** — your account e-mail will be enabled."
    )


def _ellipse_from_cov(S, center, scale2, idx=(0, 1), n=120):
    """Boundary {δ : (δ-c)' S⁻¹ (δ-c) = scale2} projected on two parameters."""
    S = np.asarray(S, float)
    i, j = idx
    S2 = S[np.ix_([i, j], [i, j])]
    try:
        L = np.linalg.cholesky(S2)
    except np.linalg.LinAlgError:
        return None, None
    th = np.linspace(0, 2 * np.pi, n)
    circ = np.vstack([np.cos(th), np.sin(th)]) * np.sqrt(max(scale2, 0.0))
    pts = (L @ circ).T + np.asarray(center, float)[[i, j]]
    return pts[:, 0], pts[:, 1]


def _param_space_plot(dep, ref_name, test_name, alpha=0.10):
    """Difference-space plot: 90 % CR ellipse of the mean parameter difference and the
    similarity-limit ellipse around the origin (both in the pooled Mahalanobis metric)."""
    if not _PLOTLY_OK or dep.get("p", 0) < 2:
        return None
    names = dep["param_names"]; ln = dep["ln_transformed"]
    lab = [("ln " if ln[k] else "") + names[k] for k in range(len(names))]
    ref_q = np.asarray(dep["ref_q"], float); test_q = np.asarray(dep["test_q"], float)
    rm = np.asarray(dep["ref_mean_q"], float); tm = np.asarray(dep["test_mean_q"], float)
    d = tm - rm
    Sp = np.asarray(dep["pooled_cov"], float)
    fig = go.Figure()
    # unit clouds, centred on the reference mean (so the origin = "no difference")
    fig.add_trace(go.Scatter(x=ref_q[:, 0] - rm[0], y=ref_q[:, 1] - rm[1], mode="markers",
                             marker=dict(color=_BLUE, size=8, opacity=0.75), name=f"{ref_name} units"))
    fig.add_trace(go.Scatter(x=test_q[:, 0] - rm[0], y=test_q[:, 1] - rm[1], mode="markers",
                             marker=dict(color=_RED, size=8, opacity=0.75, symbol="diamond"), name=f"{test_name} units"))
    # similarity limit ellipse (Mahalanobis radius = limit) around the origin
    if dep.get("limit") is not None:
        ex, ey = _ellipse_from_cov(Sp, np.zeros(len(d)), float(dep["limit"]) ** 2)
        if ex is not None:
            fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#7CFC9A", width=2, dash="dot"),
                                     name=f"Similarity limit (MSD = {dep['limit']:.2f})"))
    # 90 % confidence region of the true difference, centred at d
    KF = (dep.get("K") or 0) * (dep.get("F") or 0)
    ex, ey = _ellipse_from_cov(Sp, d, KF)
    if ex is not None:
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", fill="toself", fillcolor="rgba(255,204,0,0.12)",
                                 line=dict(color=AMBER, width=2.5), name=f"{int(round((1 - alpha) * 100))} % confidence region"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(color="white", size=11, symbol="x"),
                             name="No difference (origin)"))
    fig.add_trace(go.Scatter(x=[d[0]], y=[d[1]], mode="markers", marker=dict(color=AMBER, size=12, symbol="star"),
                             name=f"Observed difference (MSD = {dep['D']:.2f})"))
    fig.update_layout(title=dict(text="Parameter space — difference from reference (test − reference)",
                                 font=dict(color="#e8edf6", size=14)),
                      xaxis_title=f"Δ {lab[0]}", yaxis_title=f"Δ {lab[1]}",
                      plot_bgcolor="#16203F", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"),
                      height=460, margin=dict(t=54), legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)", zeroline=True, zerolinecolor="rgba(255,255,255,0.25)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zeroline=True, zerolinecolor="rgba(255,255,255,0.25)",
                     scaleanchor="x", scaleratio=1)
    return fig


def _profile_plot(t, gate, ref_name, test_name):
    if not _PLOTLY_OK:
        return None
    fig = go.Figure()
    for m, s, nm, col in ((gate["ref_mean"], gate["ref_sd"], ref_name, _BLUE),
                          (gate["test_mean"], gate["test_sd"], test_name, _RED)):
        fig.add_trace(go.Scatter(x=t, y=m, mode="lines+markers", name=nm,
                                 error_y=dict(type="data", array=s, visible=True, color=col),
                                 line=dict(color=col, width=2.5), marker=dict(size=8)))
    fig.add_hline(y=10, line=dict(color="#7CFC9A", dash="dot"), annotation_text="10 % (ICH M13B low-dissolution)",
                  annotation_position="top left", annotation_font_color="#7CFC9A")
    fig.update_layout(title=dict(text="Mean dissolution profiles ± SD", font=dict(color="#e8edf6", size=14)),
                      xaxis_title="Time", yaxis_title="% dissolved", plot_bgcolor="#16203F",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"), height=380, margin=dict(t=54),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)"); fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", rangemode="tozero")
    return fig


def render():
    st.header("Model-Dependent Similarity")
    st.markdown(
        "<p style='color:#9fb0d0;margin-top:-8px'>FDA 1997 §V.B/V.C · Sathe, Tsong & Shah (1996) · "
        "Tsong et al. (1996) · ICH M13B §2.4 applicability gate — unit-level model fit → Hotelling T² → "
        "Mahalanobis distance (MSD) → 90 % confidence region vs. similarity limit.</p>",
        unsafe_allow_html=True,
    )
    if not access.is_pro():
        _lock()
        show_literature("model_dependent")
        return

    # ── Guard: raw vessel data ────────────────────────────────────────────────
    profiles_with_raw = {
        nm: d for nm, d in st.session_state.get("profiles", {}).items()
        if d.get("raw") and d.get("vessels") and len(d.get("vessels", [])) >= 3
    }
    if len(profiles_with_raw) < 2:
        st.warning(
            "This analysis needs **raw vessel-level data** (≥3 units, 12 recommended) for at least 2 profiles. "
            f"Profiles with raw data: **{list(profiles_with_raw.keys()) or 'None'}**. "
            "Upload with 'Excel / CSV Upload (Raw Vessel Data)' in Data Input."
        )
        return
    names = list(profiles_with_raw.keys())

    # ── Inputs ────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        ref = st.selectbox("Reference profile", names, index=_get_index(names, st.session_state.get("selected_ref_id"), 0), key="md_ref")
    with c2:
        tests = [n for n in names if n != ref]
        test = st.selectbox("Test profile", tests, key="md_test")
    with c3:
        ref2_opts = ["— none (use ±δ shift limit) —"] + [n for n in names if n not in (ref, test)]
        ref2 = st.selectbox("2nd reference batch (similarity region)", ref2_opts, index=0, key="md_ref2",
                            help="FDA: the similarity region should reflect the batch-to-batch variation of approved "
                                 "reference batches. Select a second reference batch to derive it; otherwise the "
                                 "±δ % shift convention (DDSolver 'Max_MSD') is used.")
    d_ref, d_test = profiles_with_raw[ref], profiles_with_raw[test]
    t_ref = [float(x) for x in d_ref["time"]]; t_test = [float(x) for x in d_test["time"]]
    common = sorted(set(t_ref) & set(t_test))
    if len(common) < 3:
        st.error("The two profiles share fewer than 3 time points."); return

    c4, c5, c6, c7 = st.columns([2.2, 1, 1, 1])
    with c4:
        model_lab = st.selectbox("Kinetic model (fitted to every unit)", list(_MODELS.keys()), key="md_model")
    with c5:
        conf = st.selectbox("Confidence", ["90 %", "95 %"], key="md_conf")
    with c6:
        delta = st.number_input("δ shift (%)", 1.0, 50.0, 10.0, 1.0, key="md_delta",
                                help="Absolute % difference that defines the similarity limit (FDA/Tsong: 10 %).")
    with c7:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run = st.button("▶ Run analysis", type="primary", key="md_run", use_container_width=True)
    _default_t = [x for x in common if x > 0] or common   # ICH M13B: zero excluded
    use_times = st.multiselect("Time points used", common, default=_default_t, key="md_times",
                               help="ICH M13B: ≤6 points, up to the plateau; the same subset is used for every unit.")
    if len(use_times) < 3:
        st.error("Select at least 3 time points."); return

    if run:
        def _rows(d, ts):
            idx = [t_ref.index(x) if d is d_ref else [float(v) for v in d["time"]].index(x) for x in ts]
            return [list(map(float, d["raw"][i])) for i in idx]
        ref2_raw = None
        if not ref2.startswith("—"):
            d2 = profiles_with_raw[ref2]
            t2 = [float(x) for x in d2["time"]]
            if not set(use_times) <= set(t2):
                st.error("The 2nd reference batch does not contain all selected time points."); return
            ref2_raw = [list(map(float, d2["raw"][t2.index(x)])) for x in use_times]
        with st.spinner("Fitting every unit and computing the multivariate statistics…"):
            res = engine_client.msd(
                use_times, _rows(d_ref, use_times), _rows(d_test, use_times),
                model=_MODELS[model_lab], alpha=0.10 if conf.startswith("90") else 0.05,
                delta=float(delta), limit_mode="ref_batches" if ref2_raw is not None else "shift",
                ref2_raw=ref2_raw,
            )
        if not res or res.get("error"):
            st.error(f"Engine error: {res.get('error') if res else 'no response'}"); return
        res["_ref"], res["_test"] = ref, test
        st.session_state["msd_result"] = res

    res = st.session_state.get("msd_result")
    if not res or res.get("_ref") != ref or res.get("_test") != test:
        st.caption("Set the inputs and click **Run analysis**.")
        show_literature("model_dependent")
        return

    t = res["time"]; g = res["gate"]; dep = res["dependent"]; ind = res["independent"]
    tab_gate, tab_dep, tab_ind, tab_tbl = st.tabs(["1 · Applicability gate", "2 · Model-dependent", "3 · Model-independent MSD", "4 · Tables"])

    # ── Tab 1: gate ───────────────────────────────────────────────────────────
    with tab_gate:
        if g["both_plateau_below_10"]:
            st.success("✅ **ICH M13B §2.4 — both products plateau below 10 %:** no similarity test is required; "
                       "similarity can be assumed. The statistics on the next tabs are supportive only.")
        elif g["very_rapid"]:
            st.success("✅ **Very rapid dissolution (≥85 % in 15 min for both):** similar without further evaluation.")
        elif g["f2_applicable"] and not g["high_variability_sd8"]:
            st.info("ℹ️ f2 pre-conditions are met — f2 (≥50) is the primary criterion; model-dependent results are supportive.")
        else:
            st.warning("⚠️ " + g["recommendation"])
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Time points", g["n_timepoints"]); m2.metric("Units (ref / test)", f"{g['n_ref_units']} / {g['n_test_units']}")
        m3.metric("Max mean % (ref / test)", f"{g['max_mean_ref']:.1f} / {g['max_mean_test']:.1f}")
        m4.metric("Points > 85 %", g["points_over_85"]); m5.metric("CV limits (20/10 %)", "met" if g["cv_criteria_ok"] else "exceeded")
        fig = _profile_plot(t, g, ref, test)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        df = pd.DataFrame({"Time": t, f"{ref} mean": g["ref_mean"], f"{ref} SD": g["ref_sd"], f"{ref} CV%": g["ref_cv"],
                           f"{test} mean": g["test_mean"], f"{test} SD": g["test_sd"], f"{test} CV%": g["test_cv"]})
        st.dataframe(df.round(2), use_container_width=True, hide_index=True)
        st.caption("Plateau = three successive time points differing by <5 % (abs). High variability = SD >8 % at any point "
                   "(→ bootstrap f2, lower 90 % CI ≥46). f2 pre-conditions: ≥3 points, ≤1 point >85 %, CV ≤20 % (first) / ≤10 % (others), 12 units.")

    # ── Tab 2: model-dependent ────────────────────────────────────────────────
    with tab_dep:
        verdict = dep.get("similar")
        if verdict is True:
            st.success(f"✅ **SIMILAR** — upper {int(round((1 - res['alpha']) * 100))} % CR bound of the distance "
                       f"({dep['upper']:.2f}) ≤ similarity limit ({dep['limit']:.2f}).")
        elif verdict is False:
            st.error(f"❌ **NOT SIMILAR** — upper CR bound ({dep['upper']:.2f}) > similarity limit ({dep['limit']:.2f}).")
        else:
            st.warning("Similarity limit could not be established.")
        for w in dep.get("warnings", []):
            st.warning("⚠️ " + w)
        st.markdown(f"**Model:** {dep['model_label']} &nbsp;·&nbsp; `{dep['equation']}` &nbsp;·&nbsp; "
                    f"parameters {', '.join(dep['param_names'])} "
                    f"({'ln-transformed: ' + ', '.join(n for n, l in zip(dep['param_names'], dep['ln_transformed']) if l) if any(dep['ln_transformed']) else 'linear scale'})")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Mahalanobis D (MSD)", _fmt(dep["D"], 3)); k2.metric("Hotelling T²", _fmt(dep["T2"], 2))
        k3.metric(f"F({dep['p']}, {dep['df2']})", _fmt(dep["F"], 3)); k4.metric("K (scaling)", _fmt(dep["K"], 3))
        k5.metric("CR of distance", f"[{_fmt(dep['lower'], 2)}, {_fmt(dep['upper'], 2)}]"); k6.metric("Limit (Max MSD)", _fmt(dep["limit"], 2))
        st.caption(f"p-value (H₀: equal mean parameter vectors) = {_fmt(dep['p_value'], 4)} · "
                   f"similarity limit: {dep['limit_note']} · cond(S_p) = {_fmt(dep['cond'], 1)}")
        fig = _param_space_plot(dep, ref, test, alpha=float(res.get('alpha', 0.10)))
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Reading the plot: the amber ellipse is the confidence region of the true parameter difference; "
                       "the green dotted ellipse is the similarity limit (same Mahalanobis metric). Similar ⇔ the amber "
                       "ellipse lies inside the green one. The white × (origin) inside the amber ellipse ⇔ no significant difference.")

    # ── Tab 3: model-independent MSD ──────────────────────────────────────────
    with tab_ind:
        if ind["similar"]:
            st.success(f"✅ **SIMILAR (MSD, Tsong 1996)** — upper CR bound {ind['upper']:.2f} ≤ limit {ind['limit']:.2f} "
                       f"(D_g for a {ind['delta']:g} % shift at every time point).")
        else:
            st.error(f"❌ **NOT SIMILAR (MSD)** — upper CR bound {ind['upper']:.2f} > limit {ind['limit']:.2f}.")
        j1, j2, j3, j4, j5 = st.columns(5)
        j1.metric("Mahalanobis D", _fmt(ind["D"], 3)); j2.metric("Hotelling T²", _fmt(ind["T2"], 2))
        j3.metric("CR of distance", f"[{_fmt(ind['lower'], 2)}, {_fmt(ind['upper'], 2)}]")
        j4.metric("Limit D_g", _fmt(ind["limit"], 2)); j5.metric("p-value", _fmt(ind["p_value"], 4))
        if ind["limit"] and ind["D"] and ind["limit"] / max(ind["D"], 1e-9) > 20:
            st.warning(f"⚠️ The limit is {ind['limit'] / max(ind['D'], 1e-9):.0f}× the observed distance — at this variability "
                       "the MSD criterion has essentially no discriminating power (Paixão et al., EJPB 2017). FDA's Dissolution "
                       "Branch recommends the bootstrap f2 CI for high-variability data.")
        st.caption("Vectors of % dissolved at the selected time points (one per unit); pooled covariance; "
                   "D_g = √(δ' S_p⁻¹ δ) with δ = (delta, …, delta).")

    # ── Tab 4: tables ─────────────────────────────────────────────────────────
    with tab_tbl:
        names_p = dep["param_names"]
        def _tbl(rows, label):
            df = pd.DataFrame([{"Unit": r["unit"], **{n: r["params"][i] for i, n in enumerate(names_p)}, "RMSE": r["rmse"], "fit ok": r["ok"]} for r in rows])
            st.markdown(f"**{label} — unit parameters**")
            st.dataframe(df.round(4), use_container_width=True, hide_index=True)
        _tbl(dep["ref_units"], ref); _tbl(dep["test_units"], test)
        st.markdown("**Mean parameters (natural scale)**")
        st.dataframe(pd.DataFrame({"Parameter": names_p, ref: dep["ref_mean_params"], test: dep["test_mean_params"]}).round(4),
                     use_container_width=True, hide_index=True)
        q_lab = [("ln " if l else "") + n for n, l in zip(names_p, dep["ln_transformed"])]
        st.markdown("**Pooled variance–covariance matrix (transformed scale)**")
        st.dataframe(pd.DataFrame(dep["pooled_cov"], index=q_lab, columns=q_lab).round(5), use_container_width=True)
        st.markdown("**Inverse pooled matrix**")
        st.dataframe(pd.DataFrame(dep["pooled_cov_inv"], index=q_lab, columns=q_lab).round(4), use_container_width=True)
        summary = pd.DataFrame({
            "Statistic": ["p (parameters)", "n reference", "n test", "K (scaling factor)", f"F(p, n1+n2−p−1; {int(round((1-res['alpha'])*100))} %)",
                          "Hotelling T²", "Mahalanobis distance (MSD)", "Lower CR of MSD", "Upper CR of MSD", "Max MSD (similarity limit)",
                          "Upper CR ≤ Max MSD", "Similarity of R and T"],
            "Value": [str(dep["p"]), str(dep["n1"]), str(dep["n2"]), _fmt(dep["K"], 4), _fmt(dep["F"], 4), _fmt(dep["T2"], 4), _fmt(dep["D"], 4),
                      _fmt(dep["lower"], 4), _fmt(dep["upper"], 4), _fmt(dep["limit"], 4),
                      "Yes" if dep.get("similar") else "No", "Accept" if dep.get("similar") else "Reject"],
        })
        st.markdown("**Summary (DDSolver-comparable layout)**")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        csv = summary.to_csv(index=False).encode()
        st.download_button("⬇ Download summary (CSV)", csv, "model_dependent_summary.csv", "text/csv", key="md_dl")

    show_literature("model_dependent")
