"""DissolvA page: Template Builder.

Generates a blank, correctly-structured Excel template that matches the Data Input
uploader, so researchers can lay out and fill in their own studies offline.
"""
import streamlit as st
from dissolva.templates import build_blank_template_xlsx, DEFAULT_TIMES_90


def render():
    cfg = st.session_state.get("method_cfg", {}) or {}
    tu = cfg.get("time_unit", "min")

    st.markdown(
        "<h2 style='color:#FFFFFF;margin:0 0 4px;'>Template Builder</h2>"
        "<p style='color:#9fb0d0;margin:0 0 14px;'>Generate a blank, correctly-formatted "
        "Excel template for your own dissolution study. Each formulation becomes one sheet — "
        "fill in the vessel cells, then upload it under Data Input.</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        products_text = st.text_area(
            "Formulations (one per line)",
            value="Reference\nTest 1\nTest 2",
            height=120,
            help="Each line becomes a separate sheet/profile (e.g. Reference, Test 1).",
        )
        n_vessels = st.radio(
            "Vessels per formulation", [6, 12], index=1, horizontal=True,
            help="USP/FDA recommends 6 or 12 vessels.",
        )
    with c2:
        tmode = st.radio("Time points", ["Standard (0–90 min)", "Custom"], index=0)
        if tmode.startswith("Standard"):
            times = list(DEFAULT_TIMES_90)
            st.caption("0, 5, 10, 15, 30, 45, 60, 90")
        else:
            raw = st.text_input(
                "Custom time points (comma-separated)",
                value="0, 10, 20, 30, 45, 60, 90, 120",
            )
            times = []
            for tok in raw.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    times.append(float(tok))
                except ValueError:
                    pass
        time_unit = st.selectbox("Time unit", ["min", "hour"],
                                 index=(1 if tu == "hour" else 0))

    products = [p.strip() for p in products_text.splitlines() if p.strip()]

    problems = []
    if not products:
        problems.append("Add at least one formulation.")
    if len(times) < 2:
        problems.append("Add at least 2 valid time points.")
    if problems:
        for p in problems:
            st.warning(p)
        return

    st.markdown(
        f"**Preview:** {len(products)} sheet(s) &nbsp;·&nbsp; {int(n_vessels)} vessels "
        f"&nbsp;·&nbsp; {len(times)} time points ({time_unit})",
        unsafe_allow_html=True,
    )
    try:
        data = build_blank_template_xlsx(products, int(n_vessels), times, time_unit)
    except Exception as e:
        st.error(f"Could not build template: {e}")
        return

    st.download_button(
        "⬇️ Download Excel template",
        data=data,
        file_name="DissolvA_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
    st.caption("After filling the vessel cells, go to Data Input → "
               "“Excel / CSV Upload (Raw Vessel Data)” and upload this file.")
