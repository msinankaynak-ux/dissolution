"""DissolvA Academy page: educational content for the 62 dissolution kinetic models.
Category -> model selection -> a detail "page" per model (equation, mechanism,
parameter interpretation, when-to-use, literature). Called from app.py via render().
"""
import streamlit as st
from dissolva.theme import OXFORD, AMBER
from dissolva.models import MODEL_DEFS, CATEGORIES
from dissolva.academy_content import ACADEMY


def _by_category():
    grouped = {}
    for name, (func, p0, pnames, eq, ref, cat) in MODEL_DEFS.items():
        grouped.setdefault(cat, []).append(name)
    return grouped


def _model_page(name):
    func, p0, pnames, eq, ref, cat = MODEL_DEFS[name]
    c = ACADEMY.get(name, {})
    st.markdown(
        f'<div style="border-left:4px solid {AMBER};background:rgba(255,204,0,0.06);'
        f'padding:14px 18px;border-radius:6px;margin-bottom:14px;">'
        f'<div style="font-size:1.6rem;font-weight:700;color:#FFFFFF;">{name}</div>'
        f'<div style="font-size:0.85rem;color:#9fb0d0;margin-top:2px;">'
        f'Category: <b style="color:#CBD5E1;">{cat}</b> &nbsp;·&nbsp; Reference: {ref}</div></div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**Equation**")
        st.code(eq, language="text")
    with col2:
        st.markdown("**Parameters**")
        st.code(", ".join(pnames), language="text")

    st.markdown("#### 🔬 Mechanism")
    st.write(c.get("mechanism", "—"))
    st.markdown("#### 📈 Parameter & curve interpretation")
    st.write(c.get("interpretation", "—"))
    st.markdown("#### 🎯 When to use")
    st.write(c.get("use", "—"))
    st.markdown("#### 📚 Key literature")
    for r in c.get("refs", []):
        st.markdown(f"- {r}")


def render():
    st.header("🎓 DissolvA Academy — Kinetic Model School")
    st.markdown(
        "The **mechanism, parameter interpretation, applications and literature** for all "
        "62 dissolution kinetic models. Select a category and a model — each model opens on "
        "its own detail page. A summary of every model is provided at the bottom."
    )

    grouped = _by_category()
    cats = [c for c in CATEGORIES if grouped.get(c)]
    col1, col2 = st.columns(2)
    with col1:
        cat = st.selectbox("Category", cats, key="acad_cat",
                           format_func=lambda c: f"{c}  ({len(grouped.get(c, []))})")
    with col2:
        model = st.selectbox("Model", grouped.get(cat, []), key="acad_model")

    st.markdown("---")
    if model:
        _model_page(model)

    st.markdown("---")
    with st.expander("📋 Summary of all 62 models (by category)"):
        for c in cats:
            st.markdown(f"### {c}  ·  {len(grouped.get(c, []))} models")
            for m in grouped.get(c, []):
                _, _, _, eq, ref, _ = MODEL_DEFS[m]
                mech = ACADEMY.get(m, {}).get("mechanism", "")
                short = (mech[:130] + "…") if len(mech) > 130 else mech
                st.markdown(f"- **{m}** — `{eq}`  · _{ref}_  \n  {short}")
