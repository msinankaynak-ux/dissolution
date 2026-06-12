"""DissolvA Academy page: educational content for the 62 dissolution kinetic models.
Category -> model selection -> a detail "page" per model (equation, mechanism,
parameter interpretation, when-to-use, literature). Called from app.py via render().
"""
import streamlit as st
from dissolva.theme import OXFORD, AMBER
from dissolva.models import MODEL_DEFS, CATEGORIES
from dissolva.academy_content import ACADEMY
from dissolva.methods_content import METHODS, METHOD_ORDER
from dissolva import explorer
from dissolva import i18n
import os


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

    if explorer.is_explorable(name):
        st.markdown("#### 🎛️ Interactive explorer")
        st.caption("Drag the sliders — the release curve, t₅₀, t₈₀ and final % update live. "
                   "Use **Pin curve** to freeze a shape and compare. Standard published equation; "
                   "no data leaves your browser.")
        explorer.render(preselect=name)
    else:
        st.markdown("#### 🎛️ Interactive explorer")
        st.caption("An interactive slider demo is not yet available for this model — the equation "
                   "above is the reference form.")

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
    st.header(i18n.t("🎓 DissolvA Academy"))
    _t_models, _t_methods = st.tabs([i18n.t("📐 Kinetic Models"), i18n.t("🧫 Dissolution Methods")])
    with _t_models:
        _render_kinetic()
    with _t_methods:
        _render_methods()


def _render_kinetic():
    st.markdown(
        "The **mechanism, parameter interpretation, applications and literature** for all "
        "62 dissolution kinetic models. Select a category and a model — each model opens on "
        "its own detail page. A summary of every model is provided at the bottom."
    )

    grouped = _by_category()
    cats = [c for c in CATEGORIES if grouped.get(c)]
    col1, col2 = st.columns(2)
    with col1:
        cat = st.selectbox(i18n.t("Category"), cats, key="acad_cat",
                           format_func=lambda c: f"{c}  ({len(grouped.get(c, []))})")
    with col2:
        model = st.selectbox(i18n.t("Model"), grouped.get(cat, []), key="acad_model")

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


# ===========================================================================
# Dissolution Methods chapter
# ===========================================================================
def _render_methods():
    st.markdown(
        "In vitro **dissolution / drug-release methods** — what each apparatus is, the "
        "**dosage forms** it suits, an **apparatus schematic**, and key **literature**. "
        "Select a method to open its page."
    )
    keys = [k for k in METHOD_ORDER if k in METHODS]
    sel = st.selectbox(i18n.t("Method"), keys, key="acad_method",
                       format_func=lambda k: METHODS[k]["name"])
    st.markdown("---")
    if sel:
        _method_page(sel)


def _method_page(key):
    m = METHODS[key]
    st.markdown(
        f'<div style="border-left:4px solid {AMBER};background:rgba(255,204,0,0.06);'
        f'padding:14px 18px;border-radius:6px;margin-bottom:14px;">'
        f'<div style="font-size:1.5rem;font-weight:700;color:#FFFFFF;">{m["name"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### 🧪 Suitable dosage forms")
        st.write(m.get("dosage_forms", "—"))
        st.markdown("#### ⚙️ Typical conditions")
        st.write(m.get("conditions", "—"))
        st.markdown("#### 🔬 How it works")
        st.write(m.get("description", "—"))
    with col2:
        _method_figure(key, m)
    st.markdown("#### 📚 Key literature")
    for r in m.get("refs", []):
        st.markdown(f"- {r}")


def _method_figure(key, m):
    """Show a licensed image override if present (assets/methods/<key>.<ext>),
    else the built-in original SVG schematic."""
    base = os.path.join(os.path.dirname(__file__), "..", "assets", "methods")
    for ext in ("png", "jpg", "jpeg", "webp"):
        fp = os.path.join(base, f"{key}.{ext}")
        if os.path.exists(fp):
            try:
                st.image(fp, use_container_width=True,
                         caption=(m.get("figure_credit") or None))
                return
            except Exception:
                break
    svg_fp = os.path.join(base, f"{key}.svg")
    if os.path.exists(svg_fp):
        try:
            with open(svg_fp, encoding="utf-8") as fh:
                _svg = fh.read()
            st.markdown(f'<div style="background:#16203F;border-radius:10px;padding:10px;">{_svg}</div>',
                        unsafe_allow_html=True)
            if m.get("figure_credit"):
                st.caption(m["figure_credit"])
            return
        except Exception:
            pass
    st.markdown(
        f'<div style="background:#16203F;border:0.5px solid rgba(255,255,255,0.08);'
        f'border-radius:10px;padding:10px;">{m.get("svg","")}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Original schematic — DissolvA. Replace with a licensed image at assets/methods/{key}.png")
