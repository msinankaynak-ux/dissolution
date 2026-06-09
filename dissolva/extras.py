"""DissolvA — free-beta value-add features (kept out of app.py to stay tidy).

All user-facing text is English (project language rule). Each function is
defensive: a missing optional dependency or secret degrades gracefully and never
raises, so the app cannot crash because of these extras.

Contents:
- init_sentry()            — error monitoring, no-op without a DSN, no PII.
- load_demo_data()         — one-click Reference + Test dissolution profiles.
- citation_dialog()        — "Cite this tool" (APA + BibTeX), Zenodo-ready.
- consent_banner()         — dismissible GDPR cookie/consent notice.
- build_overlay_png()      — 300 dpi publication figure of the profiles.
- build_pdf_report()       — multi-page PDF (meta + figure + fit ranking).
"""
import io
import streamlit as st


# ---------------------------------------------------------------------------
# Sentry — crash reporting (privacy-safe: no PII, no request bodies)
# ---------------------------------------------------------------------------
_SENTRY_DONE = False


def init_sentry():
    """Initialise Sentry once if a DSN is configured (st.secrets[sentry].dsn or
    SENTRY_DSN env). Silent no-op otherwise. Never sends user/PII data."""
    global _SENTRY_DONE
    if _SENTRY_DONE:
        return
    _SENTRY_DONE = True
    import os
    dsn = ""
    try:
        dsn = (st.secrets.get("sentry") or {}).get("dsn") or ""
    except Exception:
        dsn = ""
    dsn = dsn or os.getenv("SENTRY_DSN", "")
    if not dsn:
        return
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0.0,      # errors only; no perf payloads
            send_default_pii=False,       # never attach user identifiers
            environment=os.getenv("ENVIRONMENT", "production"),
            release="dissolva@3.0",
        )
    except Exception:
        pass  # monitoring must never break the app


# ---------------------------------------------------------------------------
# Demo dataset — instant onboarding
# ---------------------------------------------------------------------------
def load_demo_data():
    """Populate two realistic 6-vessel immediate-release profiles (Reference and
    Test) and select them, so a first-time user can run fitting/f2 immediately."""
    import numpy as np

    time = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0]
    # Per-vessel cumulative release (%). Reference dissolves slightly faster than Test.
    ref_raw = [
        [0, 36, 57, 71, 81, 91, 97, 99],
        [0, 33, 54, 69, 79, 90, 96, 99],
        [0, 38, 59, 73, 83, 92, 98, 100],
        [0, 34, 55, 70, 80, 89, 96, 98],
        [0, 37, 58, 72, 82, 91, 97, 100],
        [0, 35, 56, 70, 80, 90, 97, 99],
    ]
    test_raw = [
        [0, 28, 47, 61, 72, 84, 92, 97],
        [0, 26, 45, 59, 70, 82, 91, 96],
        [0, 30, 49, 63, 74, 85, 93, 98],
        [0, 27, 46, 60, 71, 83, 92, 97],
        [0, 29, 48, 62, 73, 85, 93, 98],
        [0, 25, 44, 58, 69, 81, 90, 96],
    ]

    def _profile(raw):
        arr = np.array(raw, dtype=float)            # vessels x time
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rsd = np.where(mean > 0, 100.0 * sd / mean, 0.0)
        return {
            "time": list(time),
            "release": mean.round(2).tolist(),
            "sd": sd.round(2).tolist(),
            "rsd": rsd.round(2).tolist(),
            "cv": rsd.round(2).tolist(),
            "n": arr.shape[0],
            "vessels": [f"Vessel {i+1}" for i in range(arr.shape[0])],
            "raw": arr.T.tolist(),                  # time x vessels (matches importer)
        }

    st.session_state.profiles = {
        "Reference (demo)": _profile(ref_raw),
        "Test (demo)": _profile(test_raw),
    }
    st.session_state.fit_results = {}
    st.session_state.bootstrap_results = None
    st.session_state.selected_ref_id = "Reference (demo)"
    st.session_state.selected_test_id = "Test (demo)"


# ---------------------------------------------------------------------------
# Cite this tool — APA + BibTeX (Zenodo-ready)
# ---------------------------------------------------------------------------
def _citation_meta():
    """Citation fields, overridable via st.secrets[citation]."""
    meta = {
        "authors": "Kaynak, M. S.",
        "title": "DissolvA: Predictive Dissolution Suite",
        "year": "2026",
        "version": "3.0",
        "url": "https://dissanalyze.streamlit.app",
        "doi": "",  # set after a Zenodo deposit, e.g. 10.5281/zenodo.1234567
    }
    try:
        cfg = st.secrets.get("citation") or {}
        for k in meta:
            if cfg.get(k):
                meta[k] = str(cfg[k])
    except Exception:
        pass
    return meta


@st.dialog("❝ Cite this tool")
def citation_dialog():
    m = _citation_meta()
    doi_apa = f" https://doi.org/{m['doi']}" if m["doi"] else ""
    apa = (f"{m['authors']} ({m['year']}). {m['title']} (Version {m['version']}) "
           f"[Computer software]. {m['url']}.{doi_apa}")
    bib_key = f"dissolva{m['year']}"
    doi_line = f"\n  doi          = {{{m['doi']}}}," if m["doi"] else ""
    bibtex = (
        f"@software{{{bib_key},\n"
        f"  author       = {{{m['authors']}}},\n"
        f"  title        = {{{m['title']}}},\n"
        f"  year         = {{{m['year']}}},\n"
        f"  version      = {{{m['version']}}},\n"
        f"  url          = {{{m['url']}}},{doi_line}\n"
        f"  note         = {{Accessed via dissanalyze.streamlit.app}}\n"
        f"}}"
    )
    st.markdown("If DissolvA supported your work, please cite it. Use the copy "
                "button on each block (top-right on hover).")
    st.caption("APA")
    st.code(apa, language="text")
    st.caption("BibTeX")
    st.code(bibtex, language="bibtex")
    if not m["doi"]:
        st.info("A permanent **DOI** can be minted free on **Zenodo** (link your "
                "GitHub release). Once you have it, add it under `[citation]` in "
                "secrets and it will appear here automatically.")


# ---------------------------------------------------------------------------
# Consent banner — GDPR cookie/usage notice (session-level, dismissible)
# ---------------------------------------------------------------------------
def consent_banner(open_privacy):
    """Render a slim, dismissible consent notice until the user accepts it this
    session. `open_privacy` is a zero-arg callable that opens the privacy dialog.
    Buttons are scoped via their `st-key-*` classes so they stay light and compact
    (the global amber button style would otherwise make them heavy)."""
    if st.session_state.get("cookie_consent"):
        return
    st.markdown("""<style>
    .st-key-consent_bar { background:rgba(0,33,71,0.035) !important;
        border:1px solid rgba(0,33,71,0.10) !important; border-radius:10px !important; }
    .st-key-consent_accept button {
        background:#002147 !important; color:#fff !important;
        border:1px solid rgba(255,191,0,0.55) !important;
        font-family:inherit !important; font-size:0.8rem !important;
        font-weight:500 !important; padding:5px 16px !important;
        border-radius:7px !important; min-height:0 !important; }
    .st-key-consent_accept button:hover {
        background:#FFBF00 !important; color:#002147 !important;
        border-color:#FFBF00 !important; }
    .st-key-consent_details button {
        background:transparent !important; border:none !important;
        color:#5a8ab0 !important; font-family:inherit !important;
        font-size:0.78rem !important; font-weight:400 !important;
        padding:5px 4px !important; min-height:0 !important; }
    .st-key-consent_details button:hover {
        background:transparent !important; color:#FFBF00 !important; }
    </style>""", unsafe_allow_html=True)
    with st.container(border=False, key="consent_bar"):
        c1, c2, c3 = st.columns([0.80, 0.11, 0.11], vertical_alignment="center")
        with c1:
            st.markdown(
                "<div style='font-size:0.82rem;color:#5a6480;line-height:1.45;'>"
                "🍪 We use a sign-in cookie and anonymous usage stats (feature + "
                "country) to run the beta — <b style='color:#3a4660;'>your dissolution "
                "data is never stored.</b></div>", unsafe_allow_html=True)
        with c2:
            if st.button("Details", use_container_width=True, key="consent_details"):
                open_privacy()
        with c3:
            if st.button("Got it", use_container_width=True, key="consent_accept"):
                st.session_state.cookie_consent = True
                st.rerun()


# ---------------------------------------------------------------------------
# Publication exports — 300 dpi figure + PDF report
# ---------------------------------------------------------------------------
def _profiles():
    return st.session_state.get("profiles", {}) or {}


def _time_unit():
    try:
        return st.session_state.method_cfg.get("time_unit", "minutes")
    except Exception:
        return "minutes"


def has_export_data() -> bool:
    return len(_profiles()) > 0


def _overlay_fig():
    """Build a matplotlib overlay figure of all profiles (caller closes it)."""
    import matplotlib.pyplot as plt
    profs = _profiles()
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, (name, d) in enumerate(profs.items()):
        t = d.get("time", []) or []
        r = d.get("release", []) or []
        if not t or not r:
            continue
        sd = d.get("sd") or []
        mk = markers[i % len(markers)]
        if sd and any((s or 0) for s in sd) and len(sd) == len(r):
            ax.errorbar(t, r, yerr=sd, marker=mk, capsize=3, linewidth=1.5,
                        markersize=5, label=name)
        else:
            ax.plot(t, r, marker=mk, linewidth=1.5, markersize=5, label=name)
    ax.set_xlabel(f"Time ({_time_unit()})", fontsize=11)
    ax.set_ylabel("Cumulative release (%)", fontsize=11)
    ax.set_title("Dissolution profiles", fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    if profs:
        ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    return fig


def build_overlay_png(dpi: int = 300) -> bytes:
    """Publication-quality PNG of the dissolution overlay at the given DPI."""
    import matplotlib.pyplot as plt
    fig = _overlay_fig()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_pdf_report() -> bytes:
    """A self-contained multi-page PDF: cover/metadata, overlay figure, and the
    model-ranking table when a fit is present. Uses only matplotlib (no new dep)."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    meta = st.session_state.get("project_metadata", {}) or {}
    profs = _profiles()
    buf = io.BytesIO()

    with PdfPages(buf) as pdf:
        # --- Cover page ---
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.text(0.5, 0.86, "DissolvA", ha="center", fontsize=30, weight="bold",
                 color="#002147")
        fig.text(0.5, 0.82, "Predictive Dissolution Suite — Analysis Report",
                 ha="center", fontsize=12, color="#5a6480")
        lines = [
            f"Project: {meta.get('name') or 'Untitled Project'}",
            f"Analyst: {meta.get('analyst') or '-'}",
            f"Profiles: {', '.join(profs.keys()) if profs else '-'}",
            f"Reference: {st.session_state.get('selected_ref_id') or '-'}",
            f"Test: {st.session_state.get('selected_test_id') or '-'}",
        ]
        y = 0.72
        for ln in lines:
            fig.text(0.12, y, ln, fontsize=11, color="#222")
            y -= 0.035
        fig.text(0.12, 0.10,
                 "Generated by DissolvA (BETA, research use only). "
                 "FDA/EMA guidance-aligned. Dissolution data is not stored.",
                 fontsize=8, color="#8a8a8a", wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Overlay figure page ---
        if profs:
            fig = _overlay_fig()
            pdf.savefig(fig)
            plt.close(fig)

        # --- Model ranking page (only if a fit exists) ---
        fr = st.session_state.get("fit_results") or {}
        ok = {k: v for k, v in fr.items() if isinstance(v, dict) and v.get("success")}
        if ok:
            def _num(x, n):
                try:
                    xf = float(x)
                    return round(xf, n) if xf == xf else None
                except (TypeError, ValueError):
                    return None
            rows = []
            for v in ok.values():
                rows.append([
                    str(v.get("name", ""))[:24],
                    _num(v.get("r2adj"), 4),
                    _num(v.get("rmse"), 3),
                    _num(v.get("aicc"), 2),
                    _num(v.get("bic"), 2),
                ])
            # sort by AICc ascending (None last)
            rows.sort(key=lambda r: (r[3] is None, r[3] if r[3] is not None else 0))
            rows = rows[:20]
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.12, 0.93, "Model ranking (top 20 by AICc)", fontsize=14,
                     weight="bold", color="#002147")
            ax = fig.add_axes([0.08, 0.08, 0.84, 0.80])
            ax.axis("off")
            table = ax.table(
                cellText=[[("" if c is None else c) for c in r] for r in rows],
                colLabels=["Model", "R2adj", "RMSE", "AICc", "BIC"],
                cellLoc="center", loc="upper center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.3)
            pdf.savefig(fig)
            plt.close(fig)

    buf.seek(0)
    return buf.getvalue()
