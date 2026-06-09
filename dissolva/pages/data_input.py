"""DissolvA page module: Data Input. Extracted from app.py (Phase 3b modularization)."""
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
    q_time = cfg["q_time"]
    q_limit = cfg["q_limit"]
    st.header("Data Input")

    # ── Project Metadata Panel ─────────────────────────────────────────────────
    with st.expander("📁 Project Setup", expanded=not bool(st.session_state.profiles)):
        pm = st.session_state.project_metadata
        pc1, pc2 = st.columns([1, 1])
        with pc1:
            new_proj_name = st.text_input("Project Name", value=pm.get("name","Untitled Project"),
                                          key="proj_name_input")
            new_analyst   = st.text_input("Analyst / Author", value=pm.get("analyst",""),
                                          key="proj_analyst_input",
                                          placeholder="e.g. Dr. Jane Smith")
        with pc2:
            new_desc = st.text_area("Project Description", value=pm.get("description",""),
                                    key="proj_desc_input", height=88,
                                    placeholder="e.g. Bioequivalence study of ibuprofen 400mg tablets...")
        if st.button("💾 Save Project Info", key="save_proj_meta"):
            import datetime as _dt
            st.session_state.project_metadata.update({
                "name":        new_proj_name,
                "analyst":     new_analyst,
                "description": new_desc,
                "created":     st.session_state.project_metadata.get("created") or
                               _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            st.success(f"Project '{new_proj_name}' saved.")


    st.markdown(
        "<div class='info-banner'>"
        "<strong>How it works:</strong> Upload a single Excel file with one or more sheets. "
        "Each sheet = one formulation (e.g. <em>Reference</em>, <em>Test1</em>, <em>Test2</em>). "
        "Columns: first column = <strong>Time</strong>, remaining columns = one per vessel/tablet (6 or 12 per USP/FDA). "
        "The system computes Mean, SD, RSD, CV automatically and builds the dissolution profile with error bars."
        "</div>",
        unsafe_allow_html=True
    )

    input_mode = st.radio(
        "Input Mode",
        ["Excel / CSV Upload (Raw Vessel Data)", "Inline Spreadsheet Editor", "Manual Mean Entry"],
        horizontal=True
    )

    # =========================================================================
    if input_mode == "Excel / CSV Upload (Raw Vessel Data)":
    # =========================================================================

        col_up, col_fmt = st.columns([2, 1])
        with col_up:
            up = st.file_uploader(
                "Upload Excel (.xlsx) or CSV file",
                type=["xlsx", "xls", "csv"],
                key="raw_upload"
            )
        with col_fmt:
            st.markdown(
                "<div class='info-banner' style='font-size:0.82rem;'>"
                "<strong>Expected column layout:</strong><br>"
                "<code>Time | Tablet1 | Tablet2 | ... | Tablet6</code><br>"
                "Time column: any name containing 'time', 'zaman', 'min', 'hour'<br>"
                "Vessel columns: 6 or 12 tablets per USP/FDA<br>"
                "Sheet name = formulation name (e.g. Reference, Test1)"
                "</div>",
                unsafe_allow_html=True
            )

        if up:
            fname = up.name.lower()

            # -- Safe numeric cell parser — NEVER fabricates a value -----------
            # Vessel cells must be numbers. Anything else (a date that Excel
            # mis-inferred, or stray text) becomes NaN. The old code converted
            # such cells to a fake "day.month" number, silently corrupting real
            # dissolution data — removed. We coerce to NaN and warn instead.
            def _to_num(val):
                if val is None or isinstance(val, bool):
                    return np.nan
                if isinstance(val, (int, float)):
                    return float(val)
                try:
                    return float(str(val).strip())
                except Exception:
                    try:  # tolerate European decimal comma ("12,5" -> 12.5)
                        return float(str(val).strip().replace(",", "."))
                    except Exception:
                        return np.nan

            # Single shared, SAFE time-column keyword list (no bare "t"/"min"
            # that would mis-match a vessel column named e.g. "T").
            TIME_KEYWORDS = ["time", "zaman", "minute", "minutes", "saat",
                             "hour", "hours", "dakika", "sure", "sures", "süre"]

            # -- Load all sheets -----------------------------------------------
            try:
                if fname.endswith(".csv"):
                    raw_sheets = {"Sheet1": pd.read_csv(up)}
                else:
                    raw_sheets = pd.read_excel(up, sheet_name=None, engine="openpyxl")

                sheet_names = list(raw_sheets.keys())
                st.success(f"File loaded: **{up.name}** - {len(sheet_names)} sheet(s): {', '.join(sheet_names)}")

                selected_sheets = st.multiselect(
                    "Select which sheets to import as dissolution profiles:",
                    sheet_names,
                    default=sheet_names,
                    help="Each sheet will become a separate dissolution profile."
                )

                # -- Preview ---------------------------------------------------
                if selected_sheets:
                    prev_sh = st.selectbox("Preview sheet:", selected_sheets, key="prev_sel")
                    df_prev = raw_sheets[prev_sh].copy()
                    df_prev.columns = [str(c).strip() for c in df_prev.columns]

                    # Detect time col for preview (shared TIME_KEYWORDS)
                    tc_prev = next((c for c in df_prev.columns
                                    if c.lower().strip() in TIME_KEYWORDS), df_prev.columns[0])
                    vc_prev = [c for c in df_prev.columns if c != tc_prev]

                    # Parse vessel cells safely (non-numeric -> NaN, never fabricated)
                    for col in vc_prev:
                        df_prev[col] = df_prev[col].apply(_to_num)
                    df_prev[tc_prev] = pd.to_numeric(df_prev[tc_prev], errors="coerce")

                    st.markdown(f"**Raw data - {prev_sh}** ({len(vc_prev)} vessels)")
                    st.dataframe(df_prev.style.format(precision=2), use_container_width=True)

                    # Quick stats preview
                    df_prev_clean = df_prev.dropna(subset=[tc_prev])
                    t_p = df_prev_clean[tc_prev].values.astype(float)
                    vd  = df_prev_clean[vc_prev].apply(pd.to_numeric, errors="coerce")
                    mean_p = vd.mean(axis=1).values
                    sd_p   = vd.std(axis=1, ddof=1).values if len(vc_prev) > 1 else np.zeros(len(t_p))
                    rsd_p  = np.where(mean_p > 0, sd_p / mean_p * 100, 0.0)

                    df_stats_prev = pd.DataFrame({
                        f"Time ({time_unit})": t_p,
                        "Mean (%)":  mean_p.round(2),
                        "SD":        sd_p.round(2),
                        "RSD (%)":   rsd_p.round(2),
                        "CV (%)":    rsd_p.round(2),
                        "n vessels": len(vc_prev),
                    })
                    st.markdown("**Computed statistics:**")
                    st.dataframe(df_stats_prev.style.background_gradient(
                        subset=["RSD (%)"], cmap="RdYlGn_r"), use_container_width=True)

                # -- Import button ---------------------------------------------
                if selected_sheets and st.button("Import Selected Sheets as Profiles", type="primary"):
                    imported = []
                    warnings_list = []
                    for sh in selected_sheets:
                        df_raw = raw_sheets[sh].copy()
                        df_raw.columns = [str(c).strip() for c in df_raw.columns]

                        time_col = next((c for c in df_raw.columns
                                         if c.lower().strip() in TIME_KEYWORDS),
                                        df_raw.columns[0])
                        vessel_cols = [c for c in df_raw.columns if c != time_col]

                        # Guard: a sheet with no vessel columns can't be a profile
                        if not vessel_cols:
                            warnings_list.append(f"Sheet '{sh}': no vessel columns found — skipped.")
                            continue

                        # Parse vessel cells safely; count cells that fail (non-numeric)
                        _orig_nonnull = int(df_raw[vessel_cols].notna().sum().sum())
                        for col in vessel_cols:
                            df_raw[col] = df_raw[col].apply(_to_num)
                        _bad_cells = _orig_nonnull - int(df_raw[vessel_cols].notna().sum().sum())
                        if _bad_cells > 0:
                            warnings_list.append(
                                f"Sheet '{sh}': {_bad_cells} cell(s) could not be read as numbers and were "
                                f"treated as missing. Format dissolution values as Number (not Date/Text) in Excel."
                            )

                        df_raw[time_col] = pd.to_numeric(df_raw[time_col], errors="coerce")
                        df_raw = df_raw.dropna(subset=[time_col])
                        # Sort by time and drop duplicate time points (keep first)
                        df_raw = df_raw.sort_values(time_col).drop_duplicates(subset=[time_col], keep="first")
                        if df_raw.empty:
                            warnings_list.append(f"Sheet '{sh}': no valid time points — skipped.")
                            continue

                        t_vals = df_raw[time_col].values.astype(float)
                        vdata  = df_raw[vessel_cols].apply(pd.to_numeric, errors="coerce")
                        n_v    = vdata.shape[1]

                        # A dissolution profile needs ≥2 time points (fitting / f2 /
                        # bootstrap all require it); skip otherwise to avoid downstream errors.
                        if len(t_vals) < 2:
                            warnings_list.append(f"Sheet '{sh}': needs at least 2 time points — skipped.")
                            continue

                        mean_r = vdata.mean(axis=1).values
                        sd_r   = vdata.std(axis=1, ddof=1).values if n_v > 1 else np.zeros(len(t_vals))
                        rsd_r  = np.where(mean_r > 0, sd_r / mean_r * 100, 0.0)

                        # Dissolution should be (roughly) non-decreasing; flag clear reversals.
                        if mean_r.size >= 2 and np.any(np.diff(mean_r) < -2.0):
                            warnings_list.append(
                                f"Sheet '{sh}': mean release decreases at one or more points "
                                f"(non-monotonic) — please verify the data."
                            )

                        if n_v not in [6, 12]:
                            warnings_list.append(
                                f"Sheet '{sh}': {n_v} vessels found (USP/FDA recommends 6 or 12)."
                            )

                        st.session_state.profiles[sh] = {
                            "time":    t_vals.tolist(),
                            "release": mean_r.tolist(),
                            "sd":      sd_r.tolist(),
                            "rsd":     rsd_r.tolist(),
                            "cv":      rsd_r.tolist(),
                            "n":       n_v,
                            "vessels": vessel_cols,
                            "raw":     vdata.values.tolist(),
                        }
                        imported.append(sh)

                    st.success(f"Imported {len(imported)} profile(s): {', '.join(imported)}")
                    for w in warnings_list:
                        st.warning(w)

            except Exception as e:
                st.error(f"Error reading file: {e}")

    # =========================================================================
    elif input_mode == "Inline Spreadsheet Editor":
    # =========================================================================
        st.markdown(
            "<div class='info-banner'>"
            "<strong>Paste-friendly Spreadsheet Editor</strong><br>"
            "Copy your data directly from Excel or any spreadsheet and paste it below. "
            "Format: first column = Time, remaining columns = Tablet 1, Tablet 2, ... "
            "You can also type values manually. Columns are separated by tabs (as in Excel)."
            "</div>",
            unsafe_allow_html=True
        )

        ed_col1, ed_col2 = st.columns([2, 1])
        with ed_col1:
            ed_pname = st.text_input("Profile Name", "Formulation A", key="ed_pname")
        with ed_col2:
            n_tablets_hint = st.number_input(
                "Expected Vessels (hint)", min_value=1, max_value=24,
                value=6, step=1, key="ed_ntab",
                help="Helps generate the template below. Actual count is detected from pasted data."
            )

        # -- Generate template text --------------------------------------------
        default_times = [0, 5, 15, 30, 45, 60, 90, 120]
        header_row = "Time\t" + "\t".join(
            [f"Tablet {i+1}" for i in range(int(n_tablets_hint))]
        )
        data_rows = []
        for t in default_times:
            data_rows.append(str(t) + "\t" + "\t".join(["0.0"] * int(n_tablets_hint)))
        template_text = header_row + "\n" + "\n".join(data_rows)

        st.markdown(
            "<p style='font-weight:600;margin-top:12px;'>"
            "Paste your Excel data here (or edit the template):</p>",
            unsafe_allow_html=True
        )

        pasted = st.text_area(
            "Data (tab-separated, first row = header)",
            value=template_text,
            height=280,
            key="paste_area",
            help="Select all cells in Excel, press Ctrl+C, then click here and press Ctrl+V."
        )

        st.caption(
            "How to paste from Excel: Select the cells in Excel (including the header row) "
            "-> Ctrl+C -> click inside the text box above -> Ctrl+A to select all -> Ctrl+V to paste."
        )

        # -- Parse and preview -------------------------------------------------
        if pasted.strip():
            try:
                import io as _io
                df_pasted = pd.read_csv(
                    _io.StringIO(pasted.strip()),
                    sep="\t",
                    decimal=".",
                    dtype=str
                )
                # Try comma-separated if tab didn't work well
                if df_pasted.shape[1] == 1:
                    df_pasted = pd.read_csv(
                        _io.StringIO(pasted.strip()),
                        sep=",",
                        decimal=".",
                        dtype=str
                    )
                # Try semicolon
                if df_pasted.shape[1] == 1:
                    df_pasted = pd.read_csv(
                        _io.StringIO(pasted.strip()),
                        sep=";",
                        decimal=",",
                        dtype=str
                    )

                df_pasted.columns = [str(c).strip() for c in df_pasted.columns]
                time_col = df_pasted.columns[0]
                vessel_cols = list(df_pasted.columns[1:])

                # Convert to numeric
                df_pasted[time_col] = pd.to_numeric(
                    df_pasted[time_col].str.replace(",", "."), errors="coerce"
                )
                for col in vessel_cols:
                    df_pasted[col] = pd.to_numeric(
                        df_pasted[col].str.replace(",", "."), errors="coerce"
                    )
                df_pasted = df_pasted.dropna(subset=[time_col])

                st.markdown(
                    f"**Preview** - {df_pasted.shape[0]} time points, "
                    f"{len(vessel_cols)} vessels detected:"
                )
                st.dataframe(
                    df_pasted.style.format(precision=2, na_rep="-"),
                    use_container_width=True
                )

                if st.button("Compute & Add Profile", type="primary", key="ed_compute"):
                    t_vals = df_pasted[time_col].values.astype(float)
                    vdata  = df_pasted[vessel_cols].apply(
                        pd.to_numeric, errors="coerce"
                    )
                    n_v    = vdata.shape[1]
                    mean_r = vdata.mean(axis=1).values
                    sd_r   = (vdata.std(axis=1, ddof=1).values
                              if n_v > 1 else np.zeros(len(t_vals)))
                    rsd_r  = np.where(mean_r > 0, sd_r / mean_r * 100, 0.0)

                    st.session_state.profiles[ed_pname] = {
                        "time":    t_vals.tolist(),
                        "release": mean_r.tolist(),
                        "sd":      sd_r.tolist(),
                        "rsd":     rsd_r.tolist(),
                        "cv":      rsd_r.tolist(),
                        "n":       n_v,
                        "vessels": vessel_cols,
                        "raw":     vdata.values.tolist(),
                    }
                    st.success(
                        f"Profile '{ed_pname}' added - "
                        f"{n_v} vessels, {len(t_vals)} time points, "
                        f"max release = {mean_r.max():.1f}%"
                    )
                    df_stat = pd.DataFrame({
                        f"Time ({time_unit})": t_vals,
                        "Mean (%)":  mean_r.round(2),
                        "SD":        sd_r.round(3),
                        "RSD (%)":   rsd_r.round(2),
                        "CV (%)":    rsd_r.round(2),
                    })
                    st.dataframe(df_stat, use_container_width=True)

            except Exception as e:
                st.warning(
                    f"Could not parse the pasted data: {e}. "
                    "Make sure the first row is a header and values are "
                    "separated by tabs (Excel default) or commas."
                )

    # =========================================================================
    else:  # Manual Mean Entry
    # =========================================================================
        st.markdown(
            "<div class='info-banner'>"
            "Enter the already-computed mean cumulative release (%) at each time point."
            "</div>",
            unsafe_allow_html=True
        )
        c1, c2 = st.columns(2)
        with c1:
            t_str = st.text_area("Time points (comma-separated)",
                                 "0,15,30,45,60,90,120,180,240", height=100)
        with c2:
            r_str = st.text_area("Mean Cumulative Release % (comma-separated)",
                                 "0,18,35,49,62,74,82,89,94", height=100)
        pname = st.text_input("Profile Name", "Formulation A")
        if st.button("Add Profile"):
            try:
                ta = np.array([float(x.strip()) for x in t_str.split(",")])
                ra = np.array([float(x.strip()) for x in r_str.split(",")])
                if len(ta) != len(ra):
                    st.error("Arrays must have equal length.")
                else:
                    st.session_state.profiles[pname] = {
                        "time": ta.tolist(), "release": ra.tolist(),
                        "sd": None, "rsd": None, "cv": None, "n": 1, "vessels": []
                    }
                    st.success(f"Profile '{pname}' added.")
            except Exception as e:
                st.error(f"Error: {e}")

    # =========================================================================
    # LOADED PROFILES OVERVIEW (both modes)
    # =========================================================================
    if st.session_state.profiles:
        st.markdown("---")
        st.subheader("Loaded Dissolution Profiles")

        # -- Profile Renaming ─────────────────────────────────────────
        with st.expander("✏️ Edit / Rename Profiles", expanded=False):
            st.markdown("Rename a profile — all analysis results will be synchronized automatically.")
            for nm in list(st.session_state.profiles.keys()):
                r_col1, r_col2, r_col3 = st.columns([2, 2, 1])
                with r_col1:
                    st.markdown(f"**Current:** `{nm}`")
                with r_col2:
                    new_nm = st.text_input(
                        "New name", value=nm,
                        key=f"rename_{nm}",
                        label_visibility="collapsed"
                    )
                with r_col3:
                    if st.button("Rename", key=f"btn_rename_{nm}"):
                        if new_nm.strip() and new_nm != nm:
                            if _rename_profile(nm, new_nm.strip()):
                                st.success(f"'{nm}' → '{new_nm.strip()}'")
                                st.rerun()
                            else:
                                st.error("Name already exists or invalid.")

        # -- Summary cards ──────────────────────────────────────────────────────
        n_profiles = len(st.session_state.profiles)
        card_cols = st.columns(min(n_profiles, 4))
        for i, (nm, d) in enumerate(st.session_state.profiles.items()):
            with card_cols[i % 4]:
                n_v = d.get("n", 1)
                max_r = max(d["release"])
                st.markdown(
                    f"<div style='background:white;border:1px solid #ddd;"
                    f"border-left:4px solid {PALETTE[i%len(PALETTE)]};border-radius:4px;"
                    f"padding:10px;margin:4px 0;'>"
                    f"<strong>{nm}</strong><br>"
                    f"<span style='font-size:0.82rem;color:#555;'>"
                    f"n = {n_v} vessels &nbsp;|&nbsp; {len(d['time'])} time points<br>"
                    f"Max release: {max_r:.1f}%</span></div>",
                    unsafe_allow_html=True
                )

        # -- Mean profiles plot with error bars -------------------------------
        st.markdown("#### Mean Dissolution Profiles")

        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
        with opt_col1:
            show_80 = st.radio(
                f"Q Value ({q_limit:.0f}%)",
                ["Show", "Hide"], horizontal=True, key="opt_80line",
                help=f"FDA/USP Q criterion: NLT {q_limit:.0f}% dissolved at {q_time:.0f} {time_unit}."
            ) == "Show"
        with opt_col2:
            show_qt = st.radio(
                f"Q Time ({q_time:.0f} {time_unit})",
                ["Show", "Hide"], horizontal=True, key="opt_qtline",
                help=f"Vertical marker at Q time point ({q_time:.0f} {time_unit})."
            ) == "Show"
        with opt_col3:
            show_band = st.radio(
                "SD Band",
                ["Show", "Hide"], horizontal=True, key="opt_sdband",
                help="Shaded area around mean = Mean ± SD."
            ) == "Show"
        with opt_col4:
            show_errbar = st.radio(
                "Error Bars (SD)",
                ["Show", "Hide"], horizontal=True, key="opt_errbar",
                help="Vertical error bars at each time point showing SD."
            ) == "Show"

        # ── Compliance Engine: compute OOS ────────────────────────────────────
        def compute_compliance(t_arr, r_arr, q_t, q_lim):
            """Calculate release at Q-time via interpolation."""
            t_arr = np.array(t_arr, dtype=float)
            r_arr = np.array(r_arr, dtype=float)
            if q_t <= t_arr.min():
                return float(r_arr[0]), r_arr[0] >= q_lim
            if q_t >= t_arr.max():
                return float(r_arr[-1]), r_arr[-1] >= q_lim
            rel_at_qt = float(np.interp(q_t, t_arr, r_arr))
            return rel_at_qt, rel_at_qt >= q_lim

        compliance_results = {}
        for nm, d in st.session_state.profiles.items():
            ta = np.array(d["time"]); ra = np.array(d["release"])
            rel_actual, passed = compute_compliance(ta, ra, q_time, q_limit)
            compliance_results[nm] = {"rel": rel_actual, "passed": passed}

        # ── Compliance Badge row ───────────────────────────────────────────
        st.markdown("##### 🏛️ Monograph Compliance Status")
        badge_cols = st.columns(len(compliance_results) or 1)
        for i, (nm, res) in enumerate(compliance_results.items()):
            with badge_cols[i % len(badge_cols)]:
                if res["passed"]:
                    st.markdown(
                        f'<div style="background:#c6efce;border:2px solid #27ae60;'
                        f'border-radius:6px;padding:10px 14px;text-align:center;">'
                        f'<div style="font-size:1.1rem;font-weight:700;color:#1a5c2e;">✅ COMPLIANT</div>'
                        f'<div style="font-size:0.8rem;color:#1a5c2e;">{nm}</div>'
                        f'<div style="font-size:0.75rem;color:#2d7d46;">@ {q_time:.0f} {time_unit}: '
                        f'<strong>{res["rel"]:.1f}%</strong> ≥ Q={q_limit:.0f}%</div>'
                        f'</div>', unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="background:#ffc7ce;border:2px solid #e74c3c;'
                        f'border-radius:6px;padding:10px 14px;text-align:center;">'
                        f'<div style="font-size:1.1rem;font-weight:700;color:#7b1a1a;">⚠️ OOS</div>'
                        f'<div style="font-size:0.8rem;color:#7b1a1a;">{nm}</div>'
                        f'<div style="font-size:0.75rem;color:#a93226;">@ {q_time:.0f} {time_unit}: '
                        f'<strong>{res["rel"]:.1f}%</strong> < Q={q_limit:.0f}%</div>'
                        f'</div>', unsafe_allow_html=True
                    )
        st.markdown("")  # spacer

        fig, ax = plt.subplots(figsize=(10, 5))
        style_ax(fig, ax)

        for i, (nm, d) in enumerate(st.session_state.profiles.items()):
            t   = np.array(d["time"])
            r   = np.array(d["release"])
            sd  = np.array(d["sd"]) if d.get("sd") is not None else np.zeros(len(t))
            col = PALETTE[i % len(PALETTE)]
            has_sd = not np.all(sd == 0)

            if has_sd and show_errbar:
                ax.errorbar(t, r, yerr=sd, fmt="o-", color=col, lw=2,
                            ms=5, capsize=4, capthick=1.5, elinewidth=1.2,
                            alpha=0.9, label=f"{nm} (n={d.get('n',1)})")
            else:
                ax.plot(t, r, "o-", color=col, lw=2, ms=5,
                        label=f"{nm} (n={d.get('n',1)})" if d.get("n",1)>1 else nm)

            if has_sd and show_band:
                ax.fill_between(t, r - sd, r + sd, color=col, alpha=0.10)

        # Mark the critical region with a red shade in OOS case
        any_fail = any(not v["passed"] for v in compliance_results.values())
        if any_fail and show_80 and show_qt:
            # Region left of Q-time line and below Q-limit = critical target region
            ax.axvspan(0, q_time, ymin=0, ymax=q_limit/112,
                       alpha=0.07, color="#e74c3c", zorder=0)
            ax.text(q_time * 0.5, q_limit * 0.5,
                    "⚠️ CRITICAL\nTARGET ZONE",
                    ha='center', va='center', fontsize=8.5,
                    color='#7b1a1a', alpha=0.95,
                    fontweight='bold', style='italic', zorder=4,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#e74c3c', alpha=0.55, lw=0.8))

        if show_80:
            ax.axhline(q_limit, color=AMBER, lw=1.5, ls="--", alpha=0.9,
                       label=f"Q = {q_limit:.0f}% (FDA/USP)")
        if show_qt:
            ax.axvline(q_time, color="#27ae60", lw=1.4, ls=":", alpha=0.85,
                       label=f"Q-time = {q_time:.0f} {time_unit}")

        # ── Internal Spec line ─────────────────────────────────────────────
        _is_cfg = st.session_state.method_cfg
        if _is_cfg.get("internal_spec_enabled", False):
            _isn = _is_cfg.get("internal_spec_name", "Internal Spec")
            _isl = float(_is_cfg.get("internal_spec_limit", 85.0))
            _ist = float(_is_cfg.get("internal_spec_time", 45.0))
            ax.axhline(_isl, color="#9467bd", lw=1.4, ls=(0,(5,3)), alpha=0.85,
                       label=f"{_isn} = {_isl:.0f}%")
            ax.axvline(_ist, color="#9467bd", lw=1.2, ls=(0,(3,3)), alpha=0.75,
                       label=f"{_isn} t = {_ist:.0f} {time_unit}")

            # Internal Spec warning — per profile
            _is_warn_profs = []
            for _nm, _d in st.session_state.profiles.items():
                _t_arr = np.array(_d["time"])
                _r_arr = np.array(_d["release"])
                _is_idx = np.where(np.isclose(_t_arr, _ist))[0]
                if len(_is_idx) > 0:
                    _is_val = _r_arr[_is_idx[0]]
                    if _is_val < _isl:
                        _is_warn_profs.append(f"{_nm} ({_is_val:.1f}% < {_isl:.0f}%)")
            if _is_warn_profs:
                st.warning(
                    f"⚠️ **{_isn} — Below Internal Limit:**  \n"
                    + "\n".join([f"- {p}" for p in _is_warn_profs])
                    + f"\n\n*Advisory only — not a regulatory (FDA/USP) finding.*"
                )

        # OOS/COMPLIANT annotation in the chart corner
        if any_fail:
            ax.text(0.98, 0.97, "⚠️ OOS / NON-COMPLIANT",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=10, fontweight='bold', color='#c0392b',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffc7ce',
                              edgecolor='#e74c3c', alpha=0.92))
        else:
            ax.text(0.98, 0.97, "✅ MONOGRAPH COMPLIANT",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=10, fontweight='bold', color='#1a5c2e',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#c6efce',
                              edgecolor='#27ae60', alpha=0.92))

        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel("Cumulative Drug Released (%)")
        ax.set_title("Mean Dissolution Profiles  (Mean ± SD)")
        t_all = np.concatenate([np.array(d["time"]) for d in st.session_state.profiles.values()])
        ax.set_xlim(left=0, right=t_all.max() * 1.05)
        ax.set_ylim(bottom=0, top=112)
        ax.legend(fontsize=8.5)
        st.pyplot(fig)
        plt.close()

        if show_80 or show_qt:
            st.caption(
                f"Q = {q_limit:.0f}% at {q_time:.0f} {time_unit} | "
                "FDA Guidance for Industry: Dissolution Testing of Immediate Release "
                "Solid Oral Dosage Forms (1997); USP <711> Dissolution."
            )

        # ── OOS Academic Warning Panel ─────────────────────────────────────────
        for nm, res in compliance_results.items():
            if not res["passed"]:
                deficit = q_limit - res["rel"]
                st.error(
                    f"**⚠️ OUT OF SPECIFICATION (OOS) — {nm}**\n\n"
                    f"**Analysis Summary:** *{nm}* formulation released **{res['rel']:.2f}%** "
                    f"at **{q_time:.0f} {time_unit}**, which is **{deficit:.2f}%** below "
                    f"the specified pharmacopeial limit (**Q = {q_limit:.0f}%**).\n\n"
                    f"**Possible Root Causes (Academic Assessment):**\n"
                    f"- 🔬 **Slow dissolution rate:** The active ingredient's dissolution kinetics "
                    f"do not match the expected profile.\n"
                    f"- 💊 **Formulation components:** Disintegrant efficiency, lubricant ratio, "
                    f"or binder concentration should be reviewed.\n"
                    f"- 🌡️ **Stability / aging effect:** Matrix hardening or polymorphic conversion "
                    f"due to long-term storage should be evaluated.\n"
                    f"- ⚗️ **Analytical parameters:** pH, temperature, and dissolution medium "
                    f"composition should be compared against method requirements.\n\n"
                    f"**Recommendation:** Initiate an OOS investigation per USP <711> / ICH Q6A "
                    f"and re-fit the formulation using Korsmeyer–Peppas or Weibull model "
                    f"for release characterization."
                )
            else:
                st.success(
                    f"**✅ MONOGRAPH COMPLIANT — {nm}**  \n"
                    f"*{nm}* formulation released **{res['rel']:.2f}%** at {q_time:.0f} {time_unit}, "
                    f"meeting the **Q = {q_limit:.0f}%** specification. (USP <711> / FDA 1997)"
                )

        # Per-Profile Statistics -> moved to Statistical Analysis page

        if st.button("🗑️ Remove All Profiles", key="clear_profiles_btn",
                     help="Remove profiles but keep project metadata and settings."):
            st.session_state.profiles = {}
            st.session_state.fit_results = {}
            st.session_state.selected_ref_id = None
            st.session_state.selected_test_id = None
            st.session_state.bootstrap_results = None
            st.rerun()

    show_literature("Data Input")

# ===========================================================================
# PAGE: KINETIC MODEL FITTING
# ===========================================================================
