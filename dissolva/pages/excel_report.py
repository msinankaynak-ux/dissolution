"""DissolvA page module: Excel Report. Extracted from app.py (Phase 3b modularization)."""
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
from dissolva.models import (MODEL_DEFS, CATEGORIES, compute_mdt,
    compute_de, r2s, r2adj, aic_fn, msc_fn, _nz)
from dissolva.state import (current_tier, require_tier, _safe_profile_names,
    _get_index, _rename_profile, _clear_all)
from dissolva.content import show_literature, show_all_references, analyze_profile_shape
from dissolva import extras


def render():
    cfg = st.session_state.method_cfg
    time_unit = cfg["time_unit"]
    conc_unit = cfg["conc_unit"]
    dose_mg = cfg["dose_mg"]
    q_time = cfg["q_time"]
    q_limit = cfg["q_limit"]
    import datetime
    st.markdown(
        '<h2 style="color:#FFFFFF;margin:0 0 4px;">Excel Report</h2>'
        '<p style="color:#9fb0d0;font-size:0.88rem;margin:0 0 20px;">'
        'Customize your report, add your logo, select sheets, and download a professional Excel file.</p>',
        unsafe_allow_html=True
    )
    if not st.session_state.profiles:
        st.warning("No profiles loaded. Please go to Data Input first.")
        st.stop()

    # ── Publication exports (PDF report + 300 dpi figure) ───────────────────
    with st.expander("📑 Publication exports — PDF report & 300 dpi figure", expanded=False):
        st.caption("Journal-ready outputs built from your current profiles "
                   "(and model ranking, if a fit has been run).")
        pe1, pe2 = st.columns(2)
        with pe1:
            if st.button("Prepare PDF report", use_container_width=True):
                try:
                    st.session_state["_pdf_bytes"] = extras.build_pdf_report()
                except Exception as e:
                    st.session_state["_pdf_bytes"] = None
                    st.error(f"Could not build the PDF: {e}")
            if st.session_state.get("_pdf_bytes"):
                st.download_button("⬇️ Download PDF report",
                    st.session_state["_pdf_bytes"], "DissolvA_report.pdf",
                    "application/pdf", use_container_width=True)
        with pe2:
            if st.button("Prepare figure (300 dpi)", use_container_width=True):
                try:
                    st.session_state["_png_bytes"] = extras.build_overlay_png(300)
                except Exception as e:
                    st.session_state["_png_bytes"] = None
                    st.error(f"Could not build the figure: {e}")
            if st.session_state.get("_png_bytes"):
                st.download_button("⬇️ Download figure (PNG, 300 dpi)",
                    st.session_state["_png_bytes"], "DissolvA_profiles_300dpi.png",
                    "image/png", use_container_width=True)

    # ── Customization ───────────────────────────────────────────────────────
    # Get defaults from project metadata
    _pm = st.session_state.project_metadata
    _default_title  = f"DissolvA — {_pm.get('name','Dissolution Analysis Report')}"
    _default_author = _pm.get("analyst","DissolvA Team")

    with st.expander("🎨 Report Customization", expanded=True):
        rc1, rc2 = st.columns([1.2, 0.8])
        with rc1:
            report_title  = st.text_input("Report Title", _default_title)
            report_author = st.text_input("Author / Institution", _default_author)
            report_email  = st.text_input("Contact E-mail", "dissolva.app@gmail.com")
            if _pm.get("description"):
                st.caption(f"Project: {_pm['description'][:80]}")
        with rc2:
            st.markdown("**📎 Logo Upload** *(optional)*")
            st.markdown(
                '<div style="background:#16203F;border:1px solid rgba(255,255,255,0.10);border-radius:6px;'
                'padding:8px 12px;font-size:0.78rem;color:#9fb0d0;margin-bottom:8px;">'
                '<b style="color:#CBD5E1;">Requirements:</b><br>'
                '• Format: PNG or JPG<br>'
                '• Recommended size: <b>300 × 100 px</b> (landscape)<br>'
                '• Max file size: 2 MB<br>'
                '• Transparent background preferred<br>'
                '• Will appear top-left of Cover sheet'
                '</div>',
                unsafe_allow_html=True
            )
            logo_file = st.file_uploader("Upload logo", type=["png","jpg","jpeg"], key="report_logo", label_visibility="collapsed")

        st.markdown("**📋 Select Sheets to Include:**")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            inc_cover    = st.checkbox("📄 Cover Page",           value=True)
            inc_method   = st.checkbox("⚗️ Method Report",        value=True)
        with sc2:
            inc_profiles = st.checkbox("📈 Dissolution Profiles",  value=True)
            inc_stats    = st.checkbox("📊 Statistics",            value=True)
        with sc3:
            inc_fitting  = st.checkbox("🔢 Model Fitting",         value=True)
            inc_chart    = st.checkbox("📉 Dissolution Chart",     value=True)
        with sc4:
            inc_f2       = st.checkbox("⚖️ Similarity (f1/f2)",   value=True)
            inc_bootstrap= st.checkbox("🔁 Bootstrap f2",          value=bool(st.session_state.get("bootstrap_results")))

    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    if st.button("⬇️ Generate & Download Excel Report", type="primary"):
        import xlsxwriter
        buf = io.BytesIO()
        wb  = xlsxwriter.Workbook(buf, {"in_memory": True})

        fmt_t  = wb.add_format({"bold":True,"font_size":14,"font_color":"#002147","bottom":2,"bottom_color":"#FFBF00"})
        fmt_h  = wb.add_format({"bold":True,"bg_color":"#002147","font_color":"#FFBF00","border":1,"align":"center"})
        fmt_d  = wb.add_format({"border":1,"num_format":"0.0000","align":"center"})
        fmt_p  = wb.add_format({"border":1,"align":"center"})
        fmt_g  = wb.add_format({"bg_color":"#c6efce","border":1,"num_format":"0.000","align":"center"})
        fmt_b  = wb.add_format({"bg_color":"#ffc7ce","border":1,"num_format":"0.000","align":"center"})
        fmt_n  = wb.add_format({"italic":True,"font_color":"#5a6480","font_size":9})
        fmt_s  = wb.add_format({"bold":True,"bg_color":"#FFD966","font_color":"#002147","border":1})
        fmt_mh = wb.add_format({"bold":True,"bg_color":"#002147","font_color":"#FFBF00","border":1,"font_size":11})
        fmt_ml = wb.add_format({"font_color":"#002147","border":1,"font_size":10})
        fmt_mv = wb.add_format({"border":1,"font_size":10})
        fmt_pass = wb.add_format({"bold":True,"bg_color":"#c6efce","border":1,"align":"center"})
        fmt_fail = wb.add_format({"bold":True,"bg_color":"#ffc7ce","border":1,"align":"center"})

        # 1. COVER
        if inc_cover:
            ws = wb.add_worksheet("Cover")
            ws.set_column("A:A", 55); ws.set_column("B:B", 30)
            if logo_file is not None:
                try:
                    logo_io = io.BytesIO(logo_file.read())
                    ws.insert_image("B1", "logo", {"image_data": logo_io, "x_scale":0.5, "y_scale":0.5, "x_offset":5, "y_offset":5, "object_position":1})
                except Exception:
                    pass
            ws.write("A1", report_title, wb.add_format({"bold":True,"font_size":16,"font_color":"#002147","bottom":2,"bottom_color":"#FFBF00"}))
            ws.write("A2", "Professional Dissolution Analysis Report", fmt_s)
            ws.write("A3", f"Generated: {report_date}", fmt_n)
            ws.write("A4", f"Author: {report_author}", fmt_n)
            ws.write("A5", f"Contact: {report_email}", fmt_n)

        # 1b. ACTIVE SUBSTANCE (PubChem + BCS + FDA) SHEET
        _as_xl = st.session_state.get("active_substance", {})
        if _as_xl.get("fetch_done") and _as_xl.get("pubchem"):
            _pc_xl  = _as_xl["pubchem"]
            _bcs_xl = _as_xl.get("bcs_class") or {}
            _fda_xl = _as_xl.get("fda_methods", [])
            _sel_xl = _as_xl.get("selected_method")

            ws_api = wb.add_worksheet("Active Substance")
            ws_api.set_column("A:A", 35)
            ws_api.set_column("B:B", 55)

            fmt_api_t = wb.add_format({"bold":True,"font_size":13,"font_color":"#002147",
                                        "bottom":2,"bottom_color":"#FFBF00"})
            fmt_api_h = wb.add_format({"bold":True,"bg_color":"#002147","font_color":"#FFBF00","border":1})
            fmt_api_k = wb.add_format({"bold":True,"bg_color":"#dbeafe","font_color":"#002147","border":1})
            fmt_api_v = wb.add_format({"border":1,"font_color":"#1a202c"})
            fmt_api_bcs = wb.add_format({"bold":True,"bg_color":"#002147","font_color":"#FFBF00",
                                          "border":1,"font_size":11,"align":"center"})

            # Title
            ws_api.write("A1", f"Active Substance: {_as_xl['name']}", fmt_api_t)
            ws_api.write("A2", "Source: PubChem (NIH) + FDA Dissolution Methods Database", fmt_n)
            row = 3

            # PubChem chemical properties
            ws_api.write(row, 0, "PHYSICOCHEMICAL PROPERTIES", fmt_api_h)
            ws_api.write(row, 1, "Value", fmt_api_h)
            row += 1

            pc_fields = [
                ("Molecular Formula",    _pc_xl.get("formula","")),
                ("Molecular Weight (g/mol)", _pc_xl.get("mw","")),
                ("Exact Mass",           _pc_xl.get("exact_mass","")),
                ("XLogP (Lipophilicity)",   _pc_xl.get("xlogp","N/A")),
                ("TPSA (Ų)",             _pc_xl.get("tpsa","")),
                ("H-Bond Donors",        _pc_xl.get("hbd","")),
                ("H-Bond Acceptors",     _pc_xl.get("hba","")),
                ("Rotatable Bonds",      _pc_xl.get("rot_bonds","")),
                ("Charge",               _pc_xl.get("charge","")),
                ("CAS Number",           _pc_xl.get("cas","N/A")),
                ("PubChem CID",          _pc_xl.get("cid","")),
                ("IUPAC Name",           str(_pc_xl.get("name",""))[:80]),
                ("Synonyms",             ", ".join(_pc_xl.get("synonyms",[])[:5])),
            ]
            for k, v in pc_fields:
                ws_api.write(row, 0, k, fmt_api_k)
                ws_api.write(row, 1, str(v), fmt_api_v)
                row += 1

            # Experimental data
            if _pc_xl.get("exp"):
                row += 1
                ws_api.write(row, 0, "EXPERIMENTAL DATA (PubChem)", fmt_api_h)
                ws_api.write(row, 1, "", fmt_api_h)
                row += 1
                for ek, ev in _pc_xl["exp"].items():
                    ws_api.write(row, 0, ek, fmt_api_k)
                    ws_api.write(row, 1, str(ev)[:100], fmt_api_v)
                    row += 1

            # BCS class - requires experimental measurement
            row += 1
            ws_api.write(row, 0, "BCS CLASSIFICATION", fmt_api_h)
            ws_api.write(row, 1, "", fmt_api_h)
            row += 1
            ws_api.write(row, 0, "BCS Class", fmt_api_k)
            ws_api.write(row, 1, "Requires experimental solubility and permeability measurement.", fmt_api_v)
            row += 1
            ws_api.write(row, 0, "Reference", fmt_api_k)
            ws_api.write(row, 1, "FDA Guidance for Industry: Waiver of In Vivo Bioavailability (2000); Amidon et al. Pharm Res 1995; ICH M9 (2019)", fmt_api_v)
            row += 1

            # Selected FDA method
            if _sel_xl is not None and _sel_xl < len(_fda_xl):
                _fm_xl = _fda_xl[_sel_xl]
                row += 1
                ws_api.write(row, 0, "SELECTED FDA DISSOLUTION METHOD", fmt_api_h)
                ws_api.write(row, 1, "", fmt_api_h)
                row += 1
                for fk, fv in [
                    ("Drug Name",     _fm_xl.get("drug_name","")),
                    ("Dosage Form",   _fm_xl.get("dosage_form","")),
                    ("Apparatus",     _fm_xl.get("apparatus","")),
                    ("Speed (rpm)",   _fm_xl.get("speed_rpm","")),
                    ("Medium",        _fm_xl.get("medium","")),
                    ("Volume (mL)",   _fm_xl.get("volume_ml","")),
                    ("Sampling Times",_fm_xl.get("sampling_times","")),
                    ("Date Updated",  _fm_xl.get("date_updated","")),
                ]:
                    ws_api.write(row, 0, fk, fmt_api_k)
                    ws_api.write(row, 1, str(fv), fmt_api_v)
                    row += 1

            # All FDA methods
            if _fda_xl and len(_fda_xl) > 1:
                row += 1
                ws_api.write(row, 0, f"ALL FDA METHODS ({len(_fda_xl)} records)", fmt_api_h)
                ws_api.write(row, 1, "", fmt_api_h)
                row += 1
                _fda_cols = ["Drug Name","Dosage Form","Apparatus","Speed (rpm)","Medium","Volume (mL)","Sampling Times","Date Updated"]
                for ci, col in enumerate(_fda_cols):
                    ws_api.write(row, ci, col, fmt_api_h)
                row += 1
                for _frow in _fda_xl:
                    for ci, key in enumerate(["drug_name","dosage_form","apparatus","speed_rpm","medium","volume_ml","sampling_times","date_updated"]):
                        ws_api.write(row, ci, str(_frow.get(key,"")), fmt_api_v)
                    row += 1
                ws_api.set_column("A:H", 22)
            ws.write("A6", f"Profiles: {len(st.session_state.profiles)}", fmt_p)
            ws.write("A7", "Generated by DissolvA™ v3.0 | Powered by AI | 2025", fmt_n)

        # 2. METHOD REPORT
        if inc_method:
            cfg_r = st.session_state.get("method_cfg", {})
            wsM = wb.add_worksheet("Method Report")
            wsM.set_column("A:A", 36); wsM.set_column("B:B", 50)
            wsM.write("A1", "Method & Parameter Report", fmt_t)
            wsM.write("A2", f"{report_author} | {report_date}", fmt_n)
            mrow = [3]
            def write_section(title, rows_data):
                mrow[0] += 1
                wsM.merge_range(mrow[0], 0, mrow[0], 1, title, fmt_mh)
                mrow[0] += 1
                for label, value in rows_data:
                    wsM.write(mrow[0], 0, label, fmt_ml)
                    wsM.write(mrow[0], 1, str(value) if value is not None else "", fmt_mv)
                    mrow[0] += 1
            write_section("General Parameters", [
                ("Time Unit", cfg_r.get("time_unit","minutes")),
                ("Concentration Unit", cfg_r.get("conc_unit","mg/mL")),
                ("Dose (mg)", cfg_r.get("dose_mg",100.0)),
                ("Q Time Point", f"{cfg_r.get('q_time',45.0)} {cfg_r.get('time_unit','min')}"),
                ("Q Value (USP)", f"NLT {cfg_r.get('q_limit',80.0):.0f}%"),
                ("Regulatory Reference", "USP <711> / FDA Guidance 1997"),
            ])
            medium_str = cfg_r.get("medium","")
            if medium_str == "Other": medium_str = cfg_r.get("medium_custom","")
            surf = cfg_r.get("surfactant","None")
            surf_str = "None"
            if surf and surf != "None":
                if surf == "Other": surf = cfg_r.get("surfactant_custom","")
                surf_str = f"{surf} {cfg_r.get('surfactant_conc',0.0):.2f}%"
            write_section("Dissolution Method", [
                ("Apparatus", cfg_r.get("apparatus","")),
                ("Dissolution Medium", medium_str),
                ("Surfactant", surf_str),
                ("Rotation Speed (rpm)", cfg_r.get("rpm","")),
                ("Medium Volume (mL)", cfg_r.get("volume_ml","")),
                ("Temperature (degC)", cfg_r.get("temp_c","")),
                ("Notes", cfg_r.get("notes","")),
            ])
            analytical = cfg_r.get("analytical","UV-Vis Spectrophotometry")
            if analytical == "UV-Vis Spectrophotometry":
                write_section("Analytical Method", [
                    ("Method", "UV-Vis Spectrophotometry"),
                    ("Lambda max (nm)", cfg_r.get("lambda_max","")),
                    ("Slit Width (nm)", cfg_r.get("slit_nm","")),
                    ("Reference Wavelength", cfg_r.get("ref_wavelength","N/A")),
                ])
            else:
                write_section(f"Analytical Method - {analytical}", [
                    ("Method", analytical),
                    ("Column", cfg_r.get("hplc_column","")),
                    ("Column Temp (degC)", cfg_r.get("hplc_col_temp","")),
                    ("Mobile Phase A", cfg_r.get("hplc_mp_a","")),
                    ("Mobile Phase B", cfg_r.get("hplc_mp_b","")),
                    ("Flow Rate (mL/min)", cfg_r.get("hplc_flow","")),
                    ("Detection (nm)", cfg_r.get("hplc_detection","")),
                    ("Injection Volume (uL)", cfg_r.get("hplc_inj_vol","")),
                    ("Run Time (min)", cfg_r.get("hplc_run_time","")),
                ])

        # 3. DISSOLUTION PROFILES
        if inc_profiles:
            ws2 = wb.add_worksheet("Dissolution Profiles"); col = 0
            for nm, dd in st.session_state.profiles.items():
                ws2.write(0, col, nm, fmt_s)
                ws2.write(1, col, f"Time ({time_unit})", fmt_h)
                ws2.write(1, col+1, "Mean (%)", fmt_h)
                ws2.write(1, col+2, "SD", fmt_h)
                ws2.write(1, col+3, "RSD (%)", fmt_h)
                ws2.set_column(col, col+3, 14)
                sd_a  = dd.get("sd")  or [0.0]*len(dd["time"])
                rsd_a = dd.get("rsd") or [0.0]*len(dd["time"])
                for ri, (ti, rv) in enumerate(zip(dd["time"], dd["release"])):
                    ws2.write(ri+2, col, ti, fmt_d)
                    ws2.write(ri+2, col+1, round(rv,3), fmt_d)
                    ws2.write(ri+2, col+2, round(sd_a[ri],4), fmt_d)
                    ws2.write(ri+2, col+3, round(rsd_a[ri],2), fmt_d)
                col += 5

            # Dissolution Profiles - insert matplotlib chart as PNG
            try:
                import io as _io_dp
                _fig_dp, _ax_dp = plt.subplots(figsize=(8, 4.5))
                _fig_dp.patch.set_facecolor("#FDFAF5")
                _ax_dp.set_facecolor("#F8F4EC")
                _pal = ["#002147","#e6194B","#3cb44b","#4363d8","#f58231","#911eb4"]
                for _ci, (_nm, _dd) in enumerate(st.session_state.profiles.items()):
                    _ta = np.array(_dd["time"]); _ra = np.array(_dd["release"])
                    _sd = np.array(_dd.get("sd") or [0.0]*len(_ta))
                    _col = _pal[_ci % len(_pal)]
                    _ax_dp.plot(_ta, _ra, "o-", color=_col, lw=2, ms=5, label=_nm)
                    if not np.all(_sd == 0):
                        _ax_dp.fill_between(_ta, np.clip(_ra-_sd,0,None), _ra+_sd,
                                            alpha=0.12, color=_col)
                _ax_dp.axhline(q_limit, color="#FFBF00", lw=1.5, ls="--",
                               label=f"Q = {q_limit:.0f}% (FDA/USP)")
                _ax_dp.set_xlim(left=0); _ax_dp.set_ylim(bottom=0, top=112)
                _ax_dp.set_xlabel(f"Time ({time_unit})", fontsize=11)
                _ax_dp.set_ylabel("Cumulative Drug Released (%)", fontsize=11)
                _ax_dp.set_title("Mean Dissolution Profiles (Mean ± SD)", fontsize=12, color="#002147")
                _ax_dp.legend(fontsize=9, framealpha=0.9)
                for _sp in ["top","right"]: _ax_dp.spines[_sp].set_visible(False)
                _buf_dp = _io_dp.BytesIO()
                _fig_dp.tight_layout()
                _fig_dp.savefig(_buf_dp, format="png", dpi=130)
                plt.close(_fig_dp)
                _buf_dp.seek(0)
                _img_col = len(st.session_state.profiles) * 5 + 1
                _img_cell = xlsxwriter.utility.xl_col_to_name(_img_col) + "2"
                ws2.insert_image(_img_cell, "profile_chart.png", {
                    "image_data": _buf_dp, "x_scale": 1.0, "y_scale": 1.0,
                    "x_offset": 5, "y_offset": 5, "object_position": 1
                })
            except Exception as _e_dp:
                ws2.write(0, len(st.session_state.profiles)*5+1, f"Chart err: {_e_dp}", fmt_n)

        # 4. STATISTICS
        if inc_stats:
            ws3 = wb.add_worksheet("Statistics")
            ws3.write(0, 0, "Statistical Summary", fmt_t)
            ws3.write(1, 0, f"Generated: {report_date}", fmt_n)
            hdrs = [f"Time ({time_unit})", "Profile", "Mean (%)", "SD", "RSD (%)", "CV (%)", "MDT", "DE (%)"]
            for ci, h in enumerate(hdrs): ws3.write(3, ci, h, fmt_h); ws3.set_column(ci, ci, 16)
            ri2 = 4
            for nm, dd in st.session_state.profiles.items():
                ta = np.array(dd["time"]); ra = np.array(dd["release"])
                sd_a  = np.array(dd.get("sd")  or [0.0]*len(ta))
                rsd_a = np.array(dd.get("rsd") or [0.0]*len(ta))
                mdt = compute_mdt(ta, ra); de = compute_de(ta, ra)
                for i in range(len(ta)):
                    ws3.write(ri2,0,ta[i],fmt_d); ws3.write(ri2,1,nm,fmt_p)
                    ws3.write(ri2,2,round(ra[i],3),fmt_d)
                    ws3.write(ri2,3,round(sd_a[i],4),fmt_d)
                    ws3.write(ri2,4,round(rsd_a[i],2),fmt_d)
                    ws3.write(ri2,5,round(rsd_a[i],2),fmt_d)
                    ws3.write(ri2,6,round(mdt,3) if not np.isnan(mdt) else "N/A",fmt_p)
                    ws3.write(ri2,7,round(de,3),fmt_d)
                    ri2 += 1

        # 5. MODEL FITTING
        if inc_fitting:
            ws4 = wb.add_worksheet("Model Fitting")
            ws4.write(0,0,"Kinetic Model Fitting Results",fmt_t)
            ws4.write(1,0,f"Generated: {report_date}",fmt_n)
            fh = ["Model","Category","R2","R2adj","RMSE","AIC","AICc","BIC","MSC","Params","Parameters","Reference"]
            for ci,h in enumerate(fh): ws4.write(3,ci,h,fmt_h)
            ws4.set_column(0,0,26); ws4.set_column(1,1,14); ws4.set_column(10,10,45); ws4.set_column(11,11,30)
            if st.session_state.fit_results:
                # Safe rounder — None / NaN / inf metrics (possible from the backend
                # for marginal fits) become "N/A" instead of crashing round().
                def _xl(x, n):
                    try:
                        xf = float(x)
                        return round(xf, n) if (xf == xf and xf not in (float("inf"), float("-inf"))) else "N/A"
                    except (TypeError, ValueError):
                        return "N/A"
                def _num(x):
                    try:
                        xf = float(x); return xf if xf == xf else None
                    except (TypeError, ValueError):
                        return None
                # AICc ile sırala (küçük=iyi); None'lar sona
                sorted_r = sorted([(k,v) for k,v in st.session_state.fit_results.items() if v.get("success")],
                                  key=lambda x: (_num(x[1].get("aicc")) is None, _num(x[1].get("aicc")) or float("inf")))
                for ri3,(mn,v) in enumerate(sorted_r):
                    row = ri3+4; adj = _num(v.get("r2adj"))
                    ws4.write(row,0,mn,fmt_p); ws4.write(row,1,v.get("category",""),fmt_p)
                    ws4.write(row,2,_xl(v.get("r2"),4),fmt_d)
                    ws4.write(row,3,_xl(v.get("r2adj"),4),fmt_g if (adj is not None and adj>=0.9) else fmt_b)
                    ws4.write(row,4,_xl(v.get("rmse"),3),fmt_d)
                    ws4.write(row,5,_xl(v.get("aic"),2),fmt_d); ws4.write(row,6,_xl(v.get("aicc"),2),fmt_d)
                    ws4.write(row,7,_xl(v.get("bic"),2),fmt_d); ws4.write(row,8,_xl(v.get("msc"),3),fmt_d)
                    ws4.write(row,9,v.get("n_params",0),fmt_p)
                    pstr = "; ".join(f"{k}={pv:.4g}" for k,pv in (v.get("params") or {}).items() if isinstance(pv,(int,float)))
                    ws4.write(row,10,pstr,fmt_p); ws4.write(row,11,v.get("reference",""),fmt_p)
            else:
                ws4.write(4,0,"No fitting results. Run Kinetic Model Fitting first.",fmt_n)

        # 6. DISSOLUTION CHART
        if inc_chart:
            ws5 = wb.add_worksheet("Dissolution Chart")
            ws5.write("A1","Dissolution Profile Chart",fmt_t)
            ws5.write("A2",f"Mean cumulative release (%) vs Time | {report_date}",fmt_n)
            cdr = 4
            ws5.write(cdr,0,f"Time ({time_unit})",fmt_h)
            for ci,nm in enumerate(st.session_state.profiles.keys()):
                ws5.write(cdr,ci+1,nm,fmt_h); ws5.set_column(ci+1,ci+1,14)
            ws5.set_column(0,0,14)
            all_times = sorted(set(t for d in st.session_state.profiles.values() for t in d["time"]))
            for ri,ti in enumerate(all_times):
                ws5.write(cdr+1+ri,0,ti,fmt_d)
                for ci,(nm,dd) in enumerate(st.session_state.profiles.items()):
                    if ti in dd["time"]:
                        idx = dd["time"].index(ti)
                        ws5.write(cdr+1+ri,ci+1,round(dd["release"][idx],3),fmt_d)
            chart = wb.add_chart({"type":"scatter","subtype":"straight_with_markers"})
            colors  = ["#002147","#e6194B","#3cb44b","#4363d8","#f58231","#911eb4"]
            markers_list = ["circle","square","diamond","triangle","x","star"]
            n_rows = len(all_times)
            for ci,nm in enumerate(st.session_state.profiles.keys()):
                chart.add_series({
                    "name":nm,
                    "categories":["Dissolution Chart",cdr+1,0,cdr+n_rows,0],
                    "values":    ["Dissolution Chart",cdr+1,ci+1,cdr+n_rows,ci+1],
                    "line":  {"color":colors[ci%len(colors)],"width":2},
                    "marker":{"type":markers_list[ci%len(markers_list)],"size":7,
                              "fill":{"color":colors[ci%len(colors)]},"border":{"color":colors[ci%len(colors)]}},
                })
            chart.set_title({"name":"Mean Dissolution Profiles"})
            chart.set_x_axis({"name":f"Time ({time_unit})","min":0,"major_gridlines":{"visible":False}})
            chart.set_y_axis({"name":"Cumulative Release (%)","min":0,"max":105,
                             "major_gridlines":{"visible":True,"line":{"color":"#dddddd","dash_type":"dash"}}})
            chart.set_legend({"position":"bottom"})
            chart.set_size({"width":620,"height":400})
            chart.set_chartarea({"border":{"color":"#002147"},"fill":{"color":"#FDFAF5"}})
            chart.set_plotarea({"fill":{"color":"#F8F4EC"}})
            ws5.insert_chart("A10",chart)

        # 7. SIMILARITY REPORT
        if inc_f2:
            ws6 = wb.add_worksheet("Similarity Report")
            ws6.set_column("A:A",28); ws6.set_column("B:H",16)
            ws6.write("A1","f1 and f2 Similarity Analysis",fmt_t)
            ws6.write("A2","FDA Guidance: Dissolution Testing of Immediate Release Solid Oral Dosage Forms, 1997",fmt_n)
            profile_names = list(st.session_state.profiles.keys())
            # Use active selections from session state (profiles selected on f2 page)
            _ss_ref  = st.session_state.get("selected_ref_id")
            _ss_test = st.session_state.get("selected_test_id")
            ref_xl  = _ss_ref  if _ss_ref  in profile_names else (profile_names[0] if profile_names else None)
            test_xl = _ss_test if _ss_test in profile_names else (profile_names[1] if len(profile_names) > 1 else None)
            if ref_xl and test_xl and ref_xl != test_xl:
                t_rx = np.array(st.session_state.profiles[ref_xl]["time"])
                r_rx = np.array(st.session_state.profiles[ref_xl]["release"])
                t_tx = np.array(st.session_state.profiles[test_xl]["time"])
                r_tx = np.array(st.session_state.profiles[test_xl]["release"])
                cm_xl = np.intersect1d(t_rx, t_tx)
                if len(cm_xl) > 0:
                    rr_xl = np.array([r_rx[np.where(t_rx==ti)[0][0]] for ti in cm_xl])
                    rt_xl = np.array([r_tx[np.where(t_tx==ti)[0][0]] for ti in cm_xl])
                    _msk_xl = rr_xl <= 85
                    if np.any(rr_xl > 85): _msk_xl[np.where(rr_xl > 85)[0][0]] = True
                    msk_xl = _msk_xl; rrf_xl = rr_xl[msk_xl]; rtf_xl = rt_xl[msk_xl]
                    if len(rrf_xl) > 0:
                        f1_xl = float(np.sum(np.abs(rrf_xl-rtf_xl))/np.sum(rrf_xl)*100)
                        f2_xl = float(50*np.log10(100/np.sqrt(1+np.mean((rrf_xl-rtf_xl)**2))))
                        ws6.write("A4","Reference Profile",fmt_h); ws6.write("B4",ref_xl,fmt_p)
                        ws6.write("A5","Test Profile",fmt_h); ws6.write("B5",test_xl,fmt_p)
                        ws6.write("A6","Common Time Points",fmt_h); ws6.write("B6",len(cm_xl),fmt_p)
                        ws6.write("A7","Points Used (ref<=85%)",fmt_h); ws6.write("B7",len(rrf_xl),fmt_p)
                        ws6.write("A9","f1 (Difference Factor)",fmt_h)
                        ws6.write("B9",round(f1_xl,3),fmt_pass if f1_xl<=15 else fmt_fail)
                        ws6.write("C9","PASS (<=15)" if f1_xl<=15 else "FAIL (>15)",fmt_pass if f1_xl<=15 else fmt_fail)
                        ws6.write("A10","f2 (Similarity Factor)",fmt_h)
                        ws6.write("B10",round(f2_xl,3),fmt_pass if f2_xl>=50 else fmt_fail)
                        ws6.write("C10","SIMILAR (>=50)" if f2_xl>=50 else "DISSIMILAR (<50)",fmt_pass if f2_xl>=50 else fmt_fail)
                        ws6.write("A11","Max |Delta R| (%)",fmt_h); ws6.write("B11",round(float(np.max(np.abs(rrf_xl-rtf_xl))),3),fmt_p)
                        ws6.write("A13","Formulas",fmt_t)
                        ws6.write("A14","f1 = [SUM|Rt-Tt| / SUM(Rt)] x 100",fmt_n)
                        ws6.write("A15","f2 = 50 x log10(100 / sqrt(1 + (1/n) x SUM(Rt-Tt)^2))",fmt_n)
                        ws6.write("A17","Point-by-Point Comparison",fmt_t)
                        for ci,h in enumerate([f"Time ({time_unit})","Reference (%)","Test (%)","Diff (%)","Used in f2"]):
                            ws6.write(17,ci,h,fmt_h)
                        for ri,ti in enumerate(cm_xl):
                            rval=rr_xl[ri]; tval=rt_xl[ri]
                            ws6.write(18+ri,0,ti,fmt_d); ws6.write(18+ri,1,round(rval,3),fmt_d)
                            ws6.write(18+ri,2,round(tval,3),fmt_d)
                            ws6.write(18+ri,3,round(abs(rval-tval),3),fmt_d)
                            ws6.write(18+ri,4,"Yes" if rval<=85 else "No",fmt_p)
                        n_pts = len(cm_xl)
                        chart_s = wb.add_chart({"type":"scatter","subtype":"straight_with_markers"})
                        chart_s.add_series({"name":ref_xl,
                            "categories":["Similarity Report",18,0,18+n_pts-1,0],
                            "values":    ["Similarity Report",18,1,18+n_pts-1,1],
                            "line":{"color":"#002147","width":2.5},
                            "marker":{"type":"circle","size":8,"fill":{"color":"#002147"},"border":{"color":"#002147"}}})
                        chart_s.add_series({"name":test_xl,
                            "categories":["Similarity Report",18,0,18+n_pts-1,0],
                            "values":    ["Similarity Report",18,2,18+n_pts-1,2],
                            "line":{"color":"#c0392b","width":2.5,"dash_type":"dash"},
                            "marker":{"type":"square","size":8,"fill":{"color":"#c0392b"},"border":{"color":"#c0392b"}}})
                        chart_s.set_title({"name":f"f1={f1_xl:.2f} | f2={f2_xl:.2f} | {ref_xl} vs {test_xl}"})
                        chart_s.set_x_axis({"name":f"Time ({time_unit})","min":0,"major_gridlines":{"visible":False}})
                        chart_s.set_y_axis({"name":"Cumulative Release (%)","min":0,"max":105,
                                           "major_gridlines":{"visible":True,"line":{"color":"#dddddd","dash_type":"dash"}}})
                        chart_s.set_legend({"position":"bottom"})
                        chart_s.set_size({"width":600,"height":380})
                        chart_s.set_chartarea({"border":{"color":"#002147"},"fill":{"color":"#FDFAF5"}})
                        ws6.insert_chart("G4",chart_s)
            else:
                ws6.write("A4","Load at least 2 profiles to generate similarity report.",fmt_n)

        # 8. BOOTSTRAP f2
        if inc_bootstrap and st.session_state.get("bootstrap_results"):
            ws7 = wb.add_worksheet("Bootstrap f2")
            ws7.set_column("A:A", 30); ws7.set_column("B:B", 18)
            ws7.write("A1","Bootstrap f2 Analysis",fmt_t)
            ws7.write("A2","Shah VP et al. Pharm Res. 1998;15(6):889-896 | EMA/CHMP/EWP/QWP/1401/98 Rev.1",fmt_n)
            br = st.session_state["bootstrap_results"]
            f2_obs   = br.get("f2_obs",  0)
            ci_lower = br.get("ci_lower", 0)
            ci_upper = br.get("ci_upper", 0)
            f2_mean  = br.get("f2_mean",  0)
            n_iter   = br.get("n_iter", 5000)
            method   = br.get("method", "Parametric")

            # Karar
            is_similar = ci_lower >= 50
            verdict    = "SIMILAR" if is_similar else "DISSIMILAR"
            vfmt       = fmt_pass if is_similar else fmt_fail

            ws7.write("A4","Method",fmt_h);              ws7.write("B4", method, fmt_p)
            ws7.write("A5","n Iterations",fmt_h);         ws7.write("B5", n_iter, fmt_p)
            ws7.write("A6","f2 (observed)",fmt_h);        ws7.write("B6", round(f2_obs,3), fmt_p)
            ws7.write("A7","Mean Bootstrap f2",fmt_h);    ws7.write("B7", round(f2_mean,3), fmt_p)
            ws7.write("A8","90% CI Lower (5th pct)",fmt_h); ws7.write("B8", round(ci_lower,3), fmt_p)
            ws7.write("A9","90% CI Upper (95th pct)",fmt_h); ws7.write("B9", round(ci_upper,3), fmt_p)
            ws7.write("A10","FDA Criterion (CI Lower >= 50)",fmt_h)
            ws7.write("B10", verdict, vfmt)
            ws7.write("A12","Interpretation",fmt_t)
            if is_similar:
                ws7.write("A13",
                    f"The 90% CI lower bound ({ci_lower:.2f}) is >= 50. "
                    "Profiles are considered SIMILAR per bootstrap f2 criterion. "
                    "(Shah et al. 1998; FDA Guidance 1997)", fmt_n)
            else:
                ws7.write("A13",
                    f"The 90% CI lower bound ({ci_lower:.2f}) is < 50. "
                    "Profiles are NOT similar per bootstrap f2 criterion. "
                    "(Shah et al. 1998; FDA Guidance 1997)", fmt_n)

            # Bootstrap distribution chart (matplotlib PNG -> BytesIO -> insert_image)
            if br.get("f2_dist") is not None:
                try:
                    import io as _io_bs
                    _f2d = np.array(br["f2_dist"])
                    _fig_bs, _ax_bs = plt.subplots(figsize=(7,4))
                    _fig_bs.patch.set_facecolor("#FDFAF5")
                    _ax_bs.set_facecolor("#F8F4EC")
                    _ax_bs.hist(_f2d, bins=60, color="#002147", alpha=0.75, edgecolor="white", linewidth=0.4)
                    _ax_bs.axvline(50, color="#e74c3c", lw=2, ls="--", label="f2 = 50 (FDA limit)")
                    _ax_bs.axvline(ci_lower, color="#FFBF00", lw=1.8, ls=":", label=f"CI Lower = {ci_lower:.2f}")
                    _ax_bs.axvline(f2_obs,   color="#27ae60", lw=1.8, ls="-.", label=f"Observed f2 = {f2_obs:.2f}")
                    _ax_bs.set_xlabel("Bootstrap f2 Value", fontsize=11)
                    _ax_bs.set_ylabel("Frequency", fontsize=11)
                    _ax_bs.set_title(
                        f"Bootstrap f2 Distribution — {verdict}  (CI: {ci_lower:.2f}–{ci_upper:.2f})",
                        fontsize=11, color="#002147"
                    )
                    _ax_bs.legend(fontsize=9)
                    for sp in ["top","right"]: _ax_bs.spines[sp].set_visible(False)
                    _buf_bs = _io_bs.BytesIO()
                    _fig_bs.tight_layout()
                    _fig_bs.savefig(_buf_bs, format="png", dpi=130)
                    plt.close(_fig_bs)
                    _buf_bs.seek(0)
                    ws7.insert_image("D4", "bs_chart.png", {
                        "image_data": _buf_bs, "x_scale": 1.0, "y_scale": 1.0,
                        "x_offset": 5, "y_offset": 5, "object_position": 1
                    })
                except Exception as _e_bs:
                    ws7.write("D4", f"Chart error: {_e_bs}", fmt_n)

        wb.close(); buf.seek(0)
        st.success("✅ Report ready!")
        st.download_button(
            "⬇️ Download Excel Report",
            data=buf.getvalue(),
            file_name=f"DissolvA_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ===========================================================================
# PAGE: API INFORMATION
# ===========================================================================
