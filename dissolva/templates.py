"""Blank Excel template builder for DissolvA Data Input.

Produces an .xlsx whose structure exactly matches what the Data Input uploader
expects (one sheet per formulation; first column = Time; remaining columns =
one per vessel). Shared utility so the Demo-Data feature can reuse it later.
"""
import io
import re

DEFAULT_TIMES_90 = [0, 5, 10, 15, 30, 45, 60, 90]

_INVALID_SHEET = re.compile(r'[:\\/?*\[\]]')


def _safe_sheet_names(products):
    """Excel sheet names: <=31 chars, no []:*?/\\, unique, non-empty."""
    out, seen = [], set()
    for i, p in enumerate(products):
        name = _INVALID_SHEET.sub("-", str(p).strip())[:31] or f"Profile {i+1}"
        base, k = name, 2
        while name.lower() in seen:
            suffix = f" ({k})"
            name = (base[:31 - len(suffix)] + suffix)
            k += 1
        seen.add(name.lower())
        out.append(name)
    return out


def build_blank_template_xlsx(products, n_vessels, times, time_unit="min"):
    """Return .xlsx bytes: one sheet per product, Time col pre-filled, vessels blank."""
    import xlsxwriter
    if not products:
        raise ValueError("At least one formulation is required.")
    if len(times) < 2:
        raise ValueError("At least two time points are required.")
    n_vessels = int(n_vessels)

    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    hdr = wb.add_format({"bold": True, "bg_color": "#0B132B",
                         "font_color": "#FFFFFF", "border": 1, "align": "center"})
    tfmt = wb.add_format({"border": 1, "bold": True, "bg_color": "#F4F6FA"})
    cell = wb.add_format({"border": 1})

    for name in _safe_sheet_names(products):
        ws = wb.add_worksheet(name)
        ws.write(0, 0, f"Time ({time_unit})", hdr)
        for j in range(n_vessels):
            ws.write(0, j + 1, f"Vessel {j + 1}", hdr)
        for i, t in enumerate(times):
            ws.write_number(i + 1, 0, float(t), tfmt)
            for j in range(n_vessels):
                ws.write_blank(i + 1, j + 1, None, cell)
        ws.set_column(0, 0, 12)
        ws.set_column(1, n_vessels, 10)
        ws.freeze_panes(1, 1)
    wb.close()
    buf.seek(0)
    return buf.getvalue()


# ===========================================================================
# Example (demo) dataset — 1 Reference + 2 generics, 12 vessels, up to 90 min.
# Deterministic: identical values every call (reproducible). Shared by the
# in-app "Load example" action and the downloadable example .xlsx.
# ===========================================================================
DEMO_TIMES = [0, 5, 10, 15, 30, 45, 60, 90]
_DEMO_MEANS = {
    "Reference (demo)": [0, 30, 50, 64, 80, 87, 93, 99],
    "Test A (demo)":    [0, 27, 46, 60, 76, 84, 91, 98],   # similar -> f2 pass
    "Test B (demo)":    [0, 20, 35, 47, 64, 73, 82, 92],   # slower  -> OOS at Q
}


def _demo_vessel_matrix(mean, n_vessels=12):
    """Deterministic per-vessel spread around each mean. Returns time x vessel."""
    factors = [0.95 + 0.10 * i / (n_vessels - 1) for i in range(n_vessels)]
    rows = []
    for v in mean:
        if v <= 0:
            rows.append([0.0] * n_vessels)
        else:
            rows.append([min(100.0, round(v * f, 1)) for f in factors])
    return rows


def build_demo_profiles(n_vessels=12):
    """Profile dicts (session_state.profiles shape) for the example dataset."""
    import numpy as np
    out = {}
    for name, mean in _DEMO_MEANS.items():
        mat = np.array(_demo_vessel_matrix(mean, n_vessels), dtype=float)  # time x vessel
        m = mat.mean(axis=1)
        sd = mat.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rsd = np.where(m > 0, 100.0 * sd / m, 0.0)
        out[name] = {
            "time": [float(t) for t in DEMO_TIMES],
            "release": m.round(2).tolist(),
            "sd": sd.round(2).tolist(),
            "rsd": rsd.round(2).tolist(),
            "cv": rsd.round(2).tolist(),
            "n": n_vessels,
            "vessels": [f"Vessel {i+1}" for i in range(n_vessels)],
            "raw": mat.tolist(),  # time x vessel (matches importer)
        }
    return out


def build_demo_xlsx(n_vessels=12, time_unit="min"):
    """Return .xlsx bytes: example raw vessel dataset (one sheet per formulation)."""
    import xlsxwriter
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True})
    hdr = wb.add_format({"bold": True, "bg_color": "#0B132B",
                         "font_color": "#FFFFFF", "border": 1, "align": "center"})
    tfmt = wb.add_format({"border": 1, "bold": True, "bg_color": "#F4F6FA"})
    cell = wb.add_format({"border": 1})
    names = list(_DEMO_MEANS.keys())
    for orig, sheet in zip(names, _safe_sheet_names(names)):
        mat = _demo_vessel_matrix(_DEMO_MEANS[orig], n_vessels)
        ws = wb.add_worksheet(sheet)
        ws.write(0, 0, f"Time ({time_unit})", hdr)
        for j in range(n_vessels):
            ws.write(0, j + 1, f"Vessel {j + 1}", hdr)
        for i, t in enumerate(DEMO_TIMES):
            ws.write_number(i + 1, 0, float(t), tfmt)
            for j in range(n_vessels):
                ws.write_number(i + 1, j + 1, float(mat[i][j]), cell)
        ws.set_column(0, 0, 12)
        ws.set_column(1, n_vessels, 9)
        ws.freeze_panes(1, 1)
    wb.close()
    buf.seek(0)
    return buf.getvalue()
