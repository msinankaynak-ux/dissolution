"""DissolvA page module: API Information. Extracted from app.py (Phase 3b modularization)."""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import warnings
import html


def _esc(v) -> str:
    """HTML-escape external content (PubChem/PubMed/scite) before it is rendered
    via unsafe_allow_html — prevents markup/script injection from API responses."""
    return html.escape(str(v if v is not None else ""), quote=True)


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


# ── Page-specific helpers (PubChem/BCS/Lipinski/sink) ──
def _pubchem_fetch(name: str) -> dict:
    """Fetches active substance data from PubChem."""
    try:
        import requests as _req
        base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        # Find CID
        r0 = _req.get(f"{base}/compound/name/{name}/JSON",
                      timeout=10, headers={"User-Agent": "DissolvA/4.0"})
        if r0.status_code != 200:
            return {"error": f"Compound not found: {name}"}
        cid = r0.json()["PC_Compounds"][0]["id"]["id"]["cid"]

        # Basic properties
        props = "MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,TPSA,RotatableBondCount,MolecularFormula,IUPACName,ExactMass,Charge"
        r1 = _req.get(f"{base}/compound/cid/{cid}/property/{props}/JSON",
                      timeout=10, headers={"User-Agent": "DissolvA/4.0"})
        p = r1.json()["PropertyTable"]["Properties"][0]

        # Synonym and CAS
        r2 = _req.get(f"{base}/compound/cid/{cid}/synonyms/JSON",
                      timeout=8, headers={"User-Agent": "DissolvA/4.0"})
        synonyms = r2.json().get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
        cas = next((s for s in synonyms if s.replace("-","").isdigit() and len(s) > 5), "N/A")

        # GHS and hazard
        r3 = _req.get(f"{base}/compound/cid/{cid}/xrefs/RegistryID/JSON",
                      timeout=8, headers={"User-Agent": "DissolvA/4.0"})

        # Physical properties (experimental)
        r4 = _req.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
            "?heading=Experimental+Properties",
            timeout=10, headers={"User-Agent": "DissolvA/4.0"}
        )
        exp_data = {}
        if r4.status_code == 200:
            try:
                sections = r4.json().get("Record", {}).get("Section", [])
                for sec in sections:
                    if "Experimental" in sec.get("TOCHeading", ""):
                        for sub in sec.get("Section", []):
                            heading = sub.get("TOCHeading", "")
                            info = sub.get("Information", [{}])
                            if info:
                                val = info[0].get("Value", {}).get("StringWithMarkup", [{}])
                                if val:
                                    exp_data[heading] = val[0].get("String", "")
            except Exception:
                pass

        return {
            "cid":         cid,
            "name":        p.get("IUPACName", name),
            "formula":     p.get("MolecularFormula", ""),
            "mw":          p.get("MolecularWeight", 0),
            "exact_mass":  p.get("ExactMass", 0),
            "xlogp":       p.get("XLogP", None),
            "hbd":         p.get("HBondDonorCount", 0),
            "hba":         p.get("HBondAcceptorCount", 0),
            "tpsa":        p.get("TPSA", 0),
            "rot_bonds":   p.get("RotatableBondCount", 0),
            "charge":      p.get("Charge", 0),
            "cas":         cas,
            "synonyms":    synonyms[:8],
            "exp":         exp_data,
            "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "img_url":     f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG",
        }
    except Exception as e:
        return {"error": str(e)}


# BCS classification removed - requires experimental measurement (FDA 2005, Amidon 1995)
# BCS color and metadata standard (ICH M9 / FDA Guidance convention)
_BCS_META = {
    "I":   {"bg":"#c6efce","text":"#1a5c2e","border":"#86c99a",
            "desc":"High solubility · High permeability","biowaiver":"Biowaiver candidate"},
    "II":  {"bg":"#fff3cd","text":"#856404","border":"#f0d060",
            "desc":"Low solubility · High permeability","biowaiver":"Dissolution rate-limiting"},
    "III": {"bg":"#dbeafe","text":"#185fa5","border":"#93c5fd",
            "desc":"High solubility · Low permeability","biowaiver":"Permeability rate-limiting"},
    "IV":  {"bg":"#ffc7ce","text":"#9c1c1c","border":"#f09595",
            "desc":"Low solubility · Low permeability","biowaiver":"Difficult formulation"},
}

def _extract_bcs_from_scite(text: str, drug_name: str = "") -> list:
    """
    BCS class parse - supports Roman (III) AND Arabic (3) numerals.
    Scite: "acyclovir was the model BCS class 3 drug"
    PubMed: "Glipizide belongs to BCS class II"
    CRITICAL: if drug_name is given, the text must contain this drug.
    Test: 12/12 cases passed (acyclovir, glipizide, amoxicillin, atorvastatin...).
    """
    import re as _re_bcs
    if not text:
        return []
    if drug_name:
        _dl = drug_name.lower()
        _tl = text.lower()
        _variants = [_dl] + ([_dl[:5]] if len(_dl) > 6 else [])
        if not any(v in _tl for v in _variants):
            return []
    found = set()
    valid_roman = {"I","II","III","IV"}
    arabic_map  = {"1":"I","2":"II","3":"III","4":"IV"}
    patterns = [
        r'BCS\s+[Cc]lass\s+([IVX]{1,3}(?:\s+(?:and|or)\s+[IVX]{1,3})*)',
        r'BCS\s+[Cc]lass\s+([1-4](?:\s+(?:and|or)\s+[1-4])*)',
        r'[Cc]lass\s+([IVX]{1,3}(?:\s+(?:and|or)\s+[IVX]{1,3})*)\s+(?:drug|compound|according|active)',
        r'[Cc]lass\s+([1-4](?:\s+(?:and|or)\s+[1-4])*)\s+(?:drug|compound|according|active)',
        r'BCS\s+([IVX]{1,3})\b',
        r'BCS\s+([1-4])\b',
        r'biopharmaceutic[sa]*\s+classification\s+system\s+(?:class\s+)?([IVX]{1,3}|[1-4])',
        r'biopharmaceutic[sa]*\s+class(?:ification)?\s+(?:system\s+)?(?:class\s+)?([IVX]{1,3}|[1-4])',
    ]
    for p in patterns:
        for m in _re_bcs.finditer(p, text, _re_bcs.IGNORECASE):
            raw = m.group(1)
            for cls in _re_bcs.findall(r'[IVX]{1,3}', raw):
                if cls.upper() in valid_roman:
                    found.add(cls.upper())
            for num in _re_bcs.findall(r'[1-4]', raw):
                if num in arabic_map:
                    found.add(arabic_map[num])
    return sorted(found, key=lambda x: ["I","II","III","IV"].index(x))


def _bcs_badge_html(classes: list, source: str = "") -> str:
    """Returns BCS class badges as HTML."""
    if not classes:
        return (
            '<span style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,191,0,0.3);'
            'border-radius:6px;padding:4px 10px;font-size:10px;font-weight:700;color:#FFBF00;'
            'letter-spacing:0.5px;">BCS CLASS<br>'
            '<span style="font-size:9px;font-weight:400;opacity:0.7;">Requires experimental measurement</span>'
            '</span>'
        )
    badges = []
    for cls in classes:
        m = _BCS_META.get(cls, _BCS_META["IV"])
        src_html = (f'<br><span style="font-size:9px;opacity:0.7;">{source}</span>' if source else "")
        badges.append(
            f'<span style="background:{m["bg"]};color:{m["text"]};border:1px solid {m["border"]};'
            f'border-radius:6px;padding:4px 10px;font-size:11px;font-weight:700;'
            f'display:inline-block;text-align:center;">'
            f'BCS {cls}{src_html}</span>'
        )
    return " ".join(badges)


def _lipinski_check(pc: dict) -> dict:
    """Lipinski Rule of Five check."""
    mw    = float(pc.get("mw", 0))
    xlogp = pc.get("xlogp") or 0
    hbd   = int(pc.get("hbd", 0))
    hba   = int(pc.get("hba", 0))
    rules = {
        "MW ≤ 500":   (mw,    500,  mw <= 500),
        "LogP ≤ 5":   (xlogp, 5,    xlogp <= 5),
        "HBD ≤ 5":    (hbd,   5,    hbd <= 5),
        "HBA ≤ 10":   (hba,   10,   hba <= 10),
    }
    violations = sum(1 for _, _, ok in rules.values() if not ok)
    return {"rules": rules, "violations": violations, "druglike": violations <= 1}


def _sink_condition(pc: dict, volume_ml: float, dose_mg: float) -> dict:
    """
    FDA Sink Condition check.
    Rule: Dose / (Cs x Vd) < 1/3
    Cs (solubility) is estimated from PubChem or via a LogP proxy.
    """
    xlogp = pc.get("xlogp") or 0
    # Rough solubility estimate from LogP (Yalkowsky-equation-like)
    # log S ~ 0.5 - 0.01 x (MP - 25) - LogP (simplified)
    log_s_est = 0.5 - xlogp
    cs_mg_ml  = 10 ** log_s_est  # mg/mL
    ratio     = dose_mg / (cs_mg_ml * volume_ml)
    sink_ok   = ratio < (1/3)
    return {
        "cs_est_mg_ml": round(cs_mg_ml, 4),
        "ratio":        round(ratio, 4),
        "sink_ok":      sink_ok,
        "note":         "LogP-based estimate — validate with experimental Cs",
    }


def render():
    cfg = st.session_state.method_cfg
    q_time = cfg["q_time"]
    q_limit = cfg["q_limit"]
    import re as _re_api2

    # ── PubMed Hybrid Search Function (DissolvA v4.1) ────────────────────────
    # Strategy: MeSH (controlled vocabulary) + Title/Abstract (free-text fallback)
    # Scientific basis:
    #   - MeSH: Librarian-curated semantic precision (~85-95% relevance)
    #   - TIAB: Captures newly published (not yet indexed) articles
    #   - [pt] filter: Excludes editorial/letter/case-report, improves signal
    # Reference: NLM Technical Bulletin (2023), PubMed User Guide
    @st.cache_data(ttl=3600, show_spinner=False)
    def _pubmed_search(drug_name: str, max_results: int = 17) -> tuple:
        try:
            import requests as _req
            base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

            # --- HYBRID QUERY: MeSH + TIAB ---
            hybrid_query = (
                f'('
                f'"{drug_name}"[MeSH Terms] OR "{drug_name}"[Title/Abstract]'
                f') AND ('
                f'"Solubility"[MeSH] OR "Biopharmaceutics"[MeSH] '
                f'OR "Drug Liberation"[MeSH] OR "Permeability"[MeSH] '
                f'OR dissolution[Title/Abstract] '
                f'OR "in vitro release"[Title/Abstract] '
                f'OR bioavailability[Title/Abstract]'
                f') NOT ('
                f'editorial[pt] OR letter[pt] OR "case reports"[pt]'
                f')'
            )

            r1 = _req.get(f"{base}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": hybrid_query,
                    "retmax": max_results,
                    "retmode": "json",
                    "sort": "relevance",
                    "usehistory": "y",
                },
                headers={"User-Agent": "DissolvA/4.1"}, timeout=12)

            _json = r1.json().get("esearchresult", {})
            ids = _json.get("idlist", [])
            total = int(_json.get("count", 0))
            query_translation = _json.get("querytranslation", "")

            # --- FALLBACK: If hybrid returned empty, try pure free-text ---
            if not ids:
                r1b = _req.get(f"{base}/esearch.fcgi",
                    params={"db": "pubmed",
                            "term": f"{drug_name} dissolution",
                            "retmax": max_results, "retmode": "json", "sort": "relevance"},
                    headers={"User-Agent": "DissolvA/4.1"}, timeout=10)
                _jb = r1b.json().get("esearchresult", {})
                ids = _jb.get("idlist", [])
                total = int(_jb.get("count", 0))
                query_translation = "FALLBACK (free-text): " + _jb.get("querytranslation", "")

            if not ids:
                return [], 0, query_translation

            # Metadata
            r2 = _req.get(f"{base}/efetch.fcgi",
                params={"db":"pubmed","id":",".join(ids),"retmode":"xml","rettype":"abstract"},
                headers={"User-Agent":"DissolvA/4.1"}, timeout=15)
            # XML parse
            from xml.etree import ElementTree as ET
            root = ET.fromstring(r2.text)
            articles = []
            for art in root.findall(".//PubmedArticle"):
                try:
                    med = art.find(".//MedlineCitation")
                    pmid = med.findtext(".//PMID","")
                    title = med.findtext(".//ArticleTitle","")
                    # Strip HTML tags
                    title = _re_api2.sub(r'<[^>]+>', '', title)
                    # Year
                    year = (med.findtext(".//PubDate/Year") or
                            med.findtext(".//PubDate/MedlineDate","")[:4])
                    # Journal
                    journal = med.findtext(".//Journal/Title","")
                    # First author
                    authors = med.findall(".//Author")
                    first_auth = ""
                    if authors:
                        ln = authors[0].findtext("LastName","")
                        fn = authors[0].findtext("Initials","")
                        first_auth = f"{ln} {fn}".strip()
                    # DOI
                    doi = ""
                    for eid in art.findall(".//ArticleId"):
                        if eid.get("IdType") == "doi":
                            doi = eid.text or ""
                    # PMC
                    pmc = ""
                    for eid in art.findall(".//ArticleId"):
                        if eid.get("IdType") == "pmc":
                            pmc = eid.text or ""
                    # Abstract
                    ab_texts = med.findall(".//AbstractText")
                    abstract = " ".join(
                        (_re_api2.sub(r'<[^>]+>', '', t.text or "") for t in ab_texts)
                    )[:300]
                    # MeSH terms (for transparency)
                    mesh_terms = [m.findtext("DescriptorName", "")
                                  for m in art.findall(".//MeshHeading")]
                    articles.append({
                        "pmid": pmid, "title": title, "year": year,
                        "journal": journal, "first_author": first_auth,
                        "doi": doi, "pmc": pmc, "abstract": abstract,
                        "mesh": [m for m in mesh_terms if m][:5],
                    })
                except Exception:
                    pass
            return articles, total, query_translation
        except Exception as e:
            return [], 0, f"Error: {e}"

    # ── Page title ─────────────────────────────────────────────────────────
    st.markdown(
        '<h2 style="color:#FFFFFF;margin:0 0 4px;">💊 API Information</h2>'
        '<p style="color:#9fb0d0;font-size:0.87rem;margin:0 0 14px;">'
        'Active-substance physicochemical profile, FDA dissolution methods, and '
        'peer-reviewed literature — on a single screen.</p>',
        unsafe_allow_html=True
    )

    _as = st.session_state.get("active_substance", {})

    # ── Source info banner ────────────────────────────────────────────────
    st.markdown(
        '<div style="background:#16203F;border:0.5px solid rgba(255,255,255,0.08);border-radius:8px;'
        'padding:10px 16px;margin-bottom:14px;font-size:12px;color:#9fb0d0;line-height:1.7;">'
        '<strong>Sources:</strong> &nbsp;'
        '<span style="background:#dbeafe;color:#185fa5;padding:1px 7px;border-radius:10px;'
        'font-size:11px;margin-right:4px;">PubChem NIH</span> physicochemical parameters · '
        '<span style="background:#c6efce;color:#1a5c2e;padding:1px 7px;border-radius:10px;'
        'font-size:11px;margin-right:4px;">FDA Dissolution DB</span> dissolution methods · '
        '<span style="background:#fff3cd;color:#856404;padding:1px 7px;border-radius:10px;'
        'font-size:11px;margin-right:4px;">Scite</span> citation scan for BCS class · '
        '<span style="background:#f0e6ff;color:#6b21a8;padding:1px 7px;border-radius:10px;'
        'font-size:11px;">PubMed NIH</span> dissolution literature</div>',
        unsafe_allow_html=True
    )

    # ── Active substance search box ──────────────────────────────────────────────
    _search_col1, _search_col2 = st.columns([4, 1])
    with _search_col1:
        _api_query = st.text_input(
            "Active substance name",
            value=_as.get("name", ""),
            placeholder="Ibuprofen, Glipizide, Olanzapine, Metformin...",
            key="api_info_search",
            label_visibility="collapsed"
        )
    with _search_col2:
        _api_load_btn = st.button(
            "🔬 Load", key="api_info_load",
            use_container_width=True, type="primary"
        )

    if _api_load_btn and _api_query.strip():
        _substance = _api_query.strip()
        with st.spinner(f"{_substance} — PubChem + FDA + Scite loading..."):
            _pc_new = _pubchem_fetch(_substance)
            try:
                import requests as _req_load
                _fda_s = _req_load.Session()
                _fda_idx_url = "https://www.accessdata.fda.gov/scripts/cder/dissolution/index.cfm"
                _fda_srch_url = "https://www.accessdata.fda.gov/scripts/cder/dissolution/dsp_SearchResults.cfm"
                _fda_h = {"User-Agent":"Mozilla/5.0 Chrome/120.0","Referer":_fda_idx_url,"Accept":"text/html"}
                _fda_s.get(_fda_idx_url, headers=_fda_h, timeout=8)
                _fda_r = _fda_s.post(_fda_srch_url,
                    data={"SearchTerm":_substance,"basic":"1","action":"Search"},
                    headers={**_fda_h,"Content-Type":"application/x-www-form-urlencoded"},
                    timeout=15)
                from bs4 import BeautifulSoup as _BS2
                import re as _re_load
                _soup2 = _BS2(_fda_r.text, "html.parser")
                _tbl2 = _soup2.find("table", id="example") or _soup2.find("table", {"class": lambda c: c and "table" in c})
                _fda_new = []
                if _tbl2:
                    _tbody2 = _tbl2.find("tbody")
                    _rows2 = _tbody2.find_all("tr") if _tbody2 else _tbl2.find_all("tr")[1:]
                    _amap2 = {"I ":"USP I (Basket)","II":"USP II (Paddle)",
                              "III":"USP III (Reciprocating Cylinder)","IV":"USP IV (Flow-Through Cell)"}
                    for _row2 in _rows2:
                        _cols2 = [td.get_text(separator=" ", strip=True) for td in _row2.find_all(["td","th"])]
                        if len(_cols2) < 5: continue
                        _app2 = _cols2[2]
                        for _k2, _v2 in _amap2.items():
                            if _app2.upper().startswith(_k2.upper()): _app2 = _v2; break
                        _fda_new.append({
                            "drug_name":_cols2[0],"dosage_form":_cols2[1],"apparatus":_app2,
                            "speed_rpm":_cols2[3],"medium":_cols2[4],
                            "volume_ml":_cols2[5] if len(_cols2)>5 else "",
                            "sampling_times":_cols2[6] if len(_cols2)>6 else "",
                            "date_updated":_cols2[7] if len(_cols2)>7 else "",
                        })
            except Exception:
                _fda_new = []
            # Parse BCS from PubMed - during loading
            _bcs_classes_new  = []
            _bcs_source_new   = ""
            _bcs_abstract_new = ""
            _bcs_title_new    = ""
            _bcs_doi_new      = ""
            _bcs_journal_new  = ""
            _bcs_snippets_new = []

            # Layer 1: Scite API - collect all matches, sort by citation count
            _bcs_done = False
            _bcs_papers_new = []  # [{title,doi,journal,year,source,abstract,tally,snippets,classes}]
            try:
                import requests as _req_bcs2
                _sc = _req_bcs2.get(
                    "https://api.scite.ai/search/papers",
                    params={"term": f"{_substance} BCS classification",
                            "limit": 15},
                    headers={"User-Agent": "DissolvA/4.0"}, timeout=8
                )
                if _sc.status_code == 200:
                    for _sh in _sc.json().get("hits", []):
                        _txt_sc = (_sh.get("abstract","") or "")
                        _snips_sc = []
                        for _cit in _sh.get("citations", []):
                            _s = _cit.get("snippet","")
                            if _s:
                                _txt_sc += " " + _s
                                _snips_sc.append(_s)
                        _cls_sc = _extract_bcs_from_scite(_txt_sc, _substance)
                        if not _cls_sc:
                            continue
                        _a0 = _sh.get("authors",[{}])
                        _auth_name = _a0[0].get("authorName","").split()[-1] if _a0 else ""
                        _tally_total = _sh.get("tally",{}).get("total",0)
                        _rel_snips = [s for s in _snips_sc
                            if _substance.lower() in s.lower()
                            and any(x in s.lower() for x in ["bcs","biopharmaceutic","class"])]
                        _bcs_papers_new.append({
                            "classes":  _cls_sc,
                            "title":    _sh.get("title",""),
                            "doi":      _sh.get("doi",""),
                            "journal":  _sh.get("journal",""),
                            "year":     str(_sh.get("year","")),
                            "source":   f"{_auth_name} {_sh.get('year','')}".strip(),
                            "abstract": (_sh.get("abstract","") or "")[:400],
                            "tally":    _tally_total,
                            "snippets": _rel_snips[:3],
                        })
                    if _bcs_papers_new:
                        # Sort from most-cited to least
                        _bcs_papers_new.sort(key=lambda x: x["tally"], reverse=True)
                        _first = _bcs_papers_new[0]
                        _bcs_classes_new  = _first["classes"]
                        _bcs_source_new   = _first["source"]
                        _bcs_title_new    = _first["title"]
                        _bcs_abstract_new = _first["abstract"]
                        _bcs_doi_new      = _first["doi"]
                        _bcs_journal_new  = _first["journal"]
                        _bcs_snippets_new = _first["snippets"]
                        _bcs_done = True
            except Exception:
                pass

            # Layer 2: PubMed Hybrid (MeSH + TIAB) - focused on BCS class evidence
            # Strategy #1: MeSH-anchored — "Acyclovir"[MeSH] + Biopharmaceutics[MeSH] + BCS[TIAB]
            # Strategy #2: Pure TIAB - captures new/unindexed articles
            # Runs even if Scite succeeded, enriches the _bcs_papers_new list
            try:
                import requests as _req_bcs3
                import re as _re_bcs3
                from xml.etree import ElementTree as _ET3
                # Hybrid query pairs
                for _pm_term in [
                    # Strategy #1: MeSH-anchored - high precision
                    f'(("{_substance}"[MeSH Terms] OR "{_substance}"[Title/Abstract]) '
                    f'AND ("Biopharmaceutics"[MeSH] OR "Solubility"[MeSH] OR "Permeability"[MeSH]) '
                    f'AND (BCS[Title/Abstract] OR "biopharmaceutic* classif*"[Title/Abstract] '
                    f'OR "class I"[Title/Abstract] OR "class II"[Title/Abstract] '
                    f'OR "class III"[Title/Abstract] OR "class IV"[Title/Abstract])) '
                    f'NOT (editorial[pt] OR letter[pt])',
                    # Strategy #2: TIAB-only - fallback / new articles
                    f'"{_substance}"[Title/Abstract] AND '
                    f'("biopharmaceutics classification system"[Title/Abstract] '
                    f'OR "BCS class"[Title/Abstract] OR "BCS classification"[Title/Abstract])',
                ]:
                    _pm_r = _req_bcs3.get(
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                        params={"db":"pubmed","term":_pm_term,
                                "retmax":"20","retmode":"json","sort":"relevance",
                                "usehistory":"y"},
                        headers={"User-Agent":"DissolvA/4.1"}, timeout=10
                    )
                    _pm_ids = _pm_r.json().get("esearchresult",{}).get("idlist",[])
                    if not _pm_ids:
                        continue
                    _pm_f = _req_bcs3.get(
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                        params={"db":"pubmed","id":",".join(_pm_ids[:20]),
                                "retmode":"xml","rettype":"abstract"},
                        headers={"User-Agent":"DissolvA/4.1"}, timeout=15
                    )
                    _root3 = _ET3.fromstring(_pm_f.text)
                    _existing_dois = {p.get("doi","") for p in _bcs_papers_new}
                    for _art3 in _root3.findall(".//PubmedArticle"):
                        _ab3 = " ".join(t.text or "" for t in _art3.findall(".//AbstractText"))
                        _ti3 = _re_bcs3.sub(r"<[^>]+>","",
                               _art3.findtext(".//ArticleTitle") or "")
                        _cls3 = _extract_bcs_from_scite(_ab3 + " " + _ti3, _substance)
                        if not _cls3:
                            continue
                        _ln3  = _art3.findtext(".//LastName","")
                        _fn3  = _art3.findtext(".//Initials","")
                        _yr3  = (_art3.findtext(".//PubDate/Year") or
                                 _art3.findtext(".//PubDate/MedlineDate","")[:4])
                        _jnl3 = _art3.findtext(".//Journal/Title","")
                        _doi3 = ""
                        for _eid3 in _art3.findall(".//ArticleId"):
                            if _eid3.get("IdType") == "doi":
                                _doi3 = _eid3.text or ""
                        # Prevent duplication
                        if _doi3 and _doi3 in _existing_dois:
                            continue
                        _existing_dois.add(_doi3)
                        # Can't get citation count from PubMed, set 0
                        # Show "PubMed (Hybrid)" label on the tab
                        _bcs_papers_new.append({
                            "classes":  _cls3,
                            "title":    _ti3,
                            "doi":      _doi3,
                            "journal":  _jnl3,
                            "year":     _yr3,
                            "source":   f"{_ln3} {_fn3} {_yr3}".strip(),
                            "abstract": _ab3[:400],
                            "tally":    0,
                            "snippets": [],
                            "origin":   "PubMed (MeSH Hybrid)",
                        })
            except Exception:
                pass

            # Update main values from the papers list
            if _bcs_papers_new:
                # Scite (tally>0) first, PubMed fills the rest
                _bcs_papers_new.sort(key=lambda x: x.get("tally",0), reverse=True)
                _first = _bcs_papers_new[0]
                _bcs_classes_new  = _first["classes"]
                _bcs_source_new   = _first["source"]
                _bcs_title_new    = _first["title"]
                _bcs_abstract_new = _first["abstract"]
                _bcs_doi_new      = _first["doi"]
                _bcs_journal_new  = _first["journal"]
                _bcs_snippets_new = _first.get("snippets",[])

        st.session_state["active_substance"] = {
            "name": _substance, "pubchem": _pc_new, "bcs_class": None,
            "bcs_from_lit": {
                "classes":  _bcs_classes_new,
                "source":   _bcs_source_new,
                "abstract": _bcs_abstract_new,
                "title":    _bcs_title_new,
                "doi":      _bcs_doi_new,
                "journal":  _bcs_journal_new,
                "snippets": _bcs_snippets_new,
                "papers":   _bcs_papers_new,
            },
            "fda_methods": _fda_new,
            "selected_method": None, "fetch_done": True,
        }
        st.rerun()

    # Empty screen if no active substance loaded
    if not _as.get("fetch_done") or not _as.get("name"):
        st.markdown(
            '<div style="text-align:center;padding:40px 20px;color:#9fb0d0;">'
            '<div style="font-size:44px;margin-bottom:12px;">💊</div>'
            '<div style="font-size:16px;font-weight:500;color:#E6ECF8;margin-bottom:6px;">'
            'Enter active substance name and press Load</div>'
            '<div style="font-size:12px;color:#7E8DAB;line-height:1.8;">'
            'PubChem · FDA Dissolution DB · Scite · PubMed</div></div>',
            unsafe_allow_html=True
        )
    else:
        _pc  = _as.get("pubchem") or {}
        _fda = _as.get("fda_methods", [])
        _sel = _as.get("selected_method")

        # ── BCS - read from session state (parsed during loading) ─────
        _bcs_lit     = _as.get("bcs_from_lit") or {}
        _bcs_classes = _bcs_lit.get("classes", [])
        _bcs_source  = _bcs_lit.get("source", "")
        _bcs_badge   = _bcs_badge_html(_bcs_classes, _bcs_source)

        # ── Drug header ───────────────────────────────────────────────────────
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#002147,#003a7a);'
            f'border-radius:10px;padding:14px 20px;margin-bottom:12px;">'
            f'<div style="display:flex;align-items:center;gap:14px;">'
            f'<img src="{_esc(_pc.get("img_url",""))}" '
            f'style="width:80px;height:80px;object-fit:contain;background:white;'
            f'border-radius:8px;padding:4px;" onerror="this.style.display=\"none\"">'
            f'<div style="flex:1;">'
            f'<div style="font-size:20px;font-weight:700;color:white;">{_esc(_as["name"])}</div>'
            f'<div style="font-size:12px;color:rgba(255,255,255,0.6);margin-top:3px;">'
            f'{_esc(_pc.get("formula",""))} &nbsp;·&nbsp; CAS {_esc(_pc.get("cas","N/A"))} '
            f'&nbsp;·&nbsp; MW {_esc(_pc.get("mw",""))} g/mol</div>'
            f'<div style="font-size:11px;color:rgba(255,255,255,0.45);margin-top:2px;">'
            f'{_esc((_pc.get("name","") or "")[:90])}</div>'
            f'</div>'
            f'<div style="display:flex;flex-direction:column;gap:5px;align-items:flex-end;">'
            f'{_bcs_badge}'
            f'</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

        # ── 5 metric cards ─────────────────────────────────────────────────────
        _m1,_m2,_m3,_m4,_m5 = st.columns(5)
        for _col, _lbl, _val, _unit, _note in [
            (_m1,"MW",       f'{_pc.get("mw","")}',         "g/mol",  "Molecular weight"),
            (_m2,"LogP",     f'{_pc.get("xlogp","N/A")}',   "",       "Lipophilicity"),
            (_m3,"TPSA",     f'{_pc.get("tpsa","")}',       "Å²",     "Polar surface"),
            (_m4,"HBD/HBA",  f'{_pc.get("hbd","")}/{_pc.get("hba","")}', "","H-bond"),
            (_m5,"Rot.Bond", f'{_pc.get("rot_bonds","")}',  "",       "Flexibility"),
        ]:
            _col.markdown(
                f'<div style="background:#16203F;border-radius:8px;padding:10px;'
                f'text-align:center;border:1px solid rgba(255,255,255,0.08);">'
                f'<div style="font-size:9px;font-weight:700;color:#9fb0d0;'
                f'text-transform:uppercase;letter-spacing:0.4px;">{_lbl}</div>'
                f'<div style="font-size:16px;font-weight:700;color:#E6ECF8;margin-top:3px;">'
                f'{_val}<span style="font-size:10px;color:#7E8DAB;"> {_unit}</span></div>'
                f'<div style="font-size:9px;color:#7E8DAB;margin-top:1px;">{_note}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

        # -- 4 tabs ───────────────────────────────────────────────────────────
        _t1, _t2, _t3, _t4 = st.tabs([
            f"🏛️ FDA Methods ({len(_fda)})" ,
            "📖 Literature & BCS",
            "🧪 PubChem Data",
            "📚 PubMed"
        ])

        # ── TAB 1: FDA Methods ──────────────────────────────────────────────
        with _t1:
            if not _fda:
                st.info(
                    f"No parameterized record found for **{_as['name']}** in FDA Dissolution Methods Database. "
                    f"See USP monograph."
                )
            else:
                # Filter badges
                _refer_count = sum(1 for m in _fda if not m.get("apparatus") or "Refer" in m.get("apparatus",""))
                _data_count  = len(_fda) - _refer_count
                st.markdown(
                    f'<div style="font-size:12px;color:#9fb0d0;margin-bottom:10px;">'
                    f'<strong>{len(_fda)}</strong> FDA records — '
                    f'<span style="color:#185fa5;">{_data_count} parameters available</span>'
                    f'{", " + str(_refer_count) + " Refer to USP" if _refer_count else ""}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                for _fi, _fm in enumerate(_fda):
                    _is_refer = not _fm.get("apparatus") or "Refer" in _fm.get("apparatus","")
                    _is_sel   = (_sel == _fi)
                    _s_low    = _fm.get("sampling_times","").lower()
                    _unit_lbl = "hr" if ("hour" in _s_low or " hr" in _s_low) else "min"
                    _t_nums   = _re_api2.findall(r'\d+\.?\d*', _fm.get("sampling_times","").split(";")[0])

                    if not _is_refer:
                        st.markdown(
                            f'<div style="background:{"rgba(255,191,0,0.04)" if _is_sel else "#16203F"};'
                            f'border:{"2px solid #FFBF00" if _is_sel else "0.5px solid rgba(255,255,255,0.08)"};'
                            f'border-radius:10px;overflow:hidden;margin-bottom:10px;">'
                            f'<div style="background:#16203F;padding:9px 14px;'
                            f'display:flex;align-items:center;justify-content:space-between;">'
                            f'<div><span style="font-size:13px;font-weight:600;color:#E6ECF8;">'
                            f'{_fm["drug_name"]}</span>'
                            f'<span style="font-size:11px;color:#9fb0d0;margin-left:8px;">'
                            f'{_fm["dosage_form"]}</span>'
                            f'<span style="font-size:10px;color:#7E8DAB;margin-left:6px;">'
                            f'· {_fm.get("date_updated","")}</span></div>'
                            + (f'<span style="background:#FFBF00;color:#002147;font-size:9px;'
                               f'font-weight:800;padding:2px 8px;border-radius:20px;">✓ SELECTED</span>'
                               if _is_sel else
                               f'<span style="background:#dbeafe;color:#185fa5;font-size:9px;'
                               f'font-weight:600;padding:2px 8px;border-radius:20px;">🏛️ FDA</span>') +
                            f'</div>'
                            f'<div style="padding:10px 14px;">'
                            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:8px;">'
                            + "".join([
                                f'<div><div style="font-size:9px;font-weight:700;color:#9fb0d0;'
                                f'text-transform:uppercase;">{l}</div>'
                                f'<div style="font-size:12px;font-weight:600;color:#E6ECF8;">{v}</div></div>'
                                for l,v in [
                                    ("Apparatus",  _fm.get("apparatus","") or "—"),
                                    ("Speed",     (_fm.get("speed_rpm","") or "—") + " rpm"),
                                    ("Volume",   (_fm.get("volume_ml","") or "—") + " mL"),
                                    ("Temperature","37.0 °C"),
                                ]
                            ]) +
                            f'</div>'
                            f'<div style="background:#16203F;border-radius:5px;padding:6px 10px;'
                            f'font-size:11px;margin-bottom:7px;">'
                            f'<strong>Medium:</strong> {_fm.get("medium","")}</div>'
                            f'<div style="font-size:10px;margin-bottom:8px;">'
                            f'<strong>Sampling:</strong> '
                            + " ".join([
                                f'<span style="background:#002147;color:white;font-size:10px;'
                                f'padding:1px 7px;border-radius:10px;">'
                                f'{int(float(t)) if float(t)==int(float(t)) else t}</span>'
                                for t in _t_nums
                            ]) +
                            f' <span style="font-size:10px;color:#FFBF00;background:#002147;'
                            f'padding:1px 6px;border-radius:8px;">{_unit_lbl}</span>'
                            f'</div></div></div>',
                            unsafe_allow_html=True
                        )
                        _ba, _bb = st.columns([2,1])
                        with _ba:
                            if st.button(
                                ("Apply to Method Settings" if not _is_sel else "Selected"),
                                key=f"api_import_{_fi}",
                                use_container_width=True,
                                type="primary" if not _is_sel else "secondary",
                            ):
                                _cfg = st.session_state.method_cfg
                                _an = _re_api2.search(r'Apparatus\s+(\d+)', _fm.get("apparatus",""))
                                _amap = {"1":"USP I (Basket)","2":"USP II (Paddle)",
                                         "3":"USP III (Reciprocating Cylinder)","4":"USP IV (Flow-Through Cell)"}
                                if _an: _cfg["apparatus"] = _amap.get(_an.group(1), _fm["apparatus"])
                                if _fm.get("medium") and "Refer" not in _fm.get("medium",""):
                                    _cfg["medium"] = _fm["medium"]
                                if str(_fm.get("speed_rpm","")).isdigit():
                                    _cfg["rpm"] = int(_fm["speed_rpm"])
                                if str(_fm.get("volume_ml","")).isdigit():
                                    _cfg["volume_ml"] = int(_fm["volume_ml"])
                                _cfg["temp_c"] = 37.0
                                _nums = [float(x) for x in _re_api2.findall(r'\d+\.?\d*', _fm.get("sampling_times","").split(";")[0])]
                                if _nums:
                                    _is_hr = "hour" in _s_low or " hr" in _s_low
                                    _cfg["q_time"] = max(_nums) * 60 if _is_hr else max(_nums)
                                    _cfg["q_limit"] = 80.0
                                st.session_state.method_cfg = _cfg
                                st.session_state["active_substance"]["selected_method"] = _fi
                                st.success(f"✅ Method Settings updated!")
                                st.rerun()
                        with _bb:
                            st.markdown(
                                f'<a href="https://www.accessdata.fda.gov/scripts/cder/dissolution/" '
                                f'target="_blank" style="display:block;text-align:center;padding:7px;'
                                f'background:#16203F;border:1px solid rgba(255,255,255,0.08);border-radius:7px;'
                                f'font-size:12px;color:#E6ECF8;text-decoration:none;">🔗 FDA Source</a>',
                                unsafe_allow_html=True
                            )
                    else:
                        pass
                        # Refer to USP card
        # ── TAB 2: Literature & BCS ────────────────────────────────────────────
        with _t2:
            _bcs_lit2    = _as.get("bcs_from_lit") or {}
            _bcs_cls2    = _bcs_lit2.get("classes", [])
            _bcs_papers2 = _bcs_lit2.get("papers", [])
            _bcs_snips2  = _bcs_lit2.get("snippets", [])
            _bcs_src2    = _bcs_lit2.get("source", "")
            _bcs_title2  = _bcs_lit2.get("title", "")
            _bcs_doi2    = _bcs_lit2.get("doi", "")
            _bcs_journal2 = _bcs_lit2.get("journal", "")

            st.markdown(
                "<div style='font-size:11px;font-weight:700;color:#9fb0d0;"
                "text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;'>"
                f"BCS Class — {_as['name']} — Literature search "
                f"(Scite + PubMed)</div>",
                unsafe_allow_html=True
            )

            if _bcs_cls2:
                # BCS badge(s)
                _badges = _bcs_badge_html(_bcs_cls2, "")
                st.markdown(
                    f'<div style="margin-bottom:12px;">{_badges}</div>',
                    unsafe_allow_html=True
                )

                # Short description by BCS class
                for _cls_i in _bcs_cls2:
                    _m_i = _BCS_META.get(_cls_i, {})
                    st.markdown(
                        f'<div style="background:#16203F;border:0.5px solid rgba(255,255,255,0.08);'
                        f'border-radius:8px;padding:8px 14px;margin-bottom:6px;'
                        f'display:flex;align-items:center;gap:10px;">'
                        f'<span style="background:{_m_i.get("bg","#eee")};'
                        f'color:{_m_i.get("text","#333")};font-size:10px;font-weight:700;'
                        f'padding:2px 8px;border-radius:10px;white-space:nowrap;">BCS {_cls_i}</span>'
                        f'<span style="font-size:12px;color:#E6ECF8;">{_m_i.get("desc","")}</span>'
                        f'<span style="font-size:11px;color:#9fb0d0;margin-left:auto;">'
                        f'Biowaiver: {_m_i.get("biowaiver","")}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

                # Article list - from most-cited to least
                _papers_to_show = _bcs_papers2 if _bcs_papers2 else (
                    [{"title":_bcs_title2,"doi":_bcs_doi2,"journal":_bcs_journal2,
                      "source":_bcs_src2,"tally":0,"snippets":_bcs_snips2,
                      "classes":_bcs_cls2,"abstract":""}]
                    if _bcs_title2 else []
                )

                if _papers_to_show:
                    st.markdown(
                        f'<div style="font-size:11px;font-weight:700;color:#9fb0d0;'
                        f'text-transform:uppercase;letter-spacing:0.4px;margin-bottom:8px;">'
                        f'Sources ({len(_papers_to_show)} articles — by citation count)</div>',
                        unsafe_allow_html=True
                    )
                    for _pi, _p in enumerate(_papers_to_show):
                        _p_cls   = _p.get("classes", _bcs_cls2)
                        _p_m     = _BCS_META.get(_p_cls[0], {}) if _p_cls else {}
                        _tally   = _p.get("tally", 0)
                        _snips_p = _p.get("snippets", [])

                        st.markdown(
                            f'<div style="border-left:3px solid {_p_m.get("border","#e2e8f0")};'
                            f'padding:10px 14px;margin-bottom:8px;'
                            f'background:{_p_m.get("bg","#f8f9fa")}20;border-radius:0 8px 8px 0;">'
                            f'<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:4px;">'
                            f'<div style="font-size:12px;font-weight:600;color:#E6ECF8;line-height:1.4;flex:1;">'
                            f'{_esc(_p.get("title","")[:100])}{"..." if len(_p.get("title",""))>100 else ""}'
                            f'</div>'
                            f'<div style="display:flex;gap:5px;align-items:center;margin-left:8px;flex-shrink:0;">'
                            + (f'<span style="background:#dbeafe;color:#185fa5;font-size:9px;font-weight:600;'
                               f'padding:2px 6px;border-radius:8px;">Scite</span>' 
                               if _p.get("origin","Scite") == "Scite" and _tally > 0 else
                               f'<span style="background:#f0e6ff;color:#6b21a8;font-size:9px;font-weight:600;'
                               f'padding:2px 6px;border-radius:8px;">PubMed</span>') +
                            (f'<span style="background:#f0f0f0;color:#718096;font-size:10px;'
                             f'padding:2px 7px;border-radius:10px;">{_tally} citations</span>'
                             if _tally > 0 else '') +
                            f'</div>'
                            f'</div>'
                            f'<div style="font-size:10px;color:#9fb0d0;">'
                            f'{_esc(_p.get("source",""))}'
                            f'{" · " + _esc(_p.get("journal","")[:40]) if _p.get("journal") else ""}'
                            + (f' · <a href="https://doi.org/{_esc(_p.get("doi"))}" target="_blank" '
                               f'style="color:#185fa5;">DOI ↗</a>' if _p.get("doi") else "") +
                            f'</div>'
                            + ("".join([
                                f'<div style="margin-top:5px;padding:5px 8px;'
                                f'background:rgba(255,255,255,0.7);border-radius:5px;'
                                f'font-size:11px;color:#555;line-height:1.5;'
                                f'border-left:2px solid {_p_m.get("border","#ddd")};">'
                                f'<em>"{_esc(s[:200])}{"..." if len(s)>200 else ""}"</em>'
                                f'</div>'
                                for s in _snips_p
                            ]) if _snips_p else "") +
                            f'</div>',
                            unsafe_allow_html=True
                        )

            else:
                st.info(
                    f"**{_as['name']}** has no article containing a BCS class.\n\n"
                    f"BCS classification can be done via experimental solubility (pH 1.2, 4.5, 6.8) and Caco-2/PAMPA permeability "
                    f"measurement (FDA Guidance 2000, ICH M9)."
                )

            # Source note - shortened
            st.markdown(
                '<div style="margin-top:12px;padding-top:10px;border-top:0.5px solid rgba(255,255,255,0.08);'
                'font-size:10px;color:#7E8DAB;line-height:1.6;">'
                'Source: Scite (Smart Citations) · PubMed NIH (eutils API)\n'
                'BCS class requires experimental measurement - FDA Guidance 2000, ICH M9 2019, Amidon 1995.'
                '</div>',
                unsafe_allow_html=True
            )

        # ── TAB 3: PubChem ────────────────────────────────────────────────────
        with _t3:
            _lipo = _lipinski_check(_pc) if _pc and "error" not in _pc else None
            st.markdown(
                f'<div style="font-size:11px;color:#9fb0d0;margin-bottom:10px;">'
                f'Source: PubChem (NIH) CID {_pc.get("cid","")} · '
                f'Calculated values — validate with experimental data.</div>',
                unsafe_allow_html=True
            )
            _pc1, _pc2 = st.columns(2)

            with _pc1:
                st.markdown(
                    '<div style="font-size:10px;font-weight:700;color:#9fb0d0;'
                    'text-transform:uppercase;margin-bottom:8px;">Physicochemical</div>',
                    unsafe_allow_html=True
                )
                _phys = [
                    ("Molecular Formula",   _pc.get("formula","")),
                    ("Molecular Weight",  f'{_pc.get("mw","")} g/mol'),
                    ("Exact Mass",         f'{_pc.get("exact_mass","")}'),
                    ("XLogP",              f'{_pc.get("xlogp","N/A")} (lipophilicity)'),
                    ("TPSA",               f'{_pc.get("tpsa","")} Å² (limit: 140)'),
                    ("H-Bond Donor",       f'{_pc.get("hbd","")}'),
                    ("H-Bond Acceptor",    f'{_pc.get("hba","")}'),
                    ("Rotatable Bond",      f'{_pc.get("rot_bonds","")}'),
                    ("Molecular Charge",      f'{_pc.get("charge","")}'),
                    ("CAS Number",       _pc.get("cas","N/A")),
                ]
                for _k, _v in _phys:
                    _warn = _k == "TPSA" and _pc.get("tpsa",0) and float(str(_pc.get("tpsa",0)).replace("N/A","0") or 0) > 120
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:5px 0;border-bottom:0.5px solid rgba(255,255,255,0.08);">'
                        f'<span style="font-size:11px;color:#9fb0d0;">{_k}</span>'
                        f'<span style="font-size:11px;font-weight:500;'
                        f'color:{"#856404" if _warn else "#E6ECF8"};">{_v}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            with _pc2:
                if _lipo:
                    st.markdown(
                        '<div style="font-size:10px;font-weight:700;color:#9fb0d0;'
                        'text-transform:uppercase;margin-bottom:8px;">Lipinski Rule of Five</div>',
                        unsafe_allow_html=True
                    )
                    for _rule, (_val, _lim, _ok) in _lipo["rules"].items():
                        _vstr = f"{_val:.2f}" if isinstance(_val, float) else str(_val)
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;'
                            f'padding:5px 0;border-bottom:0.5px solid rgba(255,255,255,0.08);">'
                            f'<span style="font-size:11px;color:#9fb0d0;">{"✅" if _ok else "❌"} {_rule}</span>'
                            f'<span style="font-size:11px;font-weight:500;'
                            f'color:{"#1a5c2e" if _ok else "#9c1c1c"};">{_vstr}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    _ro5_color = "#c6efce" if _lipo["druglike"] else "#ffc7ce"
                    _ro5_msg = "Orally active - no violations" if _lipo["druglike"] else "Violations present"
                    st.markdown(
                        f'<div style="background:{_ro5_color};border-radius:7px;padding:7px 12px;"'
                        f'font-size:12px;font-weight:500;color:#002147;margin-top:8px;">'
                        f'{_ro5_msg}</div>',
                        unsafe_allow_html=True
                    )

                if _pc.get("synonyms"):
                    st.markdown(
                        f'<div style="margin-top:14px;">'
                        f'<div style="font-size:10px;font-weight:700;color:#9fb0d0;'
                        f'text-transform:uppercase;margin-bottom:6px;">Synonyms</div>'
                        f'<div style="font-size:11px;color:#9fb0d0;line-height:1.8;">'
                        f'{", ".join(_pc.get("synonyms",[])[:8])}</div></div>',
                        unsafe_allow_html=True
                    )

                if _pc.get("pubchem_url"):
                    st.markdown(
                        f'<a href="{_pc["pubchem_url"]}" target="_blank" '
                        f'style="display:inline-block;margin-top:12px;font-size:11px;'
                        f'color:#185fa5;">PubChem page</a>',
                        unsafe_allow_html=True
                    )

        # ── TAB 4: PubMed ─────────────────────────────────────────────────────
        with _t4:
            with st.spinner(f"Searching PubMed Hybrid (MeSH+TIAB)..."):
                _pm_results, _pm_total, _pm_qtrans = _pubmed_search(_as["name"])

            if not _pm_results:
                st.warning("No PubMed results found.")



            else:
                st.markdown(
                    f'<div style="font-size:12px;color:#9fb0d0;margin-bottom:6px;">'
                    f'<strong>{_pm_total}</strong> articles — Hybrid strategy '
                    f'(MeSH + Title/Abstract) · first {len(_pm_results)} shown</div>',
                    unsafe_allow_html=True
                )
                with st.expander("🔍 PubMed Query Translation (debug)", expanded=False):
                    st.code(_pm_qtrans or "Query translation unavailable", language="text")
                    st.caption(
                        "This shows how PubMed interpreted your query. "
                        "You can check whether MeSH terms were auto-expanded."
                    )
                for _art in _pm_results:
                    _is_oa = bool(_art.get("pmc"))
                    _tag   = "OA" if _is_oa else _art.get("year","")
                    _tc    = "#27ae60" if _is_oa else "#718096"
                    _tb    = "#c6efce" if _is_oa else "#f0f0f0"
                    st.markdown(
                        f'<div style="display:flex;gap:10px;padding:9px 12px;'
                        f'background:#16203F;border:0.5px solid rgba(255,255,255,0.08);'
                        f'border-radius:8px;margin-bottom:5px;">'
                        f'<span style="background:{_tb};color:{_tc};font-size:10px;'
                        f'font-weight:500;padding:2px 7px;border-radius:20px;'
                        f'white-space:nowrap;height:fit-content;margin-top:1px;">{_tag}</span>'
                        f'<div style="flex:1;">'
                        f'<div style="font-size:12px;font-weight:500;color:#E6ECF8;'
                        f'line-height:1.4;margin-bottom:3px;">{_esc(_art["title"][:120])}'
                        f'{"..." if len(_art["title"])>120 else ""}</div>'
                        f'<div style="font-size:10px;color:#9fb0d0;">'
                        f'{_esc(_art["first_author"])} et al. · {_esc(_art["journal"][:50])} · {_esc(_art["year"])}'
                        f'</div>'
                        + (f'<div style="font-size:10px;color:#185fa5;margin-top:1px;">'
                           f'<a href="https://doi.org/{_art["doi"]}" target="_blank" '
                           f'style="color:#185fa5;">DOI {_art["doi"]}</a></div>'
                           if _art.get("doi") else '') +
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

        # Footer
        st.markdown(
            '<div style="margin-top:16px;padding-top:10px;border-top:0.5px solid rgba(255,255,255,0.08);'
            'font-size:10px;color:#7E8DAB;display:flex;justify-content:space-between;">'
            '<span>Sources: PubMed (NIH) · PubChem · FDA Dissolution DB</span>'
            '<span>API Information — DissolvA v4.0</span>'
            '</div>',
            unsafe_allow_html=True
        )

# PAGE: ALL REFERENCES
# ===========================================================================
