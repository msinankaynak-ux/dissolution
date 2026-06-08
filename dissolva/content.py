"""DissolvA reference/literature content and the profile-shape recommendation engine.
Extracted from app.py (Phase 3a modularization)."""
import streamlit as st
import numpy as np
from dissolva.models import fda_f2_mask, f2_score


# ── Bootstrap f2 gereklilik kontrolü (paylaşılan tek kaynak) ─────────────────
def bootstrap_recommendation(profiles, ref_nm, test_nm):
    """Kullanıcının verisine göre Bootstrap f2 gerekli mi kararı.
    f1/f2 ve Bootstrap sayfaları ortak kullanır. Döndürdüğü dict:
      needs_boot, fda_cv_ok, reasons[], recommended_method, n_vessels,
      cv_max, cv_early_max, cv_late_max, f2 (hesaplanabiliyorsa)
    FDA (1997) CV kriteri (zaman bazlı): t ≤ 15 dk → CV ≤ 20%, t > 15 dk → CV ≤ 10%.
    Ek tetikleyiciler: f2 sınır bölgesi (45–55), n < 12 (FDA n ≥ 12 önerir)."""
    d_ref = profiles.get(ref_nm, {}) or {}
    d_tst = profiles.get(test_nm, {}) or {}

    # Vessel sayısı
    n_vessels = min(d_ref.get("n", 6) or 6, d_tst.get("n", 6) or 6)

    # CV (RSD) toplama — genel + FDA zaman bazlı (erken/geç)
    rsd_all, rsd_early, rsd_late = [], [], []
    for d in (d_ref, d_tst):
        rsd = d.get("rsd"); tt = d.get("time")
        if rsd:
            rsd_all.extend([x for x in rsd if x is not None])
            if tt and len(tt) == len(rsd):
                _t = np.asarray(tt, float); _r = np.asarray(rsd, float)
                # FDA: %20 erken noktalarda (örn. ≤15 dk) VE en erken örnekleme
                # noktasında (ilk nokta t>15 olsa bile) geçerli; diğerlerinde %10.
                early_mask = (_t <= 15.0) | (_t == _t.min())
                rsd_early.extend(_r[early_mask].tolist())
                rsd_late.extend(_r[~early_mask].tolist())
    cv_max       = max(rsd_all)   if rsd_all   else 0.0
    cv_early_max = max(rsd_early) if rsd_early else 0.0
    cv_late_max  = max(rsd_late)  if rsd_late  else 0.0
    has_cv = bool(rsd_all)

    # Gözlemlenen f2 (ortak zaman noktalarında, FDA 85% kuralıyla)
    f2_val = None
    try:
        t_ref = np.asarray(d_ref.get("time", []), float)
        r_ref = np.asarray(d_ref.get("release", []), float)
        t_tst = np.asarray(d_tst.get("time", []), float)
        r_tst = np.asarray(d_tst.get("release", []), float)
        common = np.intersect1d(t_ref, t_tst)
        if len(common) > 0:
            rr = np.array([r_ref[np.where(t_ref == ti)[0][0]] for ti in common])
            rt = np.array([r_tst[np.where(t_tst == ti)[0][0]] for ti in common])
            m = fda_f2_mask(rr)
            if m.any():
                f2_val = f2_score(rr[m], rt[m])
    except Exception:
        f2_val = None

    is_boundary = (f2_val is not None) and (45.0 <= f2_val <= 55.0)
    low_n       = n_vessels < 12
    high_cv     = cv_max > 15.0
    fda_cv_ok   = (cv_early_max <= 20.0) and (cv_late_max <= 10.0)

    reasons = []
    if is_boundary:
        reasons.append(f"f2 = {f2_val:.2f} → 45–55 sınır bölgesinde (istatistiksel olarak güvenilmez)")
    if not fda_cv_ok:
        if cv_early_max > 20.0:
            reasons.append(f"Erken nokta CV% = {cv_early_max:.1f}% → > 20% (FDA t≤15 dk kriteri aşıldı)")
        if cv_late_max > 10.0:
            reasons.append(f"Geç nokta CV% = {cv_late_max:.1f}% → > 10% (FDA t>15 dk kriteri aşıldı)")
    elif high_cv:
        reasons.append(f"Maks CV% = {cv_max:.1f}% → > 15% (FDA eşiği)")
    if low_n:
        reasons.append(f"n = {n_vessels} vessel → FDA n ≥ 12 önerir")

    needs_boot = is_boundary or high_cv or (not fda_cv_ok) or low_n
    recommended_method = "Nonparametric (Shah 1998)" if cv_max > 15.0 else "Parametric"

    return {
        "needs_boot": needs_boot, "fda_cv_ok": fda_cv_ok, "reasons": reasons,
        "recommended_method": recommended_method, "n_vessels": n_vessels,
        "cv_max": cv_max, "cv_early_max": cv_early_max, "cv_late_max": cv_late_max,
        "has_cv": has_cv, "f2": f2_val,
    }

# ── Literature references ────────────────────────────────────────────────────
LITERATURE = {
    "Data Input": [
        "United States Pharmacopeia. (2023). <711> Dissolution. *USP 46–NF 41*.",
        "U.S. Food and Drug Administration. (1997). Guidance for industry: Dissolution testing of immediate release solid oral dosage forms. FDA.",
    ],
    "Kinetic Model Fitting": [
        "Wagner, J. G. (1969). Interpretation of percent dissolved-time plots derived from in vitro testing of conventional tablets and capsules. *Journal of Pharmaceutical Sciences*, 58(10), 1253–1257.",
        "Higuchi, T. (1961). Rate of release of medicaments from ointment bases containing drugs in suspension. *Journal of Pharmaceutical Sciences*, 50(10), 874–875.",
        "Korsmeyer, R. W., Gurny, R., Doelker, E., Buri, P., & Peppas, N. A. (1983). Mechanisms of solute release from porous hydrophilic polymers. *International Journal of Pharmaceutics*, 15(1), 25–35.",
        "Weibull, W. (1951). A statistical distribution function of wide applicability. *Journal of Applied Mechanics*, 18(3), 293–297.",
        "Peppas, N. A., & Sahlin, J. J. (1989). A simple equation for the description of solute release. III. *International Journal of Pharmaceutics*, 57(2), 169–172.",
    ],
    "Statistical Analysis": [
        "Moore, J. W., & Flanner, H. H. (1996). Mathematical comparison of dissolution profiles. *Pharmaceutical Technology*, 20(6), 64–74.",
        "Costa, P., & Lobo, J. M. S. (2001). Modeling and comparison of dissolution profiles. *European Journal of Pharmaceutical Sciences*, 13(2), 123–133.",
    ],
    "f1 and f2 Similarity": [
        "Shah, V. P., Tsong, Y., Sathe, P., & Liu, J. P. (1998). In vitro dissolution profile comparison — statistics and analysis of the similarity factor, f2. *Pharmaceutical Research*, 15(6), 889–896.",
        "U.S. Food and Drug Administration. (1997). Guidance for industry: Dissolution testing of immediate release solid oral dosage forms. FDA.",
        "European Medicines Agency. (2010). Guideline on the investigation of bioequivalence. EMA/CPMP/EWP/QWP/1401/98 Rev. 1.",
    ],
    "Bootstrap f2 Analysis": [
        "Shah, V. P., Tsong, Y., Sathe, P., & Liu, J. P. (1998). In vitro dissolution profile comparison — statistics and analysis of the similarity factor, f2. *Pharmaceutical Research*, 15(6), 889–896.",
        "Mendyk, A., et al. (2012). KinetDS: An open source software for dissolution test data analysis. *Dissolution Technologies*, 19(1), 6–11.",
    ],
    "IVIVC Analysis": [
        "Wagner, J. G., & Nelson, E. (1964). Kinetic analysis of blood levels and urinary excretion in the absorptive phase after single doses of drug. *Journal of Pharmaceutical Sciences*, 53(11), 1392–1403.",
        "U.S. Food and Drug Administration. (1997). Guidance for industry: Extended release oral dosage forms — development, evaluation, and application of in vitro/in vivo correlations. FDA.",
        "Emami, J. (2006). In vitro–in vivo correlation: From theory to applications. *Journal of Pharmacy & Pharmaceutical Sciences*, 9(2), 169–189.",
    ],
}

def show_literature(page_key):
    refs = LITERATURE.get(page_key, [])
    if not refs:
        return
    with st.expander("📚 Literature References (APA Format)", expanded=False):
        for i, ref in enumerate(refs, 1):
            st.markdown(f"**{i}.** {ref}")

# ── Complete Program Reference List ─────────────────────────────────────────────
ALL_REFERENCES = [
    # === DISSOLUTION TESTING & REGULATORY ===
    ("Dissolution Testing & Regulatory", [
        "U.S. Food and Drug Administration. (1997). *Guidance for industry: Dissolution testing of immediate release solid oral dosage forms*. FDA, CDER.",
        "United States Pharmacopeia. (2023). <711> Dissolution. *USP 46–NF 41*. USP Convention.",
        "European Medicines Agency. (2010). *Guideline on the investigation of bioequivalence*. CPMP/EWP/QWP/1401/98 Rev. 1. EMA.",
        "U.S. Food and Drug Administration. (1997). *Guidance for industry: Extended release oral dosage forms — development, evaluation, and application of in vitro/in vivo correlations*. FDA, CDER.",
    ]),
    # === KINETIC MODELS ===
    ("Kinetic Model References", [
        "Wagner, J. G. (1969). Interpretation of percent dissolved-time plots derived from in vitro testing of conventional tablets and capsules. *Journal of Pharmaceutical Sciences*, 58(10), 1253–1257.",
        "Higuchi, T. (1961). Rate of release of medicaments from ointment bases containing drugs in suspension. *Journal of Pharmaceutical Sciences*, 50(10), 874–875.",
        "Higuchi, T. (1963). Mechanism of sustained-action medication. *Journal of Pharmaceutical Sciences*, 52(12), 1145–1149.",
        "Korsmeyer, R. W., Gurny, R., Doelker, E., Buri, P., & Peppas, N. A. (1983). Mechanisms of solute release from porous hydrophilic polymers. *International Journal of Pharmaceutics*, 15(1), 25–35.",
        "Peppas, N. A. (1985). Analysis of Fickian and non-Fickian drug release from polymers. *Pharmaceutica Acta Helvetiae*, 60(4), 110–111.",
        "Peppas, N. A., & Sahlin, J. J. (1989). A simple equation for the description of solute release. III. Coupling of diffusion and relaxation. *International Journal of Pharmaceutics*, 57(2), 169–172.",
        "Weibull, W. (1951). A statistical distribution function of wide applicability. *Journal of Applied Mechanics*, 18(3), 293–297.",
        "Hixson, A. W., & Crowell, J. H. (1931). Dependence of reaction velocity upon surface and agitation. *Industrial & Engineering Chemistry*, 23(8), 923–931.",
        "Hopfenberg, H. B. (1976). *Controlled release polymeric formulations* (ACS Symposium Series, Vol. 33). American Chemical Society.",
        "Baker, R. W., & Lonsdale, H. S. (1974). *Controlled release of biologically active agents*. Plenum Press.",
        "Makoid, M. C., Dufour, A., & Banakar, U. V. (1993). Modelling of dissolution behaviour of controlled release systems. *STP Pharma*, 3(1), 49–58.",
        "Ritger, P. L., & Peppas, N. A. (1987). A simple equation for description of solute release I. Fickian and non-Fickian release from nonswellable devices. *Journal of Controlled Release*, 5(1), 23–36.",
        "Langenbucher, F. (1972). Linearization of dissolution rate curves by the Weibull distribution. *Journal of Pharmacy and Pharmacology*, 24(12), 979–981.",
        "Gompertz, B. (1825). On the nature of the function expressive of the law of human mortality. *Philosophical Transactions of the Royal Society*, 115, 513–583.",
        "Richards, F. J. (1959). A flexible growth function for empirical use. *Journal of Experimental Botany*, 10(2), 290–301.",
        "Macheras, P., & Dokoumetzidis, A. (2000). On the heterogeneity of drug dissolution and release. *Pharmaceutical Research*, 17(2), 108–112.",
    ]),
    # === SIMILARITY & STATISTICS ===
    ("f1/f2 & Statistical Methods", [
        "Moore, J. W., & Flanner, H. H. (1996). Mathematical comparison of dissolution profiles. *Pharmaceutical Technology*, 20(6), 64–74.",
        "Shah, V. P., Tsong, Y., Sathe, P., & Liu, J. P. (1998). In vitro dissolution profile comparison — statistics and analysis of the similarity factor, f2. *Pharmaceutical Research*, 15(6), 889–896.",
        "European Medicines Agency. (2010). *Guideline on the investigation of bioequivalence*. EMA/CHMP/EWP/QWP/1401/98 Rev. 1. EMA. [Bootstrap f2 methodology — Section 4.1.1]",
        "European Medicines Agency. (2018). *Guideline on the pharmacokinetic and clinical evaluation of modified release dosage forms*. EMA/CPMP/EWP/280/96 Corr1.",
        "Costa, P., & Lobo, J. M. S. (2001). Modeling and comparison of dissolution profiles. *European Journal of Pharmaceutical Sciences*, 13(2), 123–133.",
        "Tsong, Y., Hammerstrom, T., Sathe, P., & Shah, V. P. (1996). Statistical assessment of mean differences between two dissolution data sets. *Drug Information Journal*, 30(4), 1105–1112.",
        "Polli, J. E., Rekhi, G. S., Augsburger, L. L., & Shah, V. P. (1997). Methods to compare dissolution profiles and a rationale for wide dissolution specifications for metoprolol tartrate tablets. *Journal of Pharmaceutical Sciences*, 86(6), 690–700.",
        "Anderson, N. H., Bauer, M., Boussac, N., Khan-Malek, R., Munden, P., & Sardaro, M. (1998). An evaluation of fit factors and dissolution efficiency for the comparison of in vitro dissolution profiles. *Journal of Pharmaceutical and Biomedical Analysis*, 17(4–5), 811–822.",
        "Khan, K. A. (1975). The concept of dissolution efficiency. *Journal of Pharmacy and Pharmacology*, 27(1), 48–49.",
    ]),
    # === SOFTWARE & TOOLS ===
    ("Dissolution Analysis Software", [
        "Zhang, Y., Huo, M., Zhou, J., Zou, A., Li, W., Yao, C., & Xie, S. (2010). DDSolver: An add-in program for modeling and comparison of drug dissolution profiles. *The AAPS Journal*, 12(3), 263–271. https://doi.org/10.1208/s12248-010-9185-1",
        "Zuo, J., Gao, Y., Bou-Chacra, N., & Löbenberg, R. (2014). Evaluation of the DDSolver software applications. *BioMed Research International*, 2014, Article 204925. https://doi.org/10.1155/2014/204925",
        "Mendyk, A., Jachowicz, R., Fijorek, K., Dorożyński, P., Kulinowski, P., & Polak, S. (2012). KinetDS: An open source software for dissolution test data analysis. *Dissolution Technologies*, 19(1), 6–11. https://doi.org/10.14227/DT190112P6",
        "O'Hara, T., Dunne, A., Butler, J., & Devane, J. (1998). A review of methods used to compare dissolution profile data. *Pharmaceutical Science & Technology Today*, 1(5), 214–223.",
    ]),
    # === IVIVC ===
    ("IVIVC References", [
        "Wagner, J. G., & Nelson, E. (1964). Kinetic analysis of blood levels and urinary excretion in the absorptive phase after single doses of drug. *Journal of Pharmaceutical Sciences*, 53(11), 1392–1403.",
        "Emami, J. (2006). In vitro–in vivo correlation: From theory to applications. *Journal of Pharmacy & Pharmaceutical Sciences*, 9(2), 169–189.",
        "Siepmann, J., & Siepmann, F. (2008). Mathematical modeling of drug delivery. *International Journal of Pharmaceutics*, 364(2), 328–343.",
    ]),
]

def show_all_references():
    """Show all program references."""
    st.markdown("## 📚 DissolvA — Complete Reference List")
    st.markdown(
        '<div class="info-banner">All scientific sources, regulatory guidelines, and software references '
        'used in DissolvA™ v3.0. Please cite these works in your publications.</div>',
        unsafe_allow_html=True
    )
    for section_title, refs in ALL_REFERENCES:
        st.markdown(f"### {section_title}")
        for i, ref in enumerate(refs, 1):
            st.markdown(f"**{i}.** {ref}")
        st.markdown("---")
    st.caption(
        "DissolvA™ v3.0 — Developed by M. Sinan KAYNAK, PhD | "
        "Anadolu University, Faculty of Pharmacy | dissolva.app@gmail.com"
    )


# ===========================================================================




# ── Smart Model Recommendation ──────────────────────────────────────────────────────
def analyze_profile_shape(t_arr, r_arr):
    """Analyze profile shape and recommend model categories."""
    t = np.array(t_arr, dtype=float)
    r = np.array(r_arr, dtype=float)
    n = len(r)
    r_norm = r / (r.max() + 1e-9)
    t_20idx = max(2, int(n * 0.25))  # At least 2 points, up to 25% of profile
    # Lag detection: ignore t=0 if it's always 0
    _lag_slice = r_norm[1:t_20idx] if r_norm[0] < 0.01 and n > 2 else r_norm[:t_20idx]
    has_lag = _lag_slice.mean() < 0.05 if len(_lag_slice) > 0 else False
    # Sigmoid detection
    is_sigmoid = False
    if n >= 4:
        dr = np.gradient(r_norm, t)
        dr_max_idx = int(np.argmax(dr))
        is_sigmoid = (0.25 * n) < dr_max_idx < (0.75 * n)
    # Burst detection
    _burst_slice = r_norm[1:t_20idx] if r_norm[0] < 0.01 and n > 2 else r_norm[:t_20idx]
    has_burst = _burst_slice.mean() > 0.50 if len(_burst_slice) > 0 else False
    # 80% threshold
    reaches_80 = r.max() >= 78.0
    # Erken plato
    t_80idx = max(1, int(n * 0.8))
    plateau_early = (r[-1] - r[t_80idx]) < 5.0 and not reaches_80

    if has_burst:
        return {
            'shape': 'Burst Release',
            'top_models': ['Zero Order + F0','First Order + Fmax','Higuchi + F0','KP + F0','First Order + Tlag + Fmax'],
            'categories': ['Burst Release','Basic'],
            'reason': f"Early phase high release detected (%{r_norm[:t_20idx].mean()*100:.0f} @ t₂₀). Burst release models recommended.",
            'icon': '⚡'
        }
    elif is_sigmoid:
        return {
            'shape': 'Sigmoid',
            'top_models': ['Weibull','Gompertz 1 (DDSolver)','Logistic 1 (DDSolver)','Hill Equation','Richards'],
            'categories': ['Sigmoid'],
            'reason': f"S-shaped profile detected (inflection point t={t[int(np.argmax(np.gradient(r_norm,t)))]:.1f}). Sigmoid models recommended.",
            'icon': '〜'
        }
    elif has_lag:
        return {
            'shape': 'Lag-Time',
            'top_models': ['First Order + Lag','KP + Lag','Higuchi + Lag','Zero Order + Lag','Hixson-Crowell + Lag'],
            'categories': ['Lag-Time'],
            'reason': f"Initial lag phase detected (t₂₀ avg. release: %{r_norm[:t_20idx].mean()*100:.1f}). Lag-time models recommended.",
            'icon': '⏱️'
        }
    elif plateau_early:
        return {
            'shape': 'Plateau / Fmax',
            'top_models': ['First Order + Fmax','Weibull','Makoid-Banakar','Peppas-Sahlin','Gompertz 2 (DDSolver)'],
            'categories': ['Basic','Empirical'],
            'reason': f"Profile does not reach 80% threshold (max: %{r.max():.1f}). Fmax-parameterized or empirical models recommended.",
            'icon': '📉'
        }
    else:
        return {
            'shape': 'Standard',
            'top_models': ['First Order','Weibull','Korsmeyer-Peppas','Higuchi','Hixson-Crowell'],
            'categories': ['Basic','Lag-Time'],
            'reason': f"Standard release profile (max: %{r.max():.1f}, reaches 80% threshold). Basic kinetic models recommended.",
            'icon': '✓'
        }
