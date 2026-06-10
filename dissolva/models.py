"""DissolvA kinetic model library: 62 dissolution models, the MODEL_DEFS registry,
statistical metrics (R2/AIC/MSC), MDT/DE and the fit_model curve-fitting engine.
Extracted from app.py (Phase 1.4 modularization)."""
import re
import numpy as np
from scipy.optimize import curve_fit, root
from scipy.stats import norm as sp_norm
from scipy.stats import t as _student_t
from scipy.stats import shapiro as _shapiro
from scipy.integrate import trapezoid

def _nz(x): return np.where(x > 0, x, 1e-9)

# -- 62 kinetic model functions --
def m_zero_order(t, k0): return k0 * t
def m_first_order(t, k1): return 100.0*(1-np.exp(-k1*t))
def m_higuchi(t, kH): return kH*np.sqrt(np.abs(t))
def m_hixson_crowell(t, ks):
    inner = 1.0 - ks*t/3.0
    return np.clip(100.0*(1 - np.sign(inner)*np.abs(inner)**3), 0, 100)
def m_korsmeyer_peppas(t, k, n): return k*np.abs(t)**n
def m_hopfenberg(t, kHB, n_HB): return np.clip(100.0*(1-(1-kHB*t)**n_HB), 0, 100)
def m_baker_lonsdale(t, kBL):
    res = []
    for ti in np.atleast_1d(t):
        rhs = float(kBL*ti)
        def eq(F): return [1.5*(1-(1-F[0])**(2/3))-F[0]-rhs]
        try:
            sol = root(eq, [0.5], method="hybr")
            res.append(float(np.clip(sol.x[0]*100, 0, 100)) if sol.success else np.nan)
        except: res.append(np.nan)
    return np.array(res)
def m_makoid_banakar(t, kMB, nMB, bMB): return kMB*np.abs(t)**nMB*np.exp(-bMB*t)
def m_peppas_sahlin(t, k1, k2, m): return k1*np.abs(t)**m + k2*np.abs(t)**(2*m)
def m_weibull(t, a, b, Td): return 100.0*(1-np.exp(-(np.clip(t-Td,0,None)**b)/a))
def m_gompertz(t, A, b, k): return A*np.exp(-b*np.exp(-k*t))
def m_logistic(t, A, k, t50): return A/(1+np.exp(-k*(t-t50)))
def m_quadratic(t, a, b, c): return a*t**2 + b*t + c
def m_probit(t, mu, sigma, A): return A*sp_norm.cdf(t, mu, abs(sigma))
def m_weibull_no_lag(t, a, b): return 100.0*(1-np.exp(-(t**b)/a))
def m_modified_gompertz(t, Amax, mu, lam):
    return Amax*np.exp(-np.exp(mu*np.e/Amax*(lam-t)+1))
def m_richards(t, A, k, n, t50): return A*(1+np.exp(-k*(t-t50)))**(-1/n)
def m_korsmeyer_peppas_lag(t, k, n, tlag): return k*np.clip(t-tlag,0,None)**n
def m_first_order_lag(t, k1, tlag): return 100.0*(1-np.exp(-k1*np.clip(t-tlag,0,None)))
def m_zero_order_lag(t, k0, tlag): return k0*np.clip(t-tlag,0,None)
def m_higuchi_lag(t, kH, tlag): return kH*np.sqrt(np.clip(t-tlag,0,None))
def m_double_exp(t, A1, k1, A2, k2): return A1*(1-np.exp(-k1*t))+A2*(1-np.exp(-k2*t))
def m_triple_exp(t, A1, k1, A2, k2, A3, k3):
    return A1*(1-np.exp(-k1*t))+A2*(1-np.exp(-k2*t))+A3*(1-np.exp(-k3*t))
def m_power_exp(t, A, k, n): return A*(1-np.exp(-k*t**n))
def m_brody(t, A, k, b): return A*(1-b*np.exp(-k*t))
def m_bertalanffy(t, A, k, n): return A*(1-np.exp(-k*t))**n
def m_gallagher_corrigan(t, Amax, k1, k2, tmax):
    return Amax*(1-np.exp(-k1*t))-(Amax-100)*(1-np.exp(-k2*np.clip(t-tmax,0,None)))
def m_logistic_4p(t, A, B, k, t50): return A+(B-A)/(1+np.exp(-k*(t-t50)))
def m_log_normal(t, mu, sigma, A): return A*sp_norm.cdf(np.log(_nz(t)), mu, abs(sigma))
def m_hill(t, Amax, k, n):
    tn = np.abs(t)**n; return Amax*tn/(k**n+tn)
def m_weibull_3p(t, alpha, beta, gamma):
    return 100.0*(1-np.exp(-((np.clip(t-gamma,0,None)/alpha)**beta)))
def m_exp_assoc(t, A, k): return A*(1-np.exp(-k*t))
def m_hyperbolic(t, Amax, k): return Amax*t/(k+t)
def m_linear_exp(t, A, k, b): return A*t*np.exp(-k*t)+b
def m_dose_response(t, Emin, Emax, EC50, n):
    tn = np.abs(t)**n; return Emin+(Emax-Emin)*tn/(EC50**n+tn)
def m_combined(t, kH, k1, alpha):
    return alpha*kH*np.sqrt(np.abs(t))+(1-alpha)*100*(1-np.exp(-k1*t))
def m_pade(t, a0, a1, b1): return (a0+a1*t)/(1+b1*t)
def m_hPLC(t, A, B, n): return 100*(1-(1+A*t**n)**(-B))
def m_compreg(t, k, n, m): return 100*(1-np.exp(-k*t**n))**m
def m_biexp_abs(t, Fr, ka, k):
    return Fr*100*(ka/(ka-k+1e-9))*(np.exp(-k*t)-np.exp(-ka*t))
def m_mb_mod(t, k, n, b, c): return k*np.abs(t)**n*np.exp(-b*t)+c
def m_probit_log(t, mu, sigma, A): return A*sp_norm.cdf(np.log10(_nz(t)), mu, abs(sigma))
def m_henriksen(t, A, k1, k2): return A*(np.exp(-k1*t)-np.exp(-k2*t))
def m_kpmod(t, k, n, b): return k*np.abs(t)**n/(1+b*t)
def m_fractal_fo(t, k, alpha): return 100.0*(1-np.exp(-k*t**alpha))
def m_stretched_exp(t, A, beta, tau): return A*(1-np.exp(-((t/tau)**beta)))
def m_weibull_sig(t, A, k, t50, b): return A/(1+np.exp(-k*(t-t50)))*(1-np.exp(-(t/b)**2))

# ── New models added from DDSolver / KinetDS ──────────────────────────────
# Zero-order with F0 (burst release): F = F0 + k0*t
def m_zero_order_f0(t, k0, F0): return F0 + k0 * t

# First-order with Fmax: F = Fmax*(1-exp(-k1*t))
def m_first_order_fmax(t, k1, Fmax): return Fmax*(1-np.exp(-k1*t))

# First-order with Tlag and Fmax: F = Fmax*(1-exp(-k1*(t-Tlag)))
def m_first_order_tlag_fmax(t, k1, tlag, Fmax):
    return Fmax*(1-np.exp(-k1*np.clip(t-tlag,0,None)))

# Higuchi with F0 (burst): F = F0 + kH*sqrt(t)
def m_higuchi_f0(t, kH, F0): return F0 + kH*np.sqrt(np.abs(t))

# Korsmeyer-Peppas with F0: F = F0 + kKP*t^n
def m_kp_f0(t, k, n, F0): return F0 + k*np.abs(t)**n

# Peppas-Sahlin 2: F = k1*t^0.5 + k2*t
def m_peppas_sahlin2(t, k1, k2): return k1*np.sqrt(np.abs(t)) + k2*t

# Second-order: F = k*t^2
def m_second_order(t, k): return k * t**2

# Third-order: F = k*t^3
def m_third_order(t, k): return k * t**3

# Michaelis-Menten: F = Qmax*t/(k+t)  [same as hyperbolic, explicit reference]
def m_michaelis_menten(t, Qmax, km): return Qmax*t/(km+t)

# Hixson-Crowell with Tlag
def m_hixson_tlag(t, ks, tlag):
    tc = np.clip(t-tlag,0,None)
    inner = 1.0 - ks*tc/3.0
    return np.clip(100.0*(1-np.sign(inner)*np.abs(inner)**3), 0, 100)

# Logistic 1 (DDSolver #332): F = 100*exp(a+b*log(t))/(1+exp(a+b*log(t)))
def m_logistic1_dds(t, alpha, beta):
    x = alpha + beta*np.log(_nz(t))
    return 100.0*np.exp(x)/(1+np.exp(x))

# Logistic 2 (DDSolver #333): F = Fmax*exp(a+b*log(t))/(1+exp(a+b*log(t)))
def m_logistic2_dds(t, alpha, beta, Fmax):
    x = alpha + beta*np.log(_nz(t))
    return Fmax*np.exp(x)/(1+np.exp(x))

# Gompertz 1 (DDSolver #335): F = 100*exp(-exp(a-b*log(t)))
def m_gompertz1_dds(t, alpha, beta):
    return 100.0*np.exp(-np.exp(alpha - beta*np.log(_nz(t))))

# Gompertz 2 (DDSolver #336): F = Fmax*exp(-exp(a-b*log(t)))
def m_gompertz2_dds(t, alpha, beta, Fmax):
    return Fmax*np.exp(-np.exp(alpha - beta*np.log(_nz(t))))

# Probit 1 (DDSolver #339): F = 100*Phi(a + b*log(t))
def m_probit1_dds(t, alpha, beta):
    return 100.0*sp_norm.cdf(alpha + beta*np.log10(_nz(t)))

# -- Statistics & metrics --
def r2s(y,yp):
    ss_res=np.sum((y-yp)**2); ss_tot=np.sum((y-np.mean(y))**2)
    return float(1-ss_res/ss_tot) if ss_tot>0 else 0.0
def r2adj(y,yp,p):
    n=len(y); r2=r2s(y,yp)
    return float(1-(1-r2)*(n-1)/(n-p-1)) if n>p+1 else r2
def aic_fn(y,yp,p):
    n=len(y); sse=max(np.sum((y-yp)**2),1e-12)
    return float(n*np.log(sse/n)+2*p)
def msc_fn(y,yp,p):
    n=len(y); sse=max(np.sum((y-yp)**2),1e-12); sst=max(np.sum((y-np.mean(y))**2),1e-12)
    return float(np.log(sst/sse)-2*p/n)
def rmse_fn(y,yp):
    """Root Mean Square Error — ortalama tahmin hatası, veriyle aynı birimde (%)."""
    n=len(y)
    return float(np.sqrt(np.sum((y-yp)**2)/n)) if n>0 else np.nan
def aicc_fn(y,yp,p):
    """Küçük örneklem için düzeltilmiş AIC (Hurvich & Tsai 1989).
    AICc = AIC + 2p(p+1)/(n-p-1). Dissolüsyonda az nokta olduğu için tercih edilir."""
    n=len(y); aic=aic_fn(y,yp,p)
    denom=n-p-1
    return float(aic + (2*p*(p+1))/denom) if denom>0 else float("inf")
def bic_fn(y,yp,p):
    """Bayesian (Schwarz) Information Criterion — AIC'den ağır parametre cezası: p*ln(n)."""
    n=len(y); sse=max(np.sum((y-yp)**2),1e-12)
    return float(n*np.log(sse/n)+p*np.log(n))
def compute_mdt(t,r):
    f=np.array(r)/100.0; df=np.gradient(f,t)
    num=trapezoid(t*df,t); den=trapezoid(df,t)
    return float(num/den) if abs(den)>1e-12 else np.nan
def compute_de(t,r):
    auc=trapezoid(r,t)
    return float(auc/(t[-1]*100)*100) if t[-1]>0 else np.nan

# -- f1 / f2 similarity factors & FDA 85% point rule --
def fda_f2_mask(ref_mean):
    """FDA 1997 / Shah 1998 nokta kuralı: ref ≤ 85% olan TÜM noktalar +
    85'i İLK aşan nokta dahil; sonraki >85% noktalar dışlanır.
    Boolean mask döndürür (ref_mean ile aynı uzunlukta)."""
    ref_mean = np.asarray(ref_mean, dtype=float)
    mask = ref_mean <= 85.0
    above = np.where(ref_mean > 85.0)[0]
    if len(above) > 0:
        mask[above[0]] = True
    return mask

def f2_score(rr, rt):
    """f2 = 50·log10(100/sqrt(1 + (1/n)Σ(Rt-Tt)²)). rr/rt: maskeli ortalama profiller."""
    rr = np.asarray(rr, dtype=float); rt = np.asarray(rt, dtype=float)
    return float(50*np.log10(100/np.sqrt(1+np.mean((rr-rt)**2))))

def f1_score(rr, rt):
    """f1 = Σ|Rt-Tt| / ΣRt × 100."""
    rr = np.asarray(rr, dtype=float); rt = np.asarray(rt, dtype=float)
    s = np.sum(rr)
    return float(np.sum(np.abs(rr-rt))/s*100) if s != 0 else np.nan

def bootstrap_f2(ref_raw, test_raw, point_mask, iterations=5000, seed=42,
                 method="nonparametric", progress=None):
    """Vessel-düzeyi f2 bootstrap'ı. Değerlendirme noktaları SABİT (point_mask,
    gözlem profilinden belirlenir) — her iterasyonda aynı noktalar kullanılır.

    ref_raw / test_raw : 2-D (n_timepoint × n_vessel)
    point_mask         : boolean (n_timepoint,) — fda_f2_mask(gözlem_ref_ortalaması)
    method             : 'nonparametric' (vessel resampling, Shah 1998) |
                         'parametric' (MVN(ortalama, kovaryans)'den örnekleme)
    progress           : opsiyonel callback(frac, i) ilerleme için
    Döndürür: f2 dağılımı (np.array)."""
    rng = np.random.default_rng(seed if (seed and seed > 0) else None)
    ref_raw = np.asarray(ref_raw, float); test_raw = np.asarray(test_raw, float)
    pm = np.asarray(point_mask, bool)
    nvr = ref_raw.shape[1]; nvt = test_raw.shape[1]
    if method == "parametric":
        mu_r = ref_raw.mean(axis=1); mu_t = test_raw.mean(axis=1)
        ntp = ref_raw.shape[0]
        eye = np.eye(ntp) * 1e-9  # tekil kovaryansa karşı ufak ridge
        cov_r = (np.cov(ref_raw)  if nvr > 1 else np.zeros((ntp, ntp))) + eye
        cov_t = (np.cov(test_raw) if nvt > 1 else np.zeros((test_raw.shape[0],)*2)) + np.eye(test_raw.shape[0])*1e-9
    out = []
    chunk = max(1, iterations // 100)
    for i in range(iterations):
        if method == "parametric":
            R = rng.multivariate_normal(mu_r, cov_r, size=nvr).T.mean(axis=1)
            T = rng.multivariate_normal(mu_t, cov_t, size=nvt).T.mean(axis=1)
        else:
            R = ref_raw[:,  rng.integers(0, nvr, nvr)].mean(axis=1)
            T = test_raw[:, rng.integers(0, nvt, nvt)].mean(axis=1)
        if pm.any():
            mse = np.mean((R[pm] - T[pm])**2)
            out.append(50*np.log10(100/np.sqrt(1+mse)))
        if progress is not None and (i+1) % chunk == 0:
            progress((i+1)/iterations, i+1)
    return np.array(out)

# -- Model registry --
MODEL_DEFS = {
    "Zero Order":            (m_zero_order,       [1.0],                      ["k0"],                   "F=k0*t",                         "Wagner 1969",          "Basic"),
    "First Order":           (m_first_order,      [0.05],                     ["k1"],                   "F=100*(1-exp(-k1*t))",           "Wagner 1969",          "Basic"),
    "Higuchi":               (m_higuchi,          [10.0],                     ["kH"],                   "F=kH*sqrt(t)",                   "Higuchi 1961",         "Basic"),
    "Hixson-Crowell":        (m_hixson_crowell,   [0.05],                     ["ks"],                   "M0^(1/3)-M^(1/3)=ks*t",         "Hixson 1931",          "Basic"),
    "Korsmeyer-Peppas":      (m_korsmeyer_peppas, [10.0, 0.5],                ["k","n"],                "F=k*t^n",                        "Korsmeyer 1983",       "Basic"),
    "Hopfenberg":            (m_hopfenberg,       [0.02, 2.0],                ["kHB","n"],              "F=100*[1-(1-kHB*t)^n]",          "Hopfenberg 1976",      "Basic"),
    "Baker-Lonsdale":        (m_baker_lonsdale,   [0.001],                    ["kBL"],                  "3/2*[1-(1-F)^(2/3)]-F=kBL*t",   "Baker 1974",           "Basic"),
    "Makoid-Banakar":        (m_makoid_banakar,   [10.0, 0.5, 0.01],          ["kMB","nMB","bMB"],      "F=kMB*t^nMB*exp(-bMB*t)",        "Makoid 1993",          "Basic"),
    "Peppas-Sahlin":         (m_peppas_sahlin,    [5.0, 1.0, 0.5],            ["k1","k2","m"],          "F=k1*t^m + k2*t^2m",             "Peppas 1989",          "Basic"),
    "Weibull":               (m_weibull,          [50.0, 1.0, 0.0],           ["a","b","Td"],           "F=100*(1-exp(-((t-Td)^b)/a))",   "Weibull 1951",         "Basic"),
    "Gompertz":              (m_gompertz,         [100.0, 5.0, 0.1],          ["A","b","k"],            "F=A*exp(-b*exp(-k*t))",           "Gompertz 1825",        "Basic"),
    "Logistic":              (m_logistic,         [100.0, 0.1, 30.0],         ["A","k","t50"],          "F=A/(1+exp(-k*(t-t50)))",         "Pressman 1994",        "Basic"),
    "Quadratic":             (m_quadratic,        [-0.01, 1.0, 0.0],          ["a","b","c"],            "F=a*t^2 + b*t + c",              "Polli 1997",           "Basic"),
    "Probit":                (m_probit,           [30.0, 15.0, 100.0],        ["mu","sigma","A"],       "F=A*Phi((t-mu)/sigma)",           "Shah 1998",            "Basic"),
    "Weibull (No Lag)":      (m_weibull_no_lag,   [50.0, 1.0],                ["a","b"],                "F=100*(1-exp(-t^b/a))",           "Weibull 1951",         "Lag-Time"),
    "KP + Lag":              (m_korsmeyer_peppas_lag,[10.0,0.5,5.0],          ["k","n","tlag"],         "F=k*(t-tlag)^n",                  "Modified KP",          "Lag-Time"),
    "First Order + Lag":     (m_first_order_lag,  [0.05, 5.0],                ["k1","tlag"],            "F=100*(1-exp(-k1*(t-tlag)))",     "Modified FO",          "Lag-Time"),
    "Zero Order + Lag":      (m_zero_order_lag,   [1.0, 5.0],                 ["k0","tlag"],            "F=k0*(t-tlag)",                   "Modified ZO",          "Lag-Time"),
    "Higuchi + Lag":         (m_higuchi_lag,      [10.0, 5.0],                ["kH","tlag"],            "F=kH*sqrt(t-tlag)",               "Modified Higuchi",     "Lag-Time"),
    "Probit Log":            (m_probit_log,       [1.5, 0.5, 100.0],          ["mu","sigma","A"],       "F=A*Phi((log10(t)-mu)/sigma)",    "Shah 1998",            "Lag-Time"),
    "Double Exponential":    (m_double_exp,       [60.0,0.05,40.0,0.005],     ["A1","k1","A2","k2"],    "F=A1*(1-e^-k1t)+A2*(1-e^-k2t)", "Empirical",            "Multi-Phase"),
    "Triple Exponential":    (m_triple_exp,       [40.0,0.1,40.0,0.02,20.0,0.005],["A1","k1","A2","k2","A3","k3"],"F=sum Ai*(1-e^-kit)","Empirical",             "Multi-Phase"),
    "Power-Exponential":     (m_power_exp,        [100.0,0.05,1.2],           ["A","k","n"],            "F=A*(1-exp(-k*t^n))",             "Zhang 2010",           "Multi-Phase"),
    "Biexp. Absorption":     (m_biexp_abs,        [1.0,0.2,0.05],             ["Fr","ka","k"],          "F=Fr*100*ka/(ka-k)*(exp(-k*t)-exp(-ka*t))","PK-based",            "Multi-Phase"),
    "Gallagher-Corrigan":    (m_gallagher_corrigan,[100.0,0.05,0.02,60.0],    ["Amax","k1","k2","tmax"],"Biphasic burst+slow",             "Gallagher 2000",       "Multi-Phase"),
    "Combined Higuchi+FO":   (m_combined,         [10.0,0.05,0.5],            ["kH","k1","alpha"],      "F=alpha*kH*sqrt(t)+(1-alpha)*FO","Empirical",            "Multi-Phase"),
    "Henriksen":             (m_henriksen,        [80.0,0.1,0.5],             ["A","k1","k2"],          "F=A*(exp(-k1*t)-exp(-k2*t))",     "Henriksen et al.",     "Multi-Phase"),
    "Modified Gompertz":     (m_modified_gompertz,[100.0,0.1,10.0],           ["Amax","mu","lambda"],   "F=Amax*exp(-exp(mu*e/Amax*(lam-t)+1))","Zwietering 1990", "Sigmoid"),
    "Richards":              (m_richards,         [100.0,0.05,1.0,30.0],      ["A","k","n","t50"],      "F=A*(1+exp(-k*(t-t50)))^(-1/n)", "Richards 1959",        "Sigmoid"),
    "4-Parameter Logistic":  (m_logistic_4p,      [0.0,100.0,0.1,30.0],       ["A","B","k","t50"],      "F=A+(B-A)/(1+exp(-k*(t-t50)))",   "4PL",                  "Sigmoid"),
    "Log-Normal":            (m_log_normal,       [3.5,0.5,100.0],            ["mu","sigma","A"],       "F=A*Phi((ln(t)-mu)/sigma)",       "Statistical",          "Sigmoid"),
    "Hill Equation":         (m_hill,             [100.0,30.0,1.5],           ["Amax","k","n"],         "F=Amax*t^n/(k^n+t^n)",           "Hill 1910",            "Sigmoid"),
    "Dose-Response":         (m_dose_response,    [0.0,100.0,30.0,1.0],       ["Emin","Emax","EC50","n"],"F=Emin+(Emax-Emin)*t^n/(EC50^n+t^n)","Pharmacological", "Sigmoid"),
    "Fractal First Order":   (m_fractal_fo,       [0.05, 0.8],                ["k","alpha"],            "F=100*(1-exp(-k*t^alpha))",       "Macheras 1995",        "Fractal"),
    "Stretched Exponential": (m_stretched_exp,    [100.0,0.8,30.0],           ["A","beta","tau"],       "F=A*(1-exp(-(t/tau)^beta))",      "Kohlrausch 1854",      "Fractal"),
    "Fractal Weibull":       (m_weibull_3p,       [30.0,1.2,0.0],             ["alpha","beta","gamma"], "F=100*(1-exp(-((t-gamma)/alpha)^beta))","Weibull 3P",      "Fractal"),
    "Exponential Assoc.":    (m_exp_assoc,        [100.0, 0.05],              ["A","k"],                "F=A*(1-exp(-k*t))",               "Empirical",            "Empirical"),
    "Hyperbolic":            (m_hyperbolic,       [100.0, 20.0],              ["Amax","k"],             "F=Amax*t/(k+t)",                  "Empirical",            "Empirical"),
    "Linear-Exponential":    (m_linear_exp,       [2.0, 0.02, 0.0],           ["A","k","b"],            "F=A*t*exp(-k*t)+b",               "Empirical",            "Empirical"),
    "Brody Growth":          (m_brody,            [120.0,0.05,0.8],           ["A","k","b"],            "F=A*(1-b*exp(-k*t))",             "Brody 1945",           "Empirical"),
    "Bertalanffy":           (m_bertalanffy,      [100.0,0.05,3.0],           ["A","k","n"],            "F=A*(1-exp(-k*t))^n",             "von Bertalanffy",      "Empirical"),
    "Pade Approximation":    (m_pade,             [0.0, 2.0, 0.02],           ["a0","a1","b1"],         "F=(a0+a1*t)/(1+b1*t)",           "Pade",                 "Empirical"),
    "hPLC Model":            (m_hPLC,             [0.05, 1.5, 1.0],           ["A","B","n"],            "F=100*(1-(1+A*t^n)^(-B))",       "Zuo 2014",             "Empirical"),
    "Compreg Model":         (m_compreg,          [0.05, 1.0, 2.0],           ["k","n","m"],            "F=100*(1-exp(-k*t^n))^m",         "Compressed release",   "Empirical"),
    "KP Modified":           (m_kpmod,            [10.0, 0.5, 0.01],          ["k","n","b"],            "F=k*t^n/(1+b*t)",                 "Modified KP",          "Empirical"),
    "Makoid-Banakar Mod.":   (m_mb_mod,           [10.0,0.5,0.01,0.0],        ["k","n","b","c"],        "F=k*t^n*exp(-b*t)+c",             "Extended MB",          "Empirical"),
    "Weibull-Sigmoid":       (m_weibull_sig,      [100.0,0.1,30.0,60.0],      ["A","k","t50","b"],      "Weibull x Logistic hybrid",       "Hybrid",               "Empirical"),
    # ── DDSolver / KinetDS models ─────────────────────────────────────────
    "Zero Order + F0":       (m_zero_order_f0,    [1.0, 5.0],                 ["k0","F0"],              "F=F0+k0*t (burst+linear)",        "DDSolver #303",        "Burst Release"),
    "First Order + Fmax":    (m_first_order_fmax, [0.05, 100.0],              ["k1","Fmax"],            "F=Fmax*(1-exp(-k1*t))",           "DDSolver #306",        "Burst Release"),
    "First Order + Tlag + Fmax": (m_first_order_tlag_fmax, [0.05,5.0,100.0], ["k1","tlag","Fmax"],     "F=Fmax*(1-exp(-k1*(t-tlag)))",    "DDSolver #307",        "Burst Release"),
    "Higuchi + F0":          (m_higuchi_f0,       [10.0, 5.0],                ["kH","F0"],              "F=F0+kH*sqrt(t)",                 "DDSolver #310",        "Burst Release"),
    "KP + F0":               (m_kp_f0,            [10.0, 0.5, 5.0],           ["k","n","F0"],           "F=F0+k*t^n",                      "DDSolver #313",        "Burst Release"),
    "Peppas-Sahlin 2":       (m_peppas_sahlin2,   [5.0, 1.0],                 ["k1","k2"],              "F=k1*sqrt(t)+k2*t",               "Peppas & Sahlin 1989", "Basic"),
    "Second Order":          (m_second_order,     [0.1],                      ["k"],                    "F=k*t^2",                         "DDSolver #302",        "Basic"),
    "Third Order":           (m_third_order,      [0.01],                     ["k"],                    "F=k*t^3",                         "DDSolver",             "Basic"),
    "Michaelis-Menten":      (m_michaelis_menten, [100.0, 20.0],              ["Qmax","km"],            "F=Qmax*t/(km+t)",                 "KinetDS 2012",         "Basic"),
    "Hixson-Crowell + Lag":  (m_hixson_tlag,      [0.05, 5.0],                ["ks","tlag"],            "M0^(1/3)-M^(1/3)=ks*(t-tlag)",   "DDSolver #315",        "Lag-Time"),
    "Logistic 1 (DDSolver)": (m_logistic1_dds,    [1.0, 1.0],                 ["alpha","beta"],         "F=100*exp(a+b*log(t))/(1+...)",   "DDSolver #332",        "Sigmoid"),
    "Logistic 2 (DDSolver)": (m_logistic2_dds,    [1.0, 1.0, 100.0],          ["alpha","beta","Fmax"],  "F=Fmax*exp(a+b*log(t))/(1+...)",  "DDSolver #333",        "Sigmoid"),
    "Gompertz 1 (DDSolver)": (m_gompertz1_dds,    [2.0, 1.0],                 ["alpha","beta"],         "F=100*exp(-exp(a-b*log(t)))",     "DDSolver #335",        "Sigmoid"),
    "Gompertz 2 (DDSolver)": (m_gompertz2_dds,    [2.0, 1.0, 100.0],          ["alpha","beta","Fmax"],  "F=Fmax*exp(-exp(a-b*log(t)))",    "DDSolver #336",        "Sigmoid"),
    "Probit 1 (DDSolver)":   (m_probit1_dds,      [1.5, 0.5],                 ["alpha","beta"],         "F=100*Phi(a+b*log10(t))",         "DDSolver #339",        "Sigmoid"),
}

CATEGORIES = ["Basic","Lag-Time","Burst Release","Multi-Phase","Sigmoid","Fractal","Empirical"]

# -- Physical parameter bounds (lower, upper) per model --
# Cömert ama fiziksel aralıklar: hız sabitleri ≥0, salım üsteli (0,20), lag ≥0,
# max salım ≤120%, fraksiyonlar (0,1). Listede OLMAYAN modeller sınırsız kalır
# (regresyon yok). Belirsiz işaretli katsayılı modeller (Quadratic, Pade,
# Peppas-Sahlin, log-logistic DDSolver) bilerek dışarıda bırakıldı.
_INF = np.inf
MODEL_BOUNDS = {
    "Zero Order":              ([0.0],                 [_INF]),
    "First Order":             ([0.0],                 [_INF]),
    "Higuchi":                 ([0.0],                 [_INF]),
    "Hixson-Crowell":          ([0.0],                 [_INF]),
    "Korsmeyer-Peppas":        ([0.0, 0.0],            [_INF, 20.0]),
    "Hopfenberg":              ([0.0, 0.0],            [_INF, 20.0]),
    "Baker-Lonsdale":          ([0.0],                 [_INF]),
    "Makoid-Banakar":          ([0.0, 0.0, 0.0],       [_INF, 20.0, _INF]),
    "Weibull":                 ([1e-9, 0.0, 0.0],      [_INF, _INF, _INF]),
    "Gompertz":                ([0.0, 0.0, 0.0],       [_INF, _INF, _INF]),
    "Logistic":                ([0.0, 0.0, 0.0],       [_INF, _INF, _INF]),
    "Probit":                  ([-_INF, 1e-9, 0.0],    [_INF, _INF, _INF]),
    "Weibull (No Lag)":        ([1e-9, 0.0],           [_INF, _INF]),
    "KP + Lag":                ([0.0, 0.0, 0.0],       [_INF, 20.0, _INF]),
    "First Order + Lag":       ([0.0, 0.0],            [_INF, _INF]),
    "Zero Order + Lag":        ([0.0, 0.0],            [_INF, _INF]),
    "Higuchi + Lag":           ([0.0, 0.0],            [_INF, _INF]),
    "Probit Log":              ([-_INF, 1e-9, 0.0],    [_INF, _INF, _INF]),
    "Double Exponential":      ([0.0, 0.0, 0.0, 0.0],  [_INF, _INF, _INF, _INF]),
    "Triple Exponential":      ([0.0]*6,               [_INF]*6),
    "Power-Exponential":       ([0.0, 0.0, 0.0],       [_INF, _INF, 20.0]),
    "Biexp. Absorption":       ([0.0, 0.0, 0.0],       [_INF, _INF, _INF]),
    "Gallagher-Corrigan":      ([0.0, 0.0, 0.0, 0.0],  [_INF, _INF, _INF, _INF]),
    "Combined Higuchi+FO":     ([0.0, 0.0, 0.0],       [_INF, _INF, 1.0]),
    "Henriksen":               ([0.0, 0.0, 0.0],       [_INF, _INF, _INF]),
    "Modified Gompertz":       ([0.0, 0.0, 0.0],       [_INF, _INF, _INF]),
    "Richards":                ([0.0, 0.0, 1e-9, 0.0], [_INF, _INF, _INF, _INF]),
    "Log-Normal":              ([-_INF, 1e-9, 0.0],    [_INF, _INF, _INF]),
    "Hill Equation":           ([0.0, 1e-9, 0.0],      [_INF, _INF, 20.0]),
    "Dose-Response":           ([0.0, 0.0, 1e-9, 0.0], [_INF, _INF, _INF, 20.0]),
    "Fractal First Order":     ([0.0, 0.0],            [_INF, 5.0]),
    "Stretched Exponential":   ([0.0, 0.0, 1e-9],      [_INF, 5.0, _INF]),
    "Fractal Weibull":         ([1e-9, 0.0, 0.0],      [_INF, _INF, _INF]),
    "Exponential Assoc.":      ([0.0, 0.0],            [_INF, _INF]),
    "Hyperbolic":              ([0.0, 1e-9],           [_INF, _INF]),
    "Brody Growth":            ([0.0, 0.0, -_INF],     [_INF, _INF, _INF]),
    "Bertalanffy":             ([0.0, 0.0, 1e-9],      [_INF, _INF, _INF]),
    "hPLC Model":              ([0.0, 0.0, 0.0],       [_INF, _INF, 20.0]),
    "Compreg Model":           ([0.0, 0.0, 0.0],       [_INF, 20.0, _INF]),
    "KP Modified":             ([0.0, 0.0, 0.0],       [_INF, 20.0, _INF]),
    "Makoid-Banakar Mod.":     ([0.0, 0.0, 0.0, -_INF],[_INF, 20.0, _INF, _INF]),
    "Weibull-Sigmoid":         ([0.0, 0.0, 0.0, 1e-9], [_INF, _INF, _INF, _INF]),
    "Zero Order + F0":         ([0.0, 0.0],            [_INF, _INF]),
    "First Order + Fmax":      ([0.0, 0.0],            [_INF, 120.0]),
    "First Order + Tlag + Fmax":([0.0, 0.0, 0.0],      [_INF, _INF, 120.0]),
    "Higuchi + F0":            ([0.0, 0.0],            [_INF, _INF]),
    "KP + F0":                 ([0.0, 0.0, 0.0],       [_INF, 20.0, _INF]),
    "Second Order":            ([0.0],                 [_INF]),
    "Third Order":             ([0.0],                 [_INF]),
    "Michaelis-Menten":        ([0.0, 1e-9],           [_INF, _INF]),
    "Hixson-Crowell + Lag":    ([0.0, 0.0],            [_INF, _INF]),
    "Logistic 2 (DDSolver)":   ([-_INF, -_INF, 0.0],   [_INF, _INF, 120.0]),
    "Gompertz 2 (DDSolver)":   ([-_INF, -_INF, 0.0],   [_INF, _INF, 120.0]),
}

# -- Weighted least squares: weighting scheme → (sigma, absolute_sigma) --
def _build_sigma(y, weight_scheme="none", sd=None):
    """Translate a weighting scheme into curve_fit's (sigma, absolute_sigma).

    scipy curve_fit minimizes Σ ((y_i - f_i)/sigma_i)². Weighted LS with weights
    w_i (minimize Σ w_i (y_i - f_i)²) therefore needs sigma_i = 1/sqrt(w_i):
      - "none"  → sigma=None, absolute_sigma=False  (UNCHANGED current behavior)
      - "1/y"   → w=1/y   → sigma_i = sqrt(|y_i|); absolute_sigma=False
      - "1/y2"  → w=1/y²  → sigma_i = |y_i|;       absolute_sigma=False
      - "1/sd"  → w=1/SD² → sigma_i = SD_i;        absolute_sigma=True
                  (real measurement std devs → pcov reflects true uncertainty)
    Any sigma_i <= 0 or non-finite is floored to eps = 1e-8 * max(|y|) so no point
    gets infinite weight and curve_fit never divides by zero. For "1/sd": if sd is
    None, the wrong length, or all ~0, FALL BACK to "none".

    Returns (sigma, absolute_sigma, effective_scheme).
    """
    ya = np.asarray(y, dtype=float)
    eps = 1e-8 * float(np.max(np.abs(ya))) if ya.size else 1e-8
    if eps <= 0.0:
        eps = 1e-8

    def _floor(arr):
        arr = np.asarray(arr, dtype=float)
        bad = ~np.isfinite(arr) | (arr <= 0.0)
        arr[bad] = eps
        return arr

    if weight_scheme == "1/y":
        return _floor(np.sqrt(np.abs(ya))), False, "1/y"
    if weight_scheme == "1/y2":
        return _floor(np.abs(ya)), False, "1/y2"
    if weight_scheme == "1/sd":
        sda = np.asarray(sd, dtype=float).ravel() if sd is not None else None
        if (sda is None or sda.size != ya.size
                or not np.any(np.isfinite(sda) & (sda > 0.0))):
            return None, False, "none"  # fall back, do not crash
        return _floor(sda), True, "1/sd"
    return None, False, "none"


# -- Per-parameter 95% confidence intervals (additive; DDSolver parity) --
def _compute_param_ci(pnames, popt, pcov, n_valid_points, n_params):
    """Build a {param: {value, se, ci_low, ci_high}} dict from curve_fit's pcov.
    Uses the t-distribution (NOT a fixed 1.96) with dof = n_valid_points -
    n_params, correct for the small n typical of dissolution. A saturated fit
    (dof <= 0) has no residual information to estimate uncertainty, so its CIs are
    reported as None rather than a misleading interval. Non-finite or negative
    variances likewise yield se/ci_low/ci_high = None instead of crashing/NaN."""
    dof = int(n_valid_points) - int(n_params)
    tval = float(_student_t.ppf(0.975, dof)) if dof > 0 else None
    # Only trust pcov if there is residual dof AND it is a finite 2-D square diagonal.
    cov = np.asarray(pcov, dtype=float) if pcov is not None else None
    have_cov = (dof > 0 and cov is not None and cov.ndim == 2
                and cov.shape[0] == cov.shape[1] and cov.shape[0] == len(popt))
    out = {}
    for i, pn in enumerate(pnames):
        val = float(popt[i])
        se = ci_low = ci_high = None
        if have_cov:
            var = cov[i, i]
            if np.isfinite(var) and var >= 0.0:
                se = float(np.sqrt(var))
                ci_low = float(val - tval * se)
                ci_high = float(val + tval * se)
        out[pn] = {"value": val, "se": se, "ci_low": ci_low, "ci_high": ci_high}
    return out


# -- Residual diagnostics (additive; DDSolver parity) --
def _residual_diagnostics(yv, ypv):
    """Residual analysis on the valid-masked fit points (yv observed, ypv fitted).

    Returns the residuals (observed - fitted) and fitted values (same order, so the
    UI can draw residual-vs-fitted without re-deriving anything), plus two goodness
    tests: Shapiro-Wilk normality p-value and a Wald-Wolfowitz runs-test p-value on
    the signs of the residuals about 0 (randomness/autocorrelation). p-values are
    None when not computable (too few points, zero variance, all-one-sign, etc.)."""
    residuals = [float(a) - float(b) for a, b in zip(yv, ypv)]
    fitted = [float(b) for b in ypv]
    n = len(residuals)

    # Shapiro-Wilk: only for 3<=n<=5000 and non-degenerate (zero variance raises).
    shapiro_p = None
    try:
        if 3 <= n <= 5000 and len(set(residuals)) > 1:
            shapiro_p = float(_shapiro(residuals).pvalue)
    except Exception:
        shapiro_p = None

    # Wald-Wolfowitz runs test on residual signs about 0 (zeros dropped).
    runs_p = None
    try:
        signs = [1 if r > 0 else -1 for r in residuals if r != 0]
        n1 = signs.count(1)
        n2 = signs.count(-1)
        if n1 == 0 or n2 == 0 or (n1 + n2) < 2:
            runs_p = None
        else:
            runs = 1 + sum(1 for a, b in zip(signs[:-1], signs[1:]) if a != b)
            mu = 2.0 * n1 * n2 / (n1 + n2) + 1.0
            var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)
                   / (((n1 + n2) ** 2) * (n1 + n2 - 1)))
            if var <= 0:
                runs_p = None
            else:
                z = (runs - mu) / np.sqrt(var)
                p = float(2.0 * sp_norm.sf(abs(z)))
                runs_p = float(min(1.0, max(0.0, p)))
    except Exception:
        runs_p = None

    return {"residuals": residuals, "fitted": fitted,
            "shapiro_p": shapiro_p, "runs_p": runs_p, "n_resid": n}


# -- Tx% release-time interpolation (additive; DDSolver parity) --
def _compute_tx(func, popt, t_lo, t_hi, targets=(25.0, 50.0, 80.0, 90.0), n=400):
    """Time at which the fitted curve first reaches each target % release.

    Builds a dense grid over the MEASURED time range [t_lo, t_hi] (interpolation
    only — never extrapolation), clips predictions to [0,100], and for each
    target finds the FIRST UPWARD crossing, linearly interpolating the time:
        t_cross = td[i-1] + (target-yd[i-1])*(td[i]-td[i-1])/(yd[i]-yd[i-1]).
    Edge cases: if the curve is already >= target at t_lo, report t_lo; if the
    target is never reached within the range, report None. Non-finite predictions
    break the search (no crossing inferred across a NaN gap). Returns
    {"25":..,"50":..,"80":..,"90":..} with float|None values; any failure → all-None."""
    keys = [str(int(x)) for x in targets]
    try:
        td = np.linspace(float(t_lo), float(t_hi), n)
        yd = np.asarray(func(td, *popt), dtype=float)
        yd = np.clip(yd, 0.0, 100.0)
        out = {}
        for target, key in zip(targets, keys):
            tgt = float(target)
            t_cross = None
            # Already at/above target at the start of the measured range.
            if np.isfinite(yd[0]) and yd[0] >= tgt:
                t_cross = float(td[0])
            else:
                for i in range(1, len(yd)):
                    a, b = yd[i - 1], yd[i]
                    if not (np.isfinite(a) and np.isfinite(b)):
                        continue  # break across a non-finite gap (no crossing)
                    if a < tgt <= b:
                        denom = b - a
                        if denom != 0.0:
                            t_cross = float(td[i - 1] + (tgt - a) * (td[i] - td[i - 1]) / denom)
                        else:
                            t_cross = float(td[i - 1])
                        break
            out[key] = t_cross
        return out
    except Exception:
        return {k: None for k in keys}


# -- Parameter-substituted fitted equation (additive; DDSolver parity) --
def _fitted_equation(eq_template, pnames, popt):
    """Substitute fitted parameter VALUES into the symbolic equation template.

    e.g. "F=k*t^n" with k=10.6, n=0.34 → "F=10.6*t^0.34" (like DDSolver). Each
    parameter NAME is replaced by its value formatted to 4 significant figures.
    Replacement is WHOLE-TOKEN ONLY (boundary lookarounds) and parameter names are
    processed LONGEST-FIRST so e.g. "k1" is substituted before "k" and a 1-char
    name like "A" never clobbers "Amax". The time variable "t", math tokens
    (exp/log/sqrt/Phi), digits and "100" are left untouched. Descriptive templates
    with no real formula (e.g. "Biphasic burst+slow") simply pass through. Any
    error → return the original template unchanged."""
    try:
        s = str(eq_template)
        order = sorted(range(len(pnames)), key=lambda i: len(str(pnames[i])), reverse=True)
        for i in order:
            name = str(pnames[i])
            value = f"{float(popt[i]):.4g}"
            s = re.sub(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])',
                       value, s)
        return s
    except Exception:
        return eq_template


# -- Curve-fitting engine --
def fit_model(t, y, name, weight_scheme="none", sd=None):
    func, p0, pnames, eq, ref, cat = MODEL_DEFS[name]
    bnds = MODEL_BOUNDS.get(name)
    # Weighted least squares: build curve_fit's sigma/absolute_sigma. "none" →
    # (None, False) keeps the unweighted behavior byte-for-byte identical.
    sigma, absolute_sigma, eff_scheme = _build_sigma(y, weight_scheme, sd)
    try:
        if bnds is not None:
            lo, hi = bnds
            # p0'ı sınırların içine çek (curve_fit p0 ∈ [lo,hi] ister)
            p0c = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]
            try:
                popt, pcov = curve_fit(func, t, y, p0=p0c, bounds=(lo, hi), max_nfev=25000,
                                       sigma=sigma, absolute_sigma=absolute_sigma)
                bounds_enforced = True   # bounded fit succeeded
            except Exception:
                # Bounded fit başarısızsa eski davranışa (sınırsız) düş — regresyon yok
                popt, pcov = curve_fit(func, t, y, p0=p0, maxfev=25000,
                                       sigma=sigma, absolute_sigma=absolute_sigma)
                bounds_enforced = False  # fell back to unbounded fit
        else:
            popt, pcov = curve_fit(func, t, y, p0=p0, maxfev=25000,
                                   sigma=sigma, absolute_sigma=absolute_sigma)
            bounds_enforced = None       # model has no bounds to enforce
        yp = np.array(func(t,*popt), dtype=float)
        nan_fraction = float(np.mean(np.isnan(yp)))
        valid = ~np.isnan(yp)
        if valid.sum() < 3: raise ValueError("Too few valid predictions")
        tv,yv,ypv = t[valid],y[valid],yp[valid]
        np_ = len(popt)
        param_ci = _compute_param_ci(pnames, popt, pcov, int(valid.sum()), np_)
        return {"success":True,"name":name,"category":cat,
                "bounds_enforced":bounds_enforced,"nan_fraction":nan_fraction,
                "r2":r2s(yv,ypv),"r2adj":r2adj(yv,ypv,np_),
                "aic":aic_fn(yv,ypv,np_),"aicc":aicc_fn(yv,ypv,np_),
                "bic":bic_fn(yv,ypv,np_),"msc":msc_fn(yv,ypv,np_),
                "rmse":rmse_fn(yv,ypv),
                "params":dict(zip(pnames,popt)),"param_ci":param_ci,"yp":yp,
                "n_params":np_,"equation":eq,
                "equation_fitted":_fitted_equation(eq,pnames,popt),
                "reference":ref,"error":None,
                "weight_scheme":eff_scheme,
                "diagnostics":_residual_diagnostics(yv,ypv),
                "tx":_compute_tx(func,popt,float(np.min(t)),float(np.max(t)))}
    except Exception as e:
        return {"success":False,"name":name,"category":cat,
                "bounds_enforced":None,"nan_fraction":None,
                "r2":np.nan,"r2adj":np.nan,"aic":np.nan,"aicc":np.nan,
                "bic":np.nan,"msc":np.nan,"rmse":np.nan,
                "params":{},"param_ci":{},"yp":np.full(len(t),np.nan),
                "n_params":len(p0),"equation":MODEL_DEFS[name][3],
                "equation_fitted":None,
                "reference":MODEL_DEFS[name][4],"error":str(e),
                "diagnostics":None,"tx":None}
