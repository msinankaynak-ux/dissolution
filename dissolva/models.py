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

# NOTE: The kinetic engine (62 model equations, fitting pipeline, bootstrap)
# lives server-side in the private backend (services/engine.py) and is reached
# via dissolva/engine_client.py. This module keeps only public-safe metadata
# (model registry without functions) and standard published metrics.

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


MODEL_DEFS = {
    'Zero Order': (None, [1.0], ['k0'], 'F=k0*t', 'Wagner 1969', 'Basic'),
    'First Order': (None, [0.05], ['k1'], 'F=100*(1-exp(-k1*t))', 'Wagner 1969', 'Basic'),
    'Higuchi': (None, [10.0], ['kH'], 'F=kH*sqrt(t)', 'Higuchi 1961', 'Basic'),
    'Hixson-Crowell': (None, [0.05], ['ks'], 'M0^(1/3)-M^(1/3)=ks*t', 'Hixson 1931', 'Basic'),
    'Korsmeyer-Peppas': (None, [10.0, 0.5], ['k', 'n'], 'F=k*t^n', 'Korsmeyer 1983', 'Basic'),
    'Hopfenberg': (None, [0.02, 2.0], ['kHB', 'n'], 'F=100*[1-(1-kHB*t)^n]', 'Hopfenberg 1976', 'Basic'),
    'Baker-Lonsdale': (None, [0.001], ['kBL'], '3/2*[1-(1-F)^(2/3)]-F=kBL*t', 'Baker 1974', 'Basic'),
    'Makoid-Banakar': (None, [10.0, 0.5, 0.01], ['kMB', 'nMB', 'bMB'], 'F=kMB*t^nMB*exp(-bMB*t)', 'Makoid 1993', 'Basic'),
    'Peppas-Sahlin': (None, [5.0, 1.0, 0.5], ['k1', 'k2', 'm'], 'F=k1*t^m + k2*t^2m', 'Peppas 1989', 'Basic'),
    'Weibull': (None, [50.0, 1.0, 0.0], ['a', 'b', 'Td'], 'F=100*(1-exp(-((t-Td)^b)/a))', 'Weibull 1951', 'Basic'),
    'Gompertz': (None, [100.0, 5.0, 0.1], ['A', 'b', 'k'], 'F=A*exp(-b*exp(-k*t))', 'Gompertz 1825', 'Basic'),
    'Logistic': (None, [100.0, 0.1, 30.0], ['A', 'k', 't50'], 'F=A/(1+exp(-k*(t-t50)))', 'Pressman 1994', 'Basic'),
    'Quadratic': (None, [-0.01, 1.0, 0.0], ['a', 'b', 'c'], 'F=a*t^2 + b*t + c', 'Polli 1997', 'Basic'),
    'Probit': (None, [30.0, 15.0, 100.0], ['mu', 'sigma', 'A'], 'F=A*Phi((t-mu)/sigma)', 'Shah 1998', 'Basic'),
    'Weibull (No Lag)': (None, [50.0, 1.0], ['a', 'b'], 'F=100*(1-exp(-t^b/a))', 'Weibull 1951', 'Lag-Time'),
    'KP + Lag': (None, [10.0, 0.5, 5.0], ['k', 'n', 'tlag'], 'F=k*(t-tlag)^n', 'Modified KP', 'Lag-Time'),
    'First Order + Lag': (None, [0.05, 5.0], ['k1', 'tlag'], 'F=100*(1-exp(-k1*(t-tlag)))', 'Modified FO', 'Lag-Time'),
    'Zero Order + Lag': (None, [1.0, 5.0], ['k0', 'tlag'], 'F=k0*(t-tlag)', 'Modified ZO', 'Lag-Time'),
    'Higuchi + Lag': (None, [10.0, 5.0], ['kH', 'tlag'], 'F=kH*sqrt(t-tlag)', 'Modified Higuchi', 'Lag-Time'),
    'Probit Log': (None, [1.5, 0.5, 100.0], ['mu', 'sigma', 'A'], 'F=A*Phi((log10(t)-mu)/sigma)', 'Shah 1998', 'Lag-Time'),
    'Double Exponential': (None, [60.0, 0.05, 40.0, 0.005], ['A1', 'k1', 'A2', 'k2'], 'F=A1*(1-e^-k1t)+A2*(1-e^-k2t)', 'Empirical', 'Multi-Phase'),
    'Triple Exponential': (None, [40.0, 0.1, 40.0, 0.02, 20.0, 0.005], ['A1', 'k1', 'A2', 'k2', 'A3', 'k3'], 'F=sum Ai*(1-e^-kit)', 'Empirical', 'Multi-Phase'),
    'Power-Exponential': (None, [100.0, 0.05, 1.2], ['A', 'k', 'n'], 'F=A*(1-exp(-k*t^n))', 'Zhang 2010', 'Multi-Phase'),
    'Biexp. Absorption': (None, [1.0, 0.2, 0.05], ['Fr', 'ka', 'k'], 'F=Fr*100*ka/(ka-k)*(exp(-k*t)-exp(-ka*t))', 'PK-based', 'Multi-Phase'),
    'Gallagher-Corrigan': (None, [100.0, 0.05, 0.02, 60.0], ['Amax', 'k1', 'k2', 'tmax'], 'Biphasic burst+slow', 'Gallagher 2000', 'Multi-Phase'),
    'Combined Higuchi+FO': (None, [10.0, 0.05, 0.5], ['kH', 'k1', 'alpha'], 'F=alpha*kH*sqrt(t)+(1-alpha)*FO', 'Empirical', 'Multi-Phase'),
    'Henriksen': (None, [80.0, 0.1, 0.5], ['A', 'k1', 'k2'], 'F=A*(exp(-k1*t)-exp(-k2*t))', 'Henriksen et al.', 'Multi-Phase'),
    'Modified Gompertz': (None, [100.0, 0.1, 10.0], ['Amax', 'mu', 'lambda'], 'F=Amax*exp(-exp(mu*e/Amax*(lam-t)+1))', 'Zwietering 1990', 'Sigmoid'),
    'Richards': (None, [100.0, 0.05, 1.0, 30.0], ['A', 'k', 'n', 't50'], 'F=A*(1+exp(-k*(t-t50)))^(-1/n)', 'Richards 1959', 'Sigmoid'),
    '4-Parameter Logistic': (None, [0.0, 100.0, 0.1, 30.0], ['A', 'B', 'k', 't50'], 'F=A+(B-A)/(1+exp(-k*(t-t50)))', '4PL', 'Sigmoid'),
    'Log-Normal': (None, [3.5, 0.5, 100.0], ['mu', 'sigma', 'A'], 'F=A*Phi((ln(t)-mu)/sigma)', 'Statistical', 'Sigmoid'),
    'Hill Equation': (None, [100.0, 30.0, 1.5], ['Amax', 'k', 'n'], 'F=Amax*t^n/(k^n+t^n)', 'Hill 1910', 'Sigmoid'),
    'Dose-Response': (None, [0.0, 100.0, 30.0, 1.0], ['Emin', 'Emax', 'EC50', 'n'], 'F=Emin+(Emax-Emin)*t^n/(EC50^n+t^n)', 'Pharmacological', 'Sigmoid'),
    'Fractal First Order': (None, [0.05, 0.8], ['k', 'alpha'], 'F=100*(1-exp(-k*t^alpha))', 'Macheras 1995', 'Fractal'),
    'Stretched Exponential': (None, [100.0, 0.8, 30.0], ['A', 'beta', 'tau'], 'F=A*(1-exp(-(t/tau)^beta))', 'Kohlrausch 1854', 'Fractal'),
    'Fractal Weibull': (None, [30.0, 1.2, 0.0], ['alpha', 'beta', 'gamma'], 'F=100*(1-exp(-((t-gamma)/alpha)^beta))', 'Weibull 3P', 'Fractal'),
    'Exponential Assoc.': (None, [100.0, 0.05], ['A', 'k'], 'F=A*(1-exp(-k*t))', 'Empirical', 'Empirical'),
    'Hyperbolic': (None, [100.0, 20.0], ['Amax', 'k'], 'F=Amax*t/(k+t)', 'Empirical', 'Empirical'),
    'Linear-Exponential': (None, [2.0, 0.02, 0.0], ['A', 'k', 'b'], 'F=A*t*exp(-k*t)+b', 'Empirical', 'Empirical'),
    'Brody Growth': (None, [120.0, 0.05, 0.8], ['A', 'k', 'b'], 'F=A*(1-b*exp(-k*t))', 'Brody 1945', 'Empirical'),
    'Bertalanffy': (None, [100.0, 0.05, 3.0], ['A', 'k', 'n'], 'F=A*(1-exp(-k*t))^n', 'von Bertalanffy', 'Empirical'),
    'Pade Approximation': (None, [0.0, 2.0, 0.02], ['a0', 'a1', 'b1'], 'F=(a0+a1*t)/(1+b1*t)', 'Pade', 'Empirical'),
    'hPLC Model': (None, [0.05, 1.5, 1.0], ['A', 'B', 'n'], 'F=100*(1-(1+A*t^n)^(-B))', 'Zuo 2014', 'Empirical'),
    'Compreg Model': (None, [0.05, 1.0, 2.0], ['k', 'n', 'm'], 'F=100*(1-exp(-k*t^n))^m', 'Compressed release', 'Empirical'),
    'KP Modified': (None, [10.0, 0.5, 0.01], ['k', 'n', 'b'], 'F=k*t^n/(1+b*t)', 'Modified KP', 'Empirical'),
    'Makoid-Banakar Mod.': (None, [10.0, 0.5, 0.01, 0.0], ['k', 'n', 'b', 'c'], 'F=k*t^n*exp(-b*t)+c', 'Extended MB', 'Empirical'),
    'Weibull-Sigmoid': (None, [100.0, 0.1, 30.0, 60.0], ['A', 'k', 't50', 'b'], 'Weibull x Logistic hybrid', 'Hybrid', 'Empirical'),
    'Zero Order + F0': (None, [1.0, 5.0], ['k0', 'F0'], 'F=F0+k0*t (burst+linear)', 'DDSolver #303', 'Burst Release'),
    'First Order + Fmax': (None, [0.05, 100.0], ['k1', 'Fmax'], 'F=Fmax*(1-exp(-k1*t))', 'DDSolver #306', 'Burst Release'),
    'First Order + Tlag + Fmax': (None, [0.05, 5.0, 100.0], ['k1', 'tlag', 'Fmax'], 'F=Fmax*(1-exp(-k1*(t-tlag)))', 'DDSolver #307', 'Burst Release'),
    'Higuchi + F0': (None, [10.0, 5.0], ['kH', 'F0'], 'F=F0+kH*sqrt(t)', 'DDSolver #310', 'Burst Release'),
    'KP + F0': (None, [10.0, 0.5, 5.0], ['k', 'n', 'F0'], 'F=F0+k*t^n', 'DDSolver #313', 'Burst Release'),
    'Peppas-Sahlin 2': (None, [5.0, 1.0], ['k1', 'k2'], 'F=k1*sqrt(t)+k2*t', 'Peppas & Sahlin 1989', 'Basic'),
    'Second Order': (None, [0.1], ['k'], 'F=k*t^2', 'DDSolver #302', 'Basic'),
    'Third Order': (None, [0.01], ['k'], 'F=k*t^3', 'DDSolver', 'Basic'),
    'Michaelis-Menten': (None, [100.0, 20.0], ['Qmax', 'km'], 'F=Qmax*t/(km+t)', 'KinetDS 2012', 'Basic'),
    'Hixson-Crowell + Lag': (None, [0.05, 5.0], ['ks', 'tlag'], 'M0^(1/3)-M^(1/3)=ks*(t-tlag)', 'DDSolver #315', 'Lag-Time'),
    'Logistic 1 (DDSolver)': (None, [1.0, 1.0], ['alpha', 'beta'], 'F=100*exp(a+b*log(t))/(1+...)', 'DDSolver #332', 'Sigmoid'),
    'Logistic 2 (DDSolver)': (None, [1.0, 1.0, 100.0], ['alpha', 'beta', 'Fmax'], 'F=Fmax*exp(a+b*log(t))/(1+...)', 'DDSolver #333', 'Sigmoid'),
    'Gompertz 1 (DDSolver)': (None, [2.0, 1.0], ['alpha', 'beta'], 'F=100*exp(-exp(a-b*log(t)))', 'DDSolver #335', 'Sigmoid'),
    'Gompertz 2 (DDSolver)': (None, [2.0, 1.0, 100.0], ['alpha', 'beta', 'Fmax'], 'F=Fmax*exp(-exp(a-b*log(t)))', 'DDSolver #336', 'Sigmoid'),
    'Probit 1 (DDSolver)': (None, [1.5, 0.5], ['alpha', 'beta'], 'F=100*Phi(a+b*log10(t))', 'DDSolver #339', 'Sigmoid'),
}

CATEGORIES = ['Basic', 'Lag-Time', 'Burst Release', 'Multi-Phase', 'Sigmoid', 'Fractal', 'Empirical']
