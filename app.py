import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import io

# --- 1. CONFIG & THEME ---
st.set_page_config(page_title="SmartDissolve AI", layout="wide")

# Sarı-Lacivert Sidebar Header
st.sidebar.markdown(
    """
    <div style="background-color: #002D72; padding: 15px; border-radius: 10px; border: 3px solid #FFD700; margin-bottom: 20px;">
        <h1 style="color: #FFD700; margin: 0; font-size: 1.8rem; text-align: center;">💊 SmartDissolve AI</h1>
        <p style="color: white; margin: 0; font-size: 0.8rem; text-align: center;">Predictive Dissolution Suite</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# --- 2. KINETIC MODELS ENGINE ---
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def korsmeyer_peppas(t, k, n): return k * (t**n)
def hixson_crowell(t, k): return 100 * (1 - (1 - np.maximum(k * t, 0)/100)**3)
def hopfenberg(t, k, n): return 100 * (1 - (1 - np.maximum(k * t, 0)/100)**n)
def makoid_banakar(t, k, n, c): return k * (t**n) * np.exp(-c * t)
def peppas_sahlin(t, k1, k2, m): return 100 * (k1 * (t**m) + k2 * (t**(2*m)))
def gompertz(t, xmax, k, i): return xmax * np.exp(-np.exp(k * (t - i)))
def weibull(t, a, b, td): return 100 * (1 - np.exp(- (np.maximum(t - td, 0)**b) / a))

# --- 3. CORE ANALYTICS FUNCTIONS ---
def process_data(file):
    if file is None: return None
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        mask = ~np.isnan(t)
        return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask], "n": v.shape[1]}
    except: return None

def calculate_aic(n, rss, p):
    if n <= p or rss <= 0: return 9999
    return n * np.log(rss/n) + 2 * p

def calculate_independent(t, q):
    dt = np.diff(t, prepend=0)
    de = (np.cumsum(q * dt)[-1] / (t[-1] * 100)) * 100
    mdt = np.sum((t - (dt/2)) * np.diff(q, prepend=0)) / q[-1] if q[-1] > 0 else 0
    return de, mdt

# --- 4. UI & NAVIGATION ---
selected_lang = st.sidebar.selectbox("Dil / Language:", ["Türkçe", "English"])
menu = st.sidebar.radio(
    "Analytical Suite",
    ["📈 Salım Profilleri", "🧮 Kinetik Model Fitting", "🧬 IVIVC Analizi", "📊 f1 & f2 Benzerlik Analizi"]
)

test_files = st.sidebar.file_uploader("Test Verileri (Çoklu Yükleme)", type=['xlsx', 'csv'], accept_multiple_files=True)
ref_file = st.sidebar.file_uploader("Referans Verisi (Opsiyonel)", type=['xlsx', 'csv'])

# --- 5. MAIN LOGIC ---
if test_files:
    data_registry = {f.name: process_data(f) for f in test_files if process_data(f)}
    
    if data_registry:
        selected_file = st.selectbox("Analiz Edilecek Seri:", list(data_registry.keys()))
        active = data_registry[selected_file]
        ref_data = process_data(ref_file)
        
        t_data, q_data = active["t"], active["mean"]
        de, mdt = calculate_independent(t_data, q_data)

        # MODULE: RELEASE PROFILES
        if menu == "📈 Salım Profilleri":
            st.subheader(f"📊 {selected_file} - Profil Analizi")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.errorbar(t_data, q_data, yerr=active["std"], fmt='-ok', label="Test", color='#002D72', capsize=4)
            if ref_data:
                ax.errorbar(ref_data["t"], ref_data["mean"], yerr=ref_data["std"], fmt='--sr', label="Referans", color='#FF4B4B')
            ax.set_xlabel("Zaman (dk)"); ax.set_ylabel("Kümülatif Salım (%)"); ax.legend(); ax.grid(alpha=0.2)
            st.pyplot(fig)
            
            c1, c2 = st.columns(2)
            c1.metric("Dissolution Efficiency (DE %)", f"{de:.2f}%")
            c2.metric("Mean Dissolution Time (MDT)", f"{mdt:.2f} dk")

        # MODULE: KINETIC FITTING
        elif menu == "🧮 Kinetik Model Fitting":
            st.subheader(f"🧮 {selected_file} - Kinetik Modelleme")
            models = {
                "Sıfır Derece": zero_order, "Birinci Derece": first_order, 
                "Higuchi": higuchi, "Korsmeyer-Peppas": korsmeyer_peppas,
                "Hixson-Crowell": hixson_crowell, "Gompertz": gompertz,
                "Hopfenberg": hopfenberg, "Weibull": weibull
            }
            
            fit_results = []
            fig_fit, ax_fit = plt.subplots(figsize=(10, 5))
            ax_fit.scatter(t_data, q_data, color='black', label="Deneysel Veri", zorder=5)
            
            for name, func in models.items():
                try:
                    p_count = func.__code__.co_argcount - 1
                    popt, _ = curve_fit(func, t_data, q_data, maxfev=5000)
                    q_pred = func(t_data, *popt)
                    r2 = r2_score(q_data, q_pred)
                    rss = np.sum((q_data - q_pred)**2)
                    aic = calculate_aic(len(t_data), rss, p_count)
                    fit_results.append({"Model": name, "R2": round(r2, 4), "AIC": round(aic, 2)})
                    ax_fit.plot(t_data, q_pred, label=f"{name} (R2: {r2:.3f})", alpha=0.7)
                except: continue
            
            ax_fit.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig_fit)
            st.table(pd.DataFrame(fit_results).sort_values("R2", ascending=False))

        # MODULE: SIMILARITY
        elif menu == "📊 f1 & f2 Benzerlik Analizi":
            if ref_data:
                st.subheader("f1 & f2 Benzerlik Analizi Sonuçları")
                # Zaman noktası eşleşen verileri al
                common_idx = np.intersect1d(t_data, ref_data["t"], return_indices=True)
                if len(common_idx[0]) > 0:
                    T = q_data[common_idx[1]]
                    R = ref_data["mean"][common_idx[2]]
                    n = len(common_idx[0])
                    f1 = (np.sum(np.abs(R - T)) / np.sum(R)) * 100
                    f2 = 50 * np.log10((1 + (1/n) * np.sum((R - T)**2))**-0.5 * 100)
                    
                    c1, c2 = st.columns(2)
                    c1.metric("f1 (Farklılık Faktörü)", f"{f1:.2f}")
                    c2.metric("f2 (Benzerlik Faktörü)", f"{f2:.2f}")
                    if f2 >= 50: st.success("✅ Profiller Benzer")
                    else: st.error("❌ Profiller Benzer Değil")
                else: st.error("Hata: Test ve Referans verilerinin zaman noktaları (dakika) eşleşmiyor.")
            else: st.warning("⚠️ Benzerlik analizi için lütfen sol panelden bir 'Referans Verisi' yükleyin.")

    # SIDEBAR DOWNLOAD BUTTON (MOCK)
    st.sidebar.divider()
    if st.sidebar.button("📥 Excel Raporunu Finalize Et"):
        st.sidebar.info("Rapor oluşturma motoru (xlsxwriter) hazır.")

else:
    st.info("👈 Analizi başlatmak için lütfen sol taraftan Test Verilerini yükleyin.")
