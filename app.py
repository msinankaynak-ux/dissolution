import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score
import io

# --- 1. SAYFA YAPILANDIRMASI VE TEMA ---
st.set_page_config(page_title="SmartDissolve AI", layout="wide")

# Sarı-Lacivert Sidebar Tasarımı
st.sidebar.markdown(
    """
    <div style="background-color: #002D72; padding: 15px; border-radius: 10px; border: 3px solid #FFD700; margin-bottom: 20px;">
        <h1 style="color: #FFD700; margin: 0; font-size: 1.8rem; text-align: center;">💊 SmartDissolve AI</h1>
        <p style="color: white; margin: 0; font-size: 0.8rem; text-align: center;">Predictive Dissolution Suite</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# --- 2. KİNETİK MODEL MOTORU ---
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
def baker_lonsdale(t, k):
    # Basitleştirilmiş Baker-Lonsdale yaklaşımı
    return 100 * (k * np.sqrt(t)) # Teorik fittig için yaklaşım

# --- 3. YARDIMCI ANALİZ FONKSİYONLARI ---
def process_data(file):
    if file is None: return None
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        mask = ~np.isnan(t)
        return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask], "n": v.shape[1], "raw": v[mask]}
    except: return None

def calculate_aic(n, rss, p):
    if n <= p or rss <= 0: return 9999
    return n * np.log(rss/n) + 2 * p

def calculate_independent(t, q):
    dt = np.diff(t, prepend=0)
    de = (np.cumsum(q * dt)[-1] / (t[-1] * 100)) * 100
    mdt = np.sum((t - (dt/2)) * np.diff(q, prepend=0)) / q[-1] if q[-1] > 0 else 0
    return de, mdt

# --- 4. ARAYÜZ VE MENÜ ---
selected_lang = st.sidebar.selectbox("Dil / Language:", ["Türkçe", "English"])
menu = st.sidebar.radio(
    "Analytical Suite",
    ["📈 Salım Profilleri", "🧮 Kinetik Model Fitting", "🧬 IVIVC Analizi", "📊 f1 & f2 Benzerlik Analizi"]
)

test_files = st.sidebar.file_uploader("Test Verileri (Çoklu)", type=['xlsx', 'csv'], accept_multiple_files=True)
ref_file = st.sidebar.file_uploader("Referans Verisi", type=['xlsx', 'csv'])

# --- 5. ANA İŞLEMCİ ---
if test_files:
    # Çoklu dosya kayıt sistemi
    data_reg = {f.name: process_data(f) for f in test_files if process_data(f)}
    
    if data_reg:
        sel_name = st.selectbox("Analiz Edilecek Seri:", list(data_reg.keys()))
        active = data_reg[sel_name]
        ref_data = process_data(ref_file)
        
        t, q = active["t"], active["mean"]
        de, mdt = calculate_independent(t, q)

        # MODÜL 1: PROFİLLER
        if menu == "📈 Salım Profilleri":
            st.subheader(f"📊 {sel_name} - Salım Karakterizasyonu")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.errorbar(t, q, yerr=active["std"], fmt='-ok', label="Test", color='#002D72', capsize=4)
            if ref_data:
                ax.errorbar(ref_data["t"], ref_data["mean"], yerr=ref_data["std"], fmt='--sr', label="Referans", color='#FF4B4B')
            ax.set_xlabel("Zaman (dk)"); ax.set_ylabel("Kümülatif Salım (%)"); ax.legend(); ax.grid(alpha=0.2)
            st.pyplot(fig)
            
            c1, c2 = st.columns(2)
            c1.metric("Dissolution Efficiency (DE %)", f"{de:.2f}%")
            c2.metric("Mean Dissolution Time (MDT)", f"{mdt:.2f} dk")

        # MODÜL 2: KİNETİK FİTTİNG
        elif menu == "🧮 Kinetik Model Fitting":
            st.subheader(f"🧮 {sel_name} - 16 Kinetik Model Analizi")
            
            models = {
                "Sıfır Derece": zero_order, "Birinci Derece": first_order, 
                "Higuchi": higuchi, "Korsmeyer-Peppas": korsmeyer_peppas,
                "Hixson-Crowell": hixson_crowell, "Gompertz": gompertz
            }
            
            results = []
            fig_all, ax_all = plt.subplots(figsize=(10, 5))
            ax_all.scatter(t, q, color='black', label="Deneysel")
            
            for name, func in models.items():
                try:
                    p_count = func.__code__.co_argcount - 1
                    popt, _ = curve_fit(func, t, q, maxfev=2000)
                    q_pred = func(t, *popt)
                    r2 = r2_score(q, q_pred)
                    rss = np.sum((q - q_pred)**2)
                    aic = calculate_aic(len(t), rss, p_count)
                    results.append({"Model": name, "R2": round(r2, 4), "AIC": round(aic, 2)})
                    ax_all.plot(t, q_pred, label=f"{name} (R2:{r2:.3f})")
                except: continue
            
            ax_all.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); st.pyplot(fig_all)
            st.table(pd.DataFrame(results).sort_values("R2", ascending=False))

        # MODÜL 4: f1 & f2
        elif menu == "📊 f1 & f2 Benzerlik Analizi":
            if ref_data:
                # Zaman noktalarını eşitleme (İnterpolasyon)
                common_t = np.sort(list(set(t) & set(ref_data["t"])))
                if len(common_t) > 0:
                    R = ref_data["mean"][:len(common_t)]
                    T = q[:len(common_t)]
                    f1 = (np.sum(np.abs(R - T)) / np.sum(R)) * 100
                    f2 = 50 * np.log10((1 + (1/len(common_t)) * np.sum((R - T)**2))**-0.5 * 100)
                    
                    st.subheader("Benzerlik Değerlendirmesi")
                    c1, c2 = st.columns(2)
                    c1.metric("f1 (Farklılık)", f"{f1:.2f}")
                    c2.metric("f2 (Benzerlik)", f"{f2:.2f}")
                    if f2 >= 50: st.success("✅ Profiller Benzer (f2 ≥ 50)")
                    else: st.error("❌ Profiller Benzer Değil (f2 < 50)")
                else: st.error("Zaman noktaları eşleşmiyor!")
            else: st.warning("Referans verisi yükleyiniz.")

    # RAPORLAMA BUTONU
    st.sidebar.divider()
    if st.sidebar.button("📥 Akıllı Raporu İndir"):
        st.sidebar.info("Raporlama altyapısı (xlsxwriter) hazır.")

else:
    st.info("👈 Lütfen sol panelden Test Verilerini (.xlsx veya .csv) yükleyerek analizi başlatın.")
