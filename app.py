import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# --- KİNETİK MODELLER ---
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def korsmeyer(t, k, n): return k * (t**n)
def hixson(t, k): return 100 * (1 - (1 - k * t)**3)

def calculate_aic(n, rss, p_count):
    if n <= p_count or rss <= 0: return np.inf
    return n * np.log(rss/n) + 2 * p_count

# --- ARAYÜZ YAPISI ---
st.set_page_config(page_title="PharmTech Lab v14", layout="wide")
st.sidebar.title("🔬 Pro Lab v14.0")

menu = st.sidebar.radio("İşlem Adımları:", ["📈 1. Salım Profilleri (Yayın Hazırlığı)", "🧮 2. Model Karşılaştırma & Seçim", "📊 3. Modelden Bağımsız Analiz"])

test_file = st.sidebar.file_uploader("Test Verisi Yükle", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi Yükle", type=['xlsx', 'csv'])

def load_data(file):
    if not file: return None
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        mask = ~np.isnan(t)
        return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask]}
    except: return None

test = load_data(test_file)
ref = load_data(ref_file)

if test:
    t_data, q_data = test["t"], test["mean"]

    # --- ADIM 1: SALIM PROFİLLERİ ---
    if menu == "📈 1. Salım Profilleri (Yayın Hazırlığı)":
        st.subheader("📍 Kümülatif Çözünme Profili")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(t_data, q_data, yerr=test["std"], fmt='-ok', label="Test Formülasyonu", capsize=5, linewidth=2, markersize=8)
        if ref:
            ax.errorbar(ref["t"], ref["mean"], yerr=ref["std"], fmt='--sr', label="Referans Ürün", alpha=0.7, capsize=5)
        
        ax.set_xlabel("Zaman (dakika)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Kümülatif Salım (%)", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 105); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        st.info("💡 Bu grafik doğrudan yayınlarda kullanılabilir kalitededir.")

    # --- ADIM 2: MODEL KARŞILAŞTIRMA ---
    elif menu == "🧮 2. Model Karşılaştırma & Seçim":
        st.subheader("🔍 Kinetik Model Uyumluluğu")
        
        # Fit için 0 zamanını pas geçiyoruz (matematiksel zorunluluk)
        f_mask = (t_data > 0) & (q_data > 0)
        tf, qf = t_data[f_mask], q_data[f_mask]
        
        model_defs = [
            ("Sıfır Derece", zero_order, [0.1]),
            ("Birinci Derece", first_order, [0.01]),
            ("Higuchi", higuchi, [1.0]),
            ("Korsmeyer-Peppas", korsmeyer, [1.0, 0.5]),
            ("Hixson-Crowell", hixson, [0.001])
        ]
        
        fit_results = {}
        summary_data = []

        for name, func, p0 in model_defs:
            try:
                popt, _ = curve_fit(func, tf, qf, p0=p0, maxfev=10000)
                y_pred = func(tf, *popt)
                r2 = r2_score(qf, y_pred)
                rss = np.sum((qf - y_pred)**2)
                aic = calculate_aic(len(tf), rss, len(p0))
                
                fit_results[name] = {"func": func, "popt": popt, "r2": r2}
                summary_data.append({"Model": name, "R²": round(r2, 4), "AIC": round(aic, 2), "Durum": "✅ Uygun"})
            except:
                summary_data.append({"Model": name, "R²": "-", "AIC": "-", "Durum": "❌ Bu veri için uygun değil"})

        # Tabloyu Göster
        st.table(pd.DataFrame(summary_data))
        
        # İnteraktif Seçim
        st.divider()
        st.write("### 🛠️ Grafik Üzerinde Gösterilecek Modelleri Seçin")
        selected_models = [m for m in fit_results.keys() if st.checkbox(m, value=(m == "Higuchi"))]
        
        if selected_models:
            fig_fit, ax_fit = plt.subplots(figsize=(10, 5))
            ax_fit.scatter(t_data, q_data, color='black', label="Deneysel Veri", s=50)
            t_plot = np.linspace(t_data.min(), t_data.max(), 100)
            
            for m_name in selected_models:
                m_info = fit_results[m_name]
                ax_fit.plot(t_plot, m_info["func"](t_plot, *m_info["popt"]), label=f"{m_name} (R²: {m_info['r2']:.3f})")
            
            ax_fit.set_xlabel("Zaman"); ax_fit.set_ylabel("Salım (%)"); ax_fit.legend(); ax_fit.grid(alpha=0.3)
            st.pyplot(fig_fit)

    # --- ADIM 3: MODEL BAĞIMSIZ ---
    elif menu == "📊 3. Modelden Bağımsız Analiz":
        st.subheader("📏 f1, f2 ve Verim Analizi")
        de = (np.trapz(q_data, t_data) / (np.max(t_data) * 100)) * 100
        st.metric("Çözünme Verimi (DE)", f"%{de:.2f}")
        
        if ref:
            if len(ref["t"]) == len(t_data):
                f2 = 50 * np.log10(100 / np.sqrt(1 + np.mean((ref["mean"] - q_data)**2)))
                st.metric("f2 Benzerlik Faktörü", f"{f2:.2f}")
            else:
                st.warning("Zaman noktası sayıları uyuşmadığı için f2 hesaplanamadı.")
else:
    st.info("Hocam hoş geldiniz. Lütfen sol taraftan verileri yükleyerek başlayın.")
