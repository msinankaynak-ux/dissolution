import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import io

# --- KİNETİK MOTORU ---
def zero_order(t, k0): return k0 * t
def first_order(t, k1): return 100 * (1 - np.exp(-k1 * t))
def higuchi(t, kh): return kh * np.sqrt(t)
def korsmeyer_peppas(t, k, n): return k * (t**n)
def hixson_crowell(t, khc): return 100 * (1 - (1 - khc * t)**3)
def weibull(t, alpha, beta): return 100 * (1 - np.exp(-(t**beta) / alpha))

def calculate_aic(n, rss, k):
    if n <= k or rss <= 0: return 9999
    return n * np.log(rss/n) + 2*k

def calculate_mdt(time, means):
    delta_t = np.diff(time, prepend=0)
    mid_q = np.diff(means, prepend=0)
    mid_t = time - (delta_t / 2)
    return np.sum(mid_t * mid_q) / np.sum(mid_q) if np.sum(mid_q) > 0 else 0

st.set_page_config(page_title="PharmTech Lab Pro v2.7", layout="wide")

# --- SIDEBAR (YAN MENÜ) TASARIMI ---
st.sidebar.title("🔬 Kontrol Paneli")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "İşlem Seçiniz:",
    ["📈 Dissolüsyon Profilleri", 
     "🧮 Model-Bağımlı Analiz", 
     "📊 Model-Bağımsız Analiz"]
)

st.sidebar.markdown("---")
st.sidebar.info("Hocam, verilerinizi yükledikten sonra yukarıdaki sekmelerden analizi derinleştirebilirsiniz.")

# --- VERİ YÜKLEME FONKSİYONU ---
def load_and_process(file):
    if file is None: return None
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    time = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0).values
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    return {"time": time, "mean": values.mean(axis=1).values, "std": values.std(axis=1).values, "raw": values}

# --- ANA EKRAN VE VERİ GİRİŞİ ---
st.title("PharmTech Lab: Gelişmiş Analitik Sistem")

col_u1, col_u2 = st.columns(2)
with col_u1:
    test_file = st.file_uploader("Test (Numune) Verisi", type=['xlsx', 'csv'], key="t_up")
with col_u2:
    ref_file = st.file_uploader("Referans Verisi (Kıyaslama İçin)", type=['xlsx', 'csv'], key="r_up")

test_data = load_and_process(test_file)
ref_data = load_and_process(ref_file)

if test_data is not None:
    time = test_data["time"]
    mean_q = test_data["mean"]
    std_q = test_data["std"]

    # --- 1. SEKMELİ YAPI: DİSSOLÜSYON PROFİLLERİ ---
    if menu == "📈 Dissolüsyon Profilleri":
        st.subheader("📍 Kümülatif Çözünme Profili")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(time, mean_q, yerr=std_q, fmt='-ok', label="Test (Numune)", capsize=4, linewidth=2)
        
        if ref_data is not None:
            ax.errorbar(ref_data["time"], ref_data["mean"], yerr=ref_data["std"], fmt='--sr', label="Referans", capsize=4, alpha=0.7)
        
        ax.set_xlabel("Zaman (dakika)")
        ax.set_ylabel("Kümülatif Salım (%)")
        ax.set_ylim(0, 110)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
        
        # Grafik İndirme
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button("🖼️ Profili PNG Olarak İndir", buf.getvalue(), "profil_analizi.png", "image/png")

    # --- 2. SEKMELİ YAPI: MODEL-BAĞIMLI ANALİZ ---
    elif menu == "🧮 Model-Bağımlı Analiz":
        st.subheader("🔍 Kinetik Modelleme (AIC & R² Temelli)")
        t_fit, q_fit = time[time > 0], mean_q[time > 0]
        kin_results = []
        
        models = [
            ("Zero-Order", zero_order, [1]),
            ("First-Order", first_order, [0.1]),
            ("Higuchi", higuchi, [1]),
            ("Korsmeyer-Peppas", korsmeyer_peppas, [1, 0.5]),
            ("Hixson-Crowell", hixson_crowell, [0.01]),
            ("Weibull", weibull, [100, 1])
        ]

        for name, func, p0 in models:
            try:
                popt, _ = curve_fit(func, t_fit, q_fit, p0=p0, maxfev=10000)
                y_pred = func(t_fit,
