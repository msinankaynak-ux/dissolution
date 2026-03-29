import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import io

# --- MATEMATİKSEL MODELLER ---
def zero_order(t, k0): return k0 * t
def first_order(t, k1): return 100 * (1 - np.exp(-k1 * t))
def higuchi(t, kh): return kh * np.sqrt(t)
def korsmeyer_peppas(t, k, n): return k * (t**n)
def hixson_crowell(t, khc): return 100 * (1 - (1 - khc * t)**3)
def weibull(t, alpha, beta): return 100 * (1 - np.exp(-(t**beta) / alpha))

def calculate_aic(n, rss, k):
    if n <= k or rss <= 0: return 9999
    return n * np.log(rss/n) + 2*k

# --- ARAYÜZ AYARLARI ---
st.set_page_config(page_title="PharmTech Lab v4.0", layout="wide")

# Sidebar - Navigasyon ve Veri Girişi
st.sidebar.title("🔬 PharmTech Sidebar")
main_menu = st.sidebar.selectbox(
    "Analiz Türü Seçiniz:",
    ["📈 Dissolüsyon Profilleri", "🧮 Model-Bağımlı Analiz", "📊 Model-Bağımsız Analiz"]
)

st.sidebar.divider()
st.sidebar.subheader("📁 Veri Yükleme")
test_file = st.sidebar.file_uploader("Test (Numune) Verisi", type=['xlsx', 'csv'], key="test")
ref_file = st.sidebar.file_uploader("Referans Verisi", type=['xlsx', 'csv'], key="ref")

# Veri İşleme Fonksiyonu
def process_data(file):
    if file is None: return None
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    t = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0).values
    vals = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    return {"t": t, "mean": vals.mean(axis=1).values, "std": vals.std(axis=1).values}

test_data = process_data(test_file)
ref_data = process_data(ref_file)

# --- ANA EKRAN MANTIĞI ---
if test_data is not None:
    t_data, m_data, s_data = test_data["t"], test_data["mean"], test_data["std"]

    if main_menu == "📈 Dissolüsyon Profilleri":
        st.subheader("📍 Kümülatif Çözünme Profili")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(t_data, m_data, yerr=s_data, fmt='-ok', label="Test", capsize=5, lw=2)
        if ref_data:
            ax.errorbar(ref_data["t"], ref_data["mean"], yerr=ref_data["std"], fmt='--sr', label="Referans", capsize=5, alpha=0.7)
        
        ax.set_xlabel("Zaman (dk)"); ax.set_ylabel("Salım (%)"); ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # İndirme Butonu (Hata düzeltildi: fig nesnesi burada tanımlı)
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=300)
        st.download_button("🖼️ Grafiği PNG Olarak İndir", img_buf.getvalue(), "profil.png")

    elif main_menu == "🧮 Model-Bağımlı Analiz":
        st.subheader("🔍 Tüm Kinetik Eğrilerin Fit Edilmesi")
        
        # Sadece salım olan kısımları fit et (t>0)
        mask = t_data > 0
        t_fit, q_fit = t_data[mask], m_data[mask]
        t_smooth = np.linspace(0.1, t_data.max(), 100)
        
        fig_m, ax_m = plt.subplots(figsize=(12, 7))
        ax_m.scatter(t_data, m_data, color='black', s=50, label="Deneysel Veri", zorder=5)
        
        model_list = [
            ("Sıfır Derece", zero_order, [1]),
            ("Birinci Derece", first_order, [0.1]),
            ("Higuchi", higuchi, [1]),
            ("Korsmeyer-Peppas", korsmeyer_peppas, [1, 0.5]),
            ("Hixson-Crowell", hixson_crowell, [0.01]),
            ("Weibull", weibull, [100, 1])
        ]
        
        stats = []
        for name, func, p0 in model_list:
            try:
                popt, _ = curve_fit(func, t_fit, q_fit, p0=p0, maxfev=10000)
                y_smooth = func(t_smooth, *popt)
                y_pred = func(t_fit, *popt)
                r2 = r2_score(q_fit, y_pred)
                aic = calculate_aic(len(t_fit), np.sum((q_fit - y_pred)**2), len(p0))
                
                ax_m.plot(t_smooth, y_smooth, label=f"{name} (R²: {r2:.4f})", alpha=0.8, lw=2)
                stats.append({"Model": name, "R²": r2, "AIC": aic})
            except: continue
        
        ax_m.set_xlabel("Zaman (dk)"); ax_m.set_ylabel("Salım (%)")
        ax_m.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax_m.grid(True, alpha=0.2)
        st.pyplot(fig_m)
        
        st.table(pd.DataFrame(stats).sort_values("AIC"))

    elif main_menu == "📊 Model-Bağımsız Analiz":
        st.subheader("📏 Benzerlik ve Farklılık Faktörleri (f1, f2)")
        if ref_data:
            # f1, f2 Hesaplama
            R, T = ref_data["mean"], m_data
            if len(R) == len(T):
                f1 = (np.sum(np.abs(R - T)) / np.sum(R)) * 100
                f2 = 50 * np.log10((1 + np.mean((R - T)**2))**-0.5 * 100)
                
                c1, c2 = st.columns(2)
                c1.metric("f1 (Difference)", f"{f1:.2f}")
                c2.metric("f2 (Similarity)", f"{f2:.2f}")
                
                if f2 >= 50: st.success("Sonuç: Profiller İstatistiksel Olarak Benzerdir.")
                else: st.error("Sonuç: Profiller Benzer Değildir.")
            else:
                st.error("Hata: Test ve Referans dosyasındaki zaman noktası sayıları uyuşmuyor.")
        else:
            st.warning("Lütfen karşılaştırma için sol panelden bir Referans verisi yükleyin.")

else:
    st.info("👋 Hocam hoş geldiniz. Analize başlamak için sol taraftaki sidebar üzerinden dosyalarınızı yükleyin.")
