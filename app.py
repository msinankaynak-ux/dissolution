import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score
import io

# --- KİNETİK MODELLER ---
def zero_order(t, k0): return k0 * t
def first_order(t, k1): return 100 * (1 - np.exp(-k1 * t))
def higuchi(t, kh): return kh * np.sqrt(t)
def korsmeyer_peppas(t, k, n): return k * (t**n)
def hixson_crowell(t, khc): return 100 * (1 - (1 - khc * t)**3)
def weibull(t, alpha, beta): return 100 * (1 - np.exp(-(t**beta) / alpha))

def calculate_aic(n, rss, k):
    if n <= k or rss <= 0: return 9999
    return n * np.log(rss/n) + 2*k

st.set_page_config(page_title="PharmTech Lab Pro", layout="wide")
st.title("🔬 Gelişmiş Dissolüsyon Analiz ve Raporlama Sistemi")

st.sidebar.header("📁 Veri Yönetimi")
mode = st.sidebar.radio("Analiz Modu", ["Tek Seri Analizi", "Referans vs Test Kıyaslama (f1/f2)"])

def process_data(file):
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    time = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0).values
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    mean_v = values.mean(axis=1).values
    std_v = values.std(axis=1).values
    cv_v = (std_v / mean_v * 100) if any(mean_v > 0) else np.zeros(len(mean_v))
    return time, mean_v, std_v, cv_v

# --- VERİ YÜKLEME ---
data_loaded = False
try:
    if mode == "Tek Seri Analizi":
        up_file = st.file_uploader("Excel/CSV Dosyası Yükle", type=['xlsx', 'csv'])
        if up_file:
            time, mean_q, std_q, cv_q = process_data(up_file)
            labels, means, stds, cvs = ["Numune"], [mean_q], [std_q], [cv_q]
            data_loaded = True
    else:
        c_up1, c_up2 = st.columns(2)
        ref_f = c_up1.file_uploader("Referans (R) Dosyası", type=['xlsx', 'csv'])
        test_f = c_up2.file_uploader("Test (T) Dosyası", type=['xlsx', 'csv'])
        if ref_f and test_f:
            time, r_m, r_s, r_c = process_data(ref_f)
            _, t_m, t_s, t_c = process_data(test_f)
            labels, means, stds, cvs = ["Referans", "Test"], [r_m, t_m], [r_s, t_s], [r_c, t_c]
            data_loaded = True

    if data_loaded:
        # --- 1. BÖLÜM: TEMEL DİSSOLÜSYON ---
        st.markdown("### 1️⃣ Temel Dissolüsyon Profili ve Veri Özeti")
        col1_a, col1_b = st.columns([3, 2])
        
        with col1_a:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            max_t = float(time.max())
            ax1.set_xlim(left=0, right=max_t * 1.05)
            ax1.set_ylim(bottom=0, top=110)
            for i in range(len(labels)):
                ax1.errorbar(time, means[i], yerr=stds[i], fmt='-o', label=labels[i], capsize=5, linewidth=2, markersize=8)
            ax1.set_xlabel("Zaman (dakika)")
            ax1.set_ylabel("Kümülatif Salım (%)")
            ax1.grid(True, linestyle='--', alpha=0.3)
            ax1.legend()
            st.pyplot(fig1)
            
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format="png", dpi=300, bbox_inches='tight')
            st.download_button(label="🖼️ Profili PNG Olarak İndir", data=buf1.getvalue(), file_name="dissolusyon_profili.png", mime="image/png")

        with col1_b:
            st.write("**İstatistiksel Veri Özeti**")
            summary_df = pd.DataFrame({"Zaman": time, "Ortalama (%)": means[0], "SD": stds[0], "%CV (VK)": cvs[0]})
            st.dataframe(summary_df.style.format(precision=2), use_container_width=True)

        st.divider()

        # --- 2. BÖLÜM: KİNETİK ANALİZ ---
        st.markdown("### 2️⃣ Kinetik Modelleme ve Uyumluluk Analizi")
        col2_a, col2_b = st.columns([3, 2])
        
        with col2_a:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.set_xlim(left=0, right=max_t * 1.05)
            ax2.set_ylim(bottom=0, top=110)
            for i in range(len(labels)):
                ax2.errorbar(time, means[i], yerr=stds[i], fmt='o', label=f"{labels[i]} (Deneysel)", capsize=4, alpha=0.5)

            t_fit, q_fit = time[time > 0], means[0][time > 0]
            t_plot = np.linspace(0, max_t, 100)
            kin_res = []
            models_list = [("Sıfırıncı D.", zero_order, [1]), ("Birinci D.", first_order, [0.1]), 
                           ("Higuchi", higuchi, [1]), ("Korsmeyer-Peppas", korsmeyer_peppas, [1, 0.5]),
                           ("Weibull", weibull, [100, 1])]
            
            for name, func, p0 in models_list:
                try:
                    popt, _ = curve_fit(func, t_fit, q_fit, p0=p0, maxfev=10000)
                    ax2.plot(t_plot, func(t_plot, *popt), '--', label=f"{name} Eğrisi")
                    r2 = r2_score(q_fit, func(t_fit, *popt))
                    aic = calculate_aic(len(t_fit), np.sum((q_fit-func(t_fit, *popt))**2), len(p0))
                    kin_res.append({"Model": name, "R²": round(r2, 4), "AIC": round(aic, 2)})
                except: continue
            
            ax2.set_xlabel("Zaman (dakika)")
            ax2.set_ylabel("Çözünen Madde (%)")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig2)

            buf2 = io.BytesIO()
            fig2.savefig(buf2, format="png", dpi=300, bbox_inches='tight')
            st.download_button(label="🖼️ Kinetik Grafiği PNG Olarak İndir", data=buf2.getvalue(), file_name="kinetik_analiz.png", mime="image/png")

        with col2_b:
            if mode == "Referans vs Test Kıyaslama (f1/f2)":
                mse = np.mean((means[0] - means[1])**2)
                f2 = 50 * np.log10((1 + mse)**-0.5 * 100)
                st.metric("f2 Benzerlik Faktörü", f"{f2:.2f}")
            st.write("**Model Karşılaştırma Listesi**")
            st.table(pd.DataFrame(kin_res).sort_values("AIC"))

        # --- RAPORLAMA ---
        st.divider()
        if st.checkbox("Analiz sonuçlarını onaylıyorum, raporlama adımına geç"):
            with st.expander("Rapor Detaylarını Doldurun", expanded=True):
                r1, r2, r3 = st.columns(3)
                api = r1.text_input("Etkin Madde / Ürün", "Atorvastatin")
                dosage = r1.text_input("Dozaj Formu", "10 mg Tablet")
                batch = r1.text_input("Seri No", "LOT-2026-001")
                medium = r2.text_input("Ortam", "pH 6.8 Fosfat Tamponu")
                volume = r2.text_input("Hacim (mL)", "900 mL")
                app_type = r2.selectbox("Aparat", ["USP 1", "USP 2", "USP 4"])
                analyst = r3.text_input("Analist", "")
                temp = r3.text_input("Sıcaklık (°C)", "37 ± 0.5")
                exp_date = r3.date_input("Tarih", datetime.now().date())
                if st.button("Raporu Kaydet"):
                    st.success("Analiz başarıyla kayıt altına alındı.")

except Exception as e:
    st.error(f"Hata oluştu: {e}")
