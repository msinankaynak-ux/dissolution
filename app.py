import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import io
from sklearn.metrics import r2_score

# --- KİNETİK MODELLER ---
def zero_order(t, k0): return k0 * t
def first_order(t, k1): return 100 * (1 - np.exp(-k1 * t))
def higuchi(t, kh): return kh * np.sqrt(t)
def korsmeyer_peppas(t, k, n): return k * (t**n)
def hixson_crowell(t, khc): return 100 * (1 - (1 - khc * t)**3)

def calculate_aic(n, rss, k):
    if n <= k or rss <= 0: return 9999
    return n * np.log(rss/n) + 2*k

st.set_page_config(page_title="PharmTech Pro Analiz", layout="wide")
st.title("🔬 Gelişmiş Dissolüsyon Analiz Laboratuvarı")

# --- DOSYA YÜKLEME ---
uploaded_file = st.file_uploader("Veri Dosyası Yükle (Excel veya CSV)", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        # Excel veya CSV okuma
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        # Veri İşleme
        time = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        values = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        
        mean_q = values.mean(axis=1).values
        std_q = values.std(axis=1).values
        cv_q = (std_q / mean_q) * 100
        
        # Grafik ve Kinetik Analiz
        st.subheader("📊 Dissolüsyon Profili ve Kinetik Uyumluluk")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.errorbar(time, mean_q, yerr=std_q, fmt='-o', color='darkblue', capsize=5, label="Ortalama Salım ± SD")
            ax.set_xlabel("Zaman (dakika)")
            ax.set_ylabel("Kümülatif Salım (%)")
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Kinetik Fitting (Ortalama veri üzerinden)
            t_fit = time[time > 0]
            q_fit = mean_q[time > 0]
            results = []
            models = [("Sıfırıncı D.", zero_order, [1]), ("Birinci D.", first_order, [0.1]), 
                      ("Higuchi", higuchi, [1]), ("Korsmeyer-Peppas", korsmeyer_peppas, [1, 0.5])]

            for name, func, p0 in models:
                try:
                    popt, _ = curve_fit(func, t_fit, q_fit, p0=p0, maxfev=10000)
                    y_pred = func(t_fit, *popt)
                    aic = calculate_aic(len(t_fit), np.sum((q_fit-y_pred)**2), len(p0))
                    r2 = r2_score(q_fit, y_pred)
                    results.append({"Model": name, "AIC": aic, "R²": r2})
                    t_plot = np.linspace(min(t_fit), max(t_fit), 100)
                    ax.plot(t_plot, func(t_plot, *popt), '--', alpha=0.7, label=f"{name}")
                except: continue
            
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.write("**İstatistiksel Özet**")
            summary_df = pd.DataFrame({"Zaman": time, "Ortalama": mean_q, "SD": std_q, "%CV": cv_q})
            st.dataframe(summary_df.style.format(precision=2))
            
            st.write("**Model Kıyaslaması**")
            st.table(pd.DataFrame(results).sort_values("AIC"))

                        # --- RAPORLAMA BÖLÜMÜ ---
        st.divider()
        with st.expander("📝 Rapor Detaylarını Gir ve PDF Hazırla"):
            from datetime import datetime
            now = datetime.now()
            
            c1, c2, c3 = st.columns(3)
            
            # 1. Sütun: Ürün Bilgileri
            api_name = c1.text_input("API / Etkin Madde Adı", "Atorvastatin")
            dosage = c1.text_input("Dozaj (mg)", "10 mg")
            batch = c1.text_input("Seri No (Batch)", "LOT-2026-001")
            
            # 2. Sütun: Metod Detayları
            medium = c2.text_input("Dissolüsyon Ortamı (pH)", "6.8 Fosfat Tamponu")
            volume = c2.text_input("Hacim (mL)", "900 mL")
            apparatus = c2.selectbox("Aparat", ["USP 1 (Basket)", "USP 2 (Paddle)", "USP 4 (Flow-through)"])
            
            # 3. Sütun: Operasyonel ve Zaman
            # Tarih ve saat otomatik olarak o anı gösterir ama değiştirilebilir
            exp_date = c3.date_input("Deney Tarihi", now.date())
            exp_time = c3.time_input("Deney Saati", now.time())
            temp = c3.text_input("Sıcaklık (°C)", "37 ± 0.5")
            speed = c3.text_input("Hız (rpm)", "50 rpm")
            analyst = st.text_input("Analist Adı / Soyadı", "")

            if st.button("Raporu Onayla ve Yazdır"):
                # Rapor özetini kullanıcıya göster
                st.info(f"Rapor Oluşturuldu: {api_name} ({dosage}) - {exp_date} {exp_time}")
                st.success("PDF indirme modülü bir sonraki adımda entegre edilecektir.")

    except Exception as e:
        st.error(f"Hata oluştu: {e}")
