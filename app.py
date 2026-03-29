import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score

# --- SMART KİNETİK MOTOR ---

# n Değerine Göre Mekanizma
def interpret_peppas_n(n):
    if n <= 0.45: return "(Fickian)"
    elif 0.45 < n < 0.89: return "(Anomalous)"
    return "(Case II/Relaxation)"

# MODELLER
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def hixson(t, k): return 100 * (1 - (1 - np.maximum(k * t, 0))**3)
def korsmeyer(t, k, n): return k * (t**np.clip(n, 0.1, 1.5))
def kopcha(t, a, b): return a * np.sqrt(t) + b * t
def peppas_sahlin(t, k1, k2, m): return 100 * (k1 * (t**m) + k2 * (t**(2*m)))
def gompertz(t, xmax, k, i): return xmax * np.exp(-np.exp(k * (t - i)))
def weibull_complex(t, alpha, beta, td): return 100 * (1 - np.exp(- (np.maximum(t - td, 0)**beta) / alpha))

# BAKER-LONSDALE İÇİN ÖZEL TERS ÇÖZÜM
# Bu fonksiyon Q değerini k ve t'den numerik olarak bulur. AIC hesaplamak için gereklidir.
def baker_lonsdale_for_fit(t_data, k):
    def bl_root(q_guess, t_single, k_fit):
        q_norm = q_guess / 100.0
        # logaritma/üs içinde negatif çıkmaması için clip
        q_norm = np.clip(q_norm, 0.0001, 0.9999) 
        # f(Q) = 3/2 * [1 - (1 - Q)^(2/3)] - Q - k*t = 0
        return 1.5 * (1 - (1 - q_norm)**(2/3)) - q_norm - k_fit * t_single
    
    q_results = []
    for t_s in t_data:
        sol = root(bl_root, 50.0, args=(t_s, k)) # 50'den başla
        q_results.append(np.clip(sol.x[0], 0, 100))
    return np.array(q_results)

def calculate_aic(n, rss, p_count):
    if n <= p_count or rss <= 0: return np.inf
    # AIC = n*log(RSS/n) + 2*p
    return n * np.log(rss/n) + 2 * p_count

# --- ARAYÜZ ---
st.set_page_config(page_title="PharmTech Lab v15.1", layout="wide")
st.sidebar.title("🔬 Smart Lab v15.1")

menu = st.sidebar.radio("Adımlar:", ["📈 Profiller", "🧮 Tüm Modelleri Test Et (v15.1)", "📊 f1/f2 Analizi"])

test_file = st.sidebar.file_uploader("Test Verisi", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi", type=['xlsx', 'csv'])

def load_data(file):
    if file is None: return None
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    mask = ~np.isnan(t)
    return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask]}

test = load_data(test_file)
ref = load_data(ref_file)

if test:
    t_raw, q_raw = test["t"], test["mean"]

    if menu == "📈 Profiller":
        # ... v14 grafik kodu ...
        pass

    elif menu == "🧮 Tüm Modelleri Test Et (v15.1)":
        st.subheader("🔍 Karşılaştırmalı Kinetik Tablosu ve Otomatik Karar")
        
        # Fit için veri hazırlığı (t=0 filtresi logaritmik modeller için şart)
        fit_mask = (t_raw > 0) & (q_raw > 0)
        tf, qf = t_raw[fit_mask], q_raw[fit_mask]
        
        fit_results = {}
        summary_list = []

        # MODEL TANIMLARI LİSTESİ
        model_defs = [
            ("Sıfır Derece", zero_order, [0.1], [0], [10]),
            ("Birinci Derece", first_order, [0.01], [0], [1]),
            ("Higuchi", higuchi, [1.0], [0], [100]),
            ("Korsmeyer-Peppas", korsmeyer, [1.0, 0.5], [0, 0.1], [100, 2.0]),
            ("Hixson-Crowell", hixson, [0.001], [0], [0.1]),
            ("Kopcha", kopcha, [1.0, 0.1], [0, -10], [100, 10]),
            ("Peppas-Sahlin", peppas_sahlin, [0.1, 0.1, 0.5], [0, 0, 0.1], [10, 10, 1.5]),
            ("Gompertz", gompertz, [100, 0.1, 10], [50, 0, 0], [110, 1, 120]),
            ("Weibull (w/ Td)", weibull_complex, [50, 1.0, 1.0], [1, 0.1, 0], [5000, 5.0, 30])
        ]

        # Baker-Lonsdale AIC Darboğazı Çözümü
        try:
            mt_minf = qf / 100.0
            y_bl = 1.5 * (1 - (np.maximum(1 - mt_minf, 1e-6))**(2/3)) - mt_minf
            # İlk tahmin için doğrusal regresyon
            popt_bl, _ = curve_fit(zero_order, tf, y_bl, p0=[0.001])
            k_init = popt_bl[0]
            
            # Asıl fit işlemi yüzde salım ekseninde numerik ters çözümle
            # t_raw ve q_raw üzerinden tüm zaman noktalarını fit et (tf ve qf değil)
            popt_bl_final, _ = curve_fit(baker_lonsdale_for_fit, t_raw, q_raw, p0=[k_init], maxfev=1000)
            
            y_pred_bl = baker_lonsdale_for_fit(t_raw, *popt_bl_final)
            r2_bl = r2_score(q_raw, y_pred_bl)
            rss_bl = np.sum((q_raw - y_pred_bl)**2)
            aic_bl = calculate_aic(len(t_raw), rss_bl, 1) # p_count = 1 (k)
            
            fit_results["Baker-Lonsdale"] = {"func_for_fit": baker_lonsdale_for_fit, "popt": popt_bl_final, "r2": r2_bl}
            summary_list.append({"Model": "Baker-Lonsdale", "R²": r2_bl, "AIC": aic_bl, "Adaylık": "✅ Aday", "Yorum": f"k: {popt_bl_final[0]:.5f}"})
        except Exception as e:
            summary_list.append({"Model": "Baker-Lonsdale", "R²": np.nan, "AIC": np.nan, "Adaylık": "❌ Uyumsuz", "Yorum": "Matematiksel Hata"})

        # Diğer standart modellerin fiti
        for name, func, p0, b_low, b_up in model_defs:
            try:
                # bounds ve maxfev artırıldı
                popt, _ = curve_fit(func, tf, qf, p0=p0, bounds=(b_low, b_up), maxfev=15000)
                y_pred = func(tf, *popt)
                r2 = r2_score(qf, y_pred)
                rss = np.sum((qf - y_pred)**2)
                aic = calculate_aic(len(tf), rss, len(p0))
                
                comment = "-"
                if name == "Korsmeyer-Peppas":
                    # n: 0.498 (Fickian Diffusion)
                    comment = f"n: {popt[1]:.3f} {interpret_peppas_n(popt[1])}"
                elif name == "Peppas-Sahlin":
                    comment = f"k1: {popt[0]:.2f}, k2: {popt[1]:.2f}"

                fit_results[name] = {"func_for_fit": func, "popt": popt, "r2": r2}
                summary_list.append({"Model": name, "R²": r2, "AIC": aic, "Adaylık": "✅ Aday", "Yorum": comment})
            except:
                summary_list.append({"Model": name, "R²": np.nan, "AIC": np.nan, "Adaylık": "❌ Uyumsuz", "Yorum": "Yakınsama Sağlanamadı"})

        # Tabloyu Pandas DataFrame'e Çevir
        df_summary = pd.DataFrame(summary_list)
        
        # Otorite Modeli (Best Model) Bul: NaN olmayan ve en düşük AIC
        best_row = df_summary[df_summary['Durum'] == "✅ Aday"].sort_values('AIC').first_valid_index()
        if best_row is not None:
            # Yorum kısmına karar yazısını ekle
            df_summary.at[best_row, 'Yorum'] += " 🏆 En İyi Uyum (AIC)"

        # GÖRSEL DÜZENLEME: BOLD SATIR VE İNDEKS KALDIRMA
        # pandas style ile bold yap
        df_styled = df_summary.style.apply(
            lambda x: ['font-weight: bold' if x.name == best_row else '' for i in x], axis=1
        ).format(
            {"R²": "{:.4f}", "AIC": "{:.2f}"}, na_rep="-", precision=2
        ).hide(axis="index") # İndeks sütununu (0-1-2) kaldır

        # Tabloyu Streamlit'e Ver
        st.write(df_styled)
        
        # GRAFİK SEÇİMİ (Hepsi Seçili Başlar)
        st.divider()
        st.write("### 🛠️ Grafikte Gösterilecek Modelleri Seçin (Tümü Aktif)")
        
        printable_models = [m for m in fit_results.keys()]
        # v15.1 değişikliği: value=printable_models ile hepsi seçili gelir
        selected_models = st.multiselect("Modeller:", printable_models, default=printable_models)
        
        if selected_models:
            fig_fit, ax_fit = plt.subplots(figsize=(10, 5))
            ax_fit.scatter(t_raw, q_raw, color='black', label="Deneysel Veri", s=40, zorder=5)
            # t_plot = np.linspace(t_raw.min(), t_raw.max(), 100)
            
            for m_name in selected_models:
                m_info = fit_results[m_name]
                # Modeller tf üzerinden fit edildi ama q_raw üzerinden R2 hesaplandı.
                # Baker-Lonsdale numerik çözüm olduğu için fit_mask ile sınırlanmadı.
                ax_fit.plot(t_raw, m_info["func_for_fit"](t_raw, *m_info["popt"]), label=f"{m_name}")
            
            ax_fit.set_xlabel("Zaman (dk)"); ax_fit.set_ylabel("Salım (%)"); ax_fit.legend(); ax_fit.grid(alpha=0.1)
            st.pyplot(fig_fit)

else:
    st.info("Hocam hoş geldiniz. Literatür taramasına başlamak için verileri yükleyin.")
