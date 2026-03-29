import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# --- GENİŞLETİLMİŞ MODEL KÜTÜPHANESİ ---

# 1. Temel Modeller
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def hixson(t, k): return 100 * (1 - (1 - k * t)**3)
def korsmeyer(t, k, n): return k * (t**n)

# 2. Yeni Eklenen Geometrik ve Kompleks Modeller (Görsellerden)
def baker_lonsdale(t, k):
    # 3/2 * [1 - (1 - Mt/Minf)^(2/3)] - (Mt/Minf) = k * t
    # Bu model Mt/Minf (0-1 arası) ile çalışır. Çıktı Q (%)'dir.
    # Tersten çözümü kompleks olduğu için fit sırasında Q'yu Mt/Minf'e çevireceğiz.
    # Pratik fit için basitleştirilmiş form:
    return k * t # Bu modelin fit edilmesi için veri ön işlemi gerekir, aşağıda handle edildi.

def kopcha(t, a, b):
    # Mt = a * t^0.5 + b * t
    return a * np.sqrt(t) + b * t

def peppas_sahlin(t, k1, k2, m):
    # Mt / Minf = k1 * t^m + k2 * t^(2m)
    # Çıktı Q (%) olması için 100 ile çarpıyoruz.
    return 100 * (k1 * (t**m) + k2 * (t**(2*m)))

def weibull_complex(t, alpha, beta, td):
    # M = M0 * [1 - exp(-(t-Td)^beta / alpha)]
    # t < Td ise salım 0'dır.
    q = 100 * (1 - np.exp(- (np.maximum(t - td, 0)**beta) / alpha))
    return q

def gompertz(t, xmax, k, i):
    # Xt = Xmax * exp(-exp(k * (t-i)))
    return xmax * np.exp(-np.exp(k * (t - i)))

# --- HEYECAN VERİCİ YENİ YORUMLAMA FONKSİYONU ---
def interpret_peppas_sahlin(k1, k2):
    # k1: difüzyonel katkı, k2: gevşeme/erozyon katkısı
    ratio = abs(k1) / (abs(k1) + abs(k2)) if (abs(k1) + abs(k2)) > 0 else 0.5
    if k1 > 0 and k2 > 0:
        if ratio > 0.75: return f"Baskın Difüzyon (%{ratio*100:.0f})"
        elif ratio < 0.25: return f"Baskın Erozyon (%{(1-ratio)*100:.0f})"
        else: return "Kombine Mekanizma"
    return "Kompleks Salım"

def calculate_aic(n, rss, p_count):
    if n <= p_count or rss <= 0: return np.inf
    return n * np.log(rss/n) + 2 * p_count

# --- ARAYÜZ ---
st.set_page_config(page_title="PharmTech Lab v15", layout="wide")
st.sidebar.title("🔬 Pro Lab v15.0")

menu = st.sidebar.radio("İşlem Adımları:", ["📈 1. Salım Profilleri", "🧮 2. Tüm Modelleri Test Et", "📊 3. Modelden Bağımsız"])

test_file = st.sidebar.file_uploader("Test Verisi", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi", type=['xlsx', 'csv'])

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

    if menu == "📈 1. Salım Profilleri":
        st.subheader("📍 Yayın Kalitesinde Salım Profili")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(t_data, q_data, yerr=test["std"], fmt='-ok', label="Test", capsize=5, linewidth=2)
        if ref:
            ax.errorbar(ref["t"], ref["mean"], yerr=ref["std"], fmt='--sr', label="Referans", alpha=0.7)
        ax.set_xlabel("Zaman (dk)"); ax.set_ylabel("Salım (%)"); ax.set_ylim(0, 105); ax.legend(); ax.grid(True, linestyle='--')
        st.pyplot(fig)

    elif menu == "🧮 2. Tüm Modelleri Test Et":
        st.subheader("🔍 Literatürdeki Tüm Modellerin Karşılaştırması")
        
        # Fit için veri hazırlığı (t=0 Peppas gibi modellerde hataya neden olur)
        f_mask = (t_data > 0) & (q_data > 0)
        tf, qf = t_data[f_mask], q_data[f_mask]
        
        # MODEL TANIMLARI LİSTESİ
        # (İsim, Fonksiyon, [p0], [Bounds_Low], [Bounds_High])
        model_defs = [
            ("Sıfır Derece", zero_order, [0.1], [0], [10]),
            ("Birinci Derece", first_order, [0.01], [0], [1]),
            ("Higuchi", higuchi, [1.0], [0], [100]),
            ("Korsmeyer-Peppas", korsmeyer, [1.0, 0.5], [0, 0.1], [100, 2.0]),
            ("Hixson-Crowell", hixson, [0.001], [0], [0.1]),
            # Yeni Eklenenler
            ("Kopcha", kopcha, [1.0, 0.1], [0, -10], [100, 10]),
            ("Peppas-Sahlin", peppas_sahlin, [0.1, 0.1, 0.5], [0, 0, 0.1], [10, 10, 1.5]),
            ("Gompertz", gompertz, [100, 0.1, 10], [50, 0, 0], [110, 1, 120]),
            ("Weibull (w/ Td)", weibull_complex, [50, 1.0, 1.0], [1, 0.1, 0], [5000, 5.0, 30])
        ]
        
        fit_results = {}
        summary_data = []

        # BAKER-LONSDALE İÇİN ÖZEL FİT (Mt/Minf dönüşümü gerektirir)
        try:
            mt_minf = qf / 100.0
            # 3/2 * [1 - (1 - y)^(2/3)] - y
            y_bl = 1.5 * (1 - (np.maximum(1 - mt_minf, 1e-6))**(2/3)) - mt_minf
            # y_bl = k * t -> Eksen t, Veri y_bl. Doğrusal regresyon.
            popt_bl, _ = curve_fit(zero_order, tf, y_bl, p0=[0.001])
            
            # Grafiği çizmek için Mt/Minf'i Q'ya geri çeviremeyiz (numerik çözüm gerekir),
            # bu yüzden R2'yi y_bl ekseninde hesaplıyoruz.
            r2_bl = r2_score(y_bl, zero_order(tf, *popt_bl))
            summary_data.append({"Model": "Baker-Lonsdale", "R²": round(r2_bl, 4), "AIC": "-", "Durum": "✅ Uygun", "Yorum": f"k: {popt_bl[0]:.5f}"})
        except:
            summary_data.append({"Model": "Baker-Lonsdale", "R²": "-", "AIC": "-", "Durum": "❌ Uyumsuz", "Yorum": "-"})

        # DİĞER STANDART MODELLERİN FİTİ
        for name, func, p0, b_low, b_up in model_defs:
            try:
                popt, _ = curve_fit(func, tf, qf, p0=p0, bounds=(b_low, b_up), maxfev=15000)
                y_pred = func(tf, *popt)
                r2 = r2_score(qf, y_pred)
                rss = np.sum((qf - y_pred)**2)
                aic = calculate_aic(len(tf), rss, len(p0))
                
                comment = "-"
                if name == "Peppas-Sahlin":
                    comment = interpret_peppas_sahlin(popt[0], popt[1])
                elif name == "Korsmeyer-Peppas":
                    comment = f"n: {popt[1]:.3f}"
                elif name == "Weibull (w/ Td)":
                    comment = f"Td: {popt[2]:.1f} dk"

                fit_results[name] = {"func": func, "popt": popt, "r2": r2}
                summary_data.append({"Model": name, "R²": round(r2, 4), "AIC": round(aic, 2), "Durum": "✅ Uygun", "Yorum": comment})
            except:
                summary_data.append({"Model": name, "R²": "-", "AIC": "-", "Durum": "❌ Uyumsuz", "Yorum": "Yakınsama Hatası"})

        st.table(pd.DataFrame(summary_data))
        
        # İNTERAKTİF SEÇİM VE GRAFİK
        st.divider()
        st.write("### 🛠️ Grafikte Gösterilecek Modelleri Seçin (En iyi AIC'ye göre)")
        # Baker-Lonsdale grafiği farklı eksende olduğu için seçimden çıkarıldı.
        printable_models = [m for m in fit_results.keys()]
        selected_models = [m for m in printable_models if st.checkbox(m, value=(m == "Peppas-Sahlin"))]
        
        if selected_models:
            fig_fit, ax_fit = plt.subplots(figsize=(10, 5))
            ax_fit.scatter(t_data, q_data, color='black', label="Deneysel", s=40, zorder=5)
            t_plot = np.linspace(t_data.min(), t_data.max(), 100)
            
            for m_name in selected_models:
                m_info = fit_results[m_name]
                ax_fit.plot(t_plot, m_info["func"](t_plot, *m_info["popt"]), label=f"{m_name} (R²: {m_info['r2']:.3f})")
            
            ax_fit.set_xlabel("Zaman"); ax_fit.set_ylabel("Salım (%)"); ax_fit.legend(); ax_fit.grid(alpha=0.2)
            st.pyplot(fig_fit)

else:
    st.info("Hocam, literatür taramasına başlamak için verileri yükleyin.")
