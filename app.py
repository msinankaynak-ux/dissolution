import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score

# --- ÇEKİRDEK FONKSİYONLAR ---

def interpret_peppas_n(n):
    """Korsmeyer-Peppas n değerine göre mekanizma yorumu."""
    if n <= 0.45: return "(Fickian Difüzyon)"
    elif 0.45 < n < 0.89: return "(Anomalous/Non-Fickian)"
    return "(Süper Case II)"

def calculate_aic(n, rss, p_count):
    if n <= p_count or rss <= 0: return np.inf
    return n * np.log(rss/n) + 2 * p_count

# Modeller
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def korsmeyer(t, k, n): return k * (t**n)
def baker_lonsdale_fit(t, k):
    # Basitleştirilmiş çözüm: AIC için Q tahmini üretir
    def bl_root(q, t_val, k_val):
        qn = np.clip(q/100.0, 1e-5, 0.9999)
        return 1.5 * (1 - (1 - qn)**(2/3)) - qn - k_val * t_val
    return np.array([root(bl_root, 50, args=(ts, k)).x[0] for ts in t])

# --- ARAYÜZ ---
st.set_page_config(page_title="PharmTech Lab v15.2", layout="wide")
st.sidebar.title("🔬 Smart Lab v15.2")

menu = st.sidebar.radio("Analiz Paneli:", ["📈 Profil Analizi", "🧮 Kinetik Modelleme (v15.2)"])
test_file = st.sidebar.file_uploader("Test Verisi (Excel/CSV)", type=['xlsx', 'csv'])

if test_file:
    df_raw = pd.read_excel(test_file) if test_file.name.endswith('.xlsx') else pd.read_csv(test_file)
    t = pd.to_numeric(df_raw.iloc[:, 0]).values
    q = df_raw.iloc[:, 1:].mean(axis=1).values
    
    if menu == "🧮 Kinetik Modelleme (v15.2)":
        st.subheader("📊 Karşılaştırmalı Kinetik Tablosu")
        
        results = []
        # Fit işlemleri (Örnek modeller)
        models = [
            ("Sıfır Derece", zero_order, [0.1]),
            ("Birinci Derece", first_order, [0.01]),
            ("Higuchi", higuchi, [1.0]),
            ("Korsmeyer-Peppas", korsmeyer, [1.0, 0.5])
        ]
        
        for name, func, p0 in models:
            try:
                popt, _ = curve_fit(func, t[t>0], q[t>0], p0=p0, maxfev=5000)
                y_pred = func(t[t>0], *popt)
                r2 = r2_score(q[t>0], y_pred)
                rss = np.sum((q[t>0] - y_pred)**2)
                aic = calculate_aic(len(t[t>0]), rss, len(p0))
                
                yorum = "-"
                if name == "Korsmeyer-Peppas":
                    yorum = f"n: {popt[1]:.3f} {interpret_peppas_n(popt[1])}"
                
                results.append({"Model": name, "R²": r2, "AIC": aic, "Model Uygunluğu": "✅ Hesaplanabilir", "Yorum": yorum})
            except:
                results.append({"Model": name, "R²": 0, "AIC": 999, "Model Uygunluğu": "❌ Veri Uygun Değil", "Yorum": "Hata"})

        df_res = pd.DataFrame(results)
        
        # Karar Mekanizması: En düşük AIC'yi bul
        valid_models = df_res[df_res["Model Uygunluğu"] == "✅ Hesaplanabilir"]
        if not valid_models.empty:
            best_idx = valid_models["AIC"].idxmin()
            df_res.at[best_idx, "Yorum"] += " 🏆 En Uygun Model"
        else:
            best_idx = -1

        # Tabloyu görselleştir (İndeksi kaldır ve Bold yap)
        def highlight_best(s):
            return ['font-weight: bold' if s.name == best_idx else '' for _ in s]

        st.table(df_res.style.apply(highlight_best, axis=1).format({"R²": "{:.4f}", "AIC": "{:.2f}"}))

        # GRAFİK
        st.divider()
        st.write("### 📈 Model Uyumu Grafiği")
        fig, ax = plt.subplots()
        ax.scatter(t, q, color='black', label="Deneysel")
        # Grafikte tüm modelleri göster (Kullanıcı isterse multiselect ile daraltabilir)
        for name, func, p0 in models:
             try:
                popt, _ = curve_fit(func, t[t>0], q[t>0], p0=p0)
                ax.plot(t, func(t, *popt), label=name)
             except: continue
        ax.legend(); st.pyplot(fig)
