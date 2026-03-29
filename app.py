import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import io
from sklearn.metrics import r2_score

# --- MODELLER ---
def zero_order(t, k0): return k0 * t
def first_order(t, k1): return 100 * (1 - np.exp(-k1 * t))
def higuchi(t, kh): return kh * np.sqrt(t)
def korsmeyer_peppas(t, k, n): return k * (t**n)
def hixson_crowell(t, khc): return 100 * (1 - (1 - khc * t)**3)

def calculate_aic(n, rss, k):
    if n <= k or rss <= 0: return 9999
    return n * np.log(rss/n) + 2*k

# --- ARAYÜZ ---
st.set_page_config(page_title="PharmTech Pro", layout="wide")
st.title("🔬 PharmTech Dissolution Analysis")
st.markdown("Verilerinizi yükleyin, tüm modelleri saniyeler içinde kıyaslayalım.")

uploaded_file = st.file_uploader("Veri Dosyası (Excel veya CSV)", type=['xlsx', 'csv', 'txt'])

if uploaded_file:
    try:
        # Veri okuma ve temizleme (TR format desteği)
        content = uploaded_file.read().decode("utf-8")
        cleaned = content.replace(",", ".").replace(";", ",")
        df = pd.read_csv(io.StringIO(cleaned))
        
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        q = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        valid = ~np.isnan(t) & ~np.isnan(q)
        t, q = t[valid & (t > 0)], q[valid & (t > 0)]

        results = []
        models = [("Sıfırıncı Derece", zero_order, [1]), ("Birinci Derece", first_order, [0.1]), 
                  ("Higuchi", higuchi, [1]), ("Korsmeyer-Peppas", korsmeyer_peppas, [1, 0.5]), 
                  ("Hixson-Crowell", hixson_crowell, [0.01])]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(t, q, color='red', label='Deneysel Veri', zorder=5)

        best_aic = float('inf')
        best_name = ""

        for name, func, p0 in models:
            try:
                popt, _ = curve_fit(func, t, q, p0=p0, maxfev=10000)
                y_pred = func(t, *popt)
                aic = calculate_aic(len(t), np.sum((q-y_pred)**2), len(p0))
                r2 = r2_score(q, y_pred)
                results.append({"Model": name, "AIC": round(aic, 2), "R²": round(r2, 4)})
                
                t_plot = np.linspace(min(t), max(t), 100)
                ax.plot(t_plot, func(t_plot, *popt), label=f"{name} (R²:{round(r2,2)})")
                
                if aic < best_aic:
                    best_aic, best_name = aic, name
            except: continue

        st.subheader(f"✅ En Uygun Mekanizma: {best_name}")
        col1, col2 = st.columns([2, 1])
        with col1:
            ax.set_xlabel("Zaman (dk)"); ax.set_ylabel("Salım (%)"); ax.legend(); st.pyplot(fig)
        with col2:
            st.table(pd.DataFrame(results).sort_values("AIC"))
            
    except Exception as e:
        st.error(f"Hata: {e}")