import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simps
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import io

# --- MODELLER ---
def zero_order(t, k0): return k0 * t
def first_order(t, k1): return 100 * (1 - np.exp(-k1 * t))
def higuchi(t, kh): return kh * np.sqrt(t)
def korsmeyer_peppas(t, k, n): return k * (t**n)
def hixson_crowell(t, khc): return 100 * (1 - (1 - khc * t)**3)
def weibull(t, alpha, beta): return 100 * (1 - np.exp(-(t**beta) / alpha))

# --- YARDIMCI HESAPLAMALAR ---
def calculate_aic(n, rss, k):
    if n <= k or rss <= 0: return 9999
    return n * np.log(rss/n) + 2*k

def calculate_de(t, q):
    # Dissolution Efficiency (DE): Eğri altındaki alanın toplam alana oranı
    area = np.trapz(q, t)
    total_area = t.max() * 100
    return (area / total_area) * 100

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="PharmTech Lab Pro v6.0", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🔬 PharmTech Pro")
menu = st.sidebar.radio("Analiz Seçeneği:", ["📈 Profil Analizi", "🧮 Kinetik Modelleme", "📊 Karşılaştırmalı İstatistik"])

st.sidebar.divider()
test_file = st.sidebar.file_uploader("Test Verisi (Excel/CSV)", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi (Excel/CSV)", type=['xlsx', 'csv'])

def load_data(file):
    if file is None: return None
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    return {"t": t, "mean": v.mean(axis=1).values, "std": v.std(axis=1).values, "raw": v}

test = load_data(test_file)
ref = load_data(ref_file)

if test:
    t, m, s = test["t"], test["mean"], test["std"]
    
    # --- 1. PROFİL ANALİZİ ---
    if menu == "📈 Profil Analizi":
        st.subheader("📍 Dissolüsyon Profili ve Verim (DE)")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(t, m, yerr=s, fmt='-o', color='#1f77b4', label="Test", capsize=5, elinewidth=1)
        if ref:
            ax.errorbar(ref["t"], ref["mean"], yerr=ref["std"], fmt='--s', color='#ff7f0e', label="Referans", capsize=5, alpha=0.7)
        
        ax.set_ylim(0, 105); ax.set_xlabel("Zaman (dk)"); ax.set_ylabel("Salım (%)")
        ax.legend(); ax.grid(True, alpha=0.2)
        st.pyplot(fig)
        
        de_test = calculate_de(t, m)
        st.info(f"**Test Verisi Çözünme Verimi (DE):** %{de_test:.2f}")

    # --- 2. KİNETİK MODELLEME ---
    elif menu == "🧮 Kinetik Modelleme":
        st.subheader("🔍 Akıllı Model Seçimi (AIC & R² Bazlı)")
        
        mask = t > 0
        t_f, q_f = t[mask], m[mask]
        t_plot = np.linspace(0.1, t.max(), 100)
        
        models = [
            ("Zero-Order", zero_order, [1]),
            ("First-Order", first_order, [0.1]),
            ("Higuchi", higuchi, [1]),
            ("Korsmeyer-Peppas", korsmeyer_peppas, [1, 0.5]),
            ("Hixson-Crowell", hixson_crowell, [0.01]),
            ("Weibull", weibull, [100, 1])
        ]
        
        fig_m, ax_m = plt.subplots(figsize=(12, 6))
        ax_m.scatter(t, m, c='black', label="Gözlemlenen", zorder=10)
        
        results = []
        for name, func, p0 in models:
            try:
                popt, _ = curve_fit(func, t_f, q_f, p0=p0, maxfev=5000)
                y_pred = func(t_f, *popt)
                r2 = r2_score(q_f, y_pred)
                aic = calculate_aic(len(t_f), np.sum((q_f - y_pred)**2), len(p0))
                ax_m.plot(t_plot, func(t_plot, *popt), label=f"{name}", alpha=0.7)
                results.append({"Model": name, "R²": r2, "AIC": aic, "Parametreler": popt})
            except: continue
        
        ax_m.legend(bbox_to_anchor=(1,1)); ax_m.grid(alpha=0.1)
        st.pyplot(fig_m)
        
        res_df = pd.DataFrame(results).sort_values("AIC")
        st.table(res_df[["Model", "R²", "AIC"]])
        
        best_model = res_df.iloc[0]["Model"]
        st.success(f"🎯 **Önerilen En İyi Model:** {best_model} (En düşük AIC değerine göre)")

    # --- 3. İSTATİSTİKSEL KARŞILAŞTIRMA ---
    elif menu == "📊 Karşılaştırmalı İstatistik":
        st.subheader("📏 Model-Bağımsız Parametreler")
        if ref:
            # f1, f2
            R, T = ref["mean"], m
            if len(R) == len(T):
                f1 = (np.sum(np.abs(R - T)) / np.sum(R)) * 100
                f2 = 50 * np.log10((1 + np.mean((R - T)**2))**-0.5 * 100)
                
                c1, c2 = st.columns(2)
                c1.metric("f1 (Difference)", f"{f1:.2f}")
                c2.metric("f2 (Similarity)", f"{f2:.2f}")
                
                if f2 >= 50: st.success("Profiller Benzerdir.")
                else: st.error("Profiller Benzer Değildir.")
            else:
                st.warning("Zaman noktaları uyuşmuyor, f1/f2 hesaplanamadı.")
        else:
            st.warning("Karşılaştırma için Referans verisi yükleyiniz.")

else:
    st.info("👋 Hocam, analiz için sol panelden verileri yükleyebilirsiniz.")
