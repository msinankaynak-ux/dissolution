import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score
import io

# --- 1. DİL DESTEĞİ SÖZLÜĞÜ (TERMINOLOGY) ---
LANG_DICT = {
    "Türkçe": {
        "time": "Zaman", "release": "Salım", "test": "Test", "ref": "Referans",
        "model_fit": "Model Uygunluğu", "comment": "Yorum", "calc": "✅ Hesaplanabilir",
        "unsuitable": "❌ Uyumsuz", "best": "🏆 En Uygun Model", "sd": "S. Sapma", "rsd": "RSD (%)"
    },
    "English": {
        "time": "Time", "release": "Release", "test": "Test", "ref": "Reference",
        "model_fit": "Model Suitability", "comment": "Comment", "calc": "✅ Calculable",
        "unsuitable": "❌ Unsuitable", "best": "🏆 Best Fit", "sd": "Std. Dev.", "rsd": "RSD (%)"
    },
    "Chinese (中文)": {
        "time": "时间", "release": "释放率", "test": "测试组", "ref": "对照组",
        "model_fit": "模型适用性", "comment": "备注", "calc": "✅ 可计算",
        "unsuitable": "❌ 不适用", "best": "🏆 最佳拟合", "sd": "标准差", "rsd": "相对标准偏差 (%)"
    }
}

# --- 2. MATEMATİKSEL MODELLER (SABİT ÇATI) ---
def interpret_peppas_n(n, lang):
    if n <= 0.45: return "(Fickian)" if lang != "Türkçe" else "(Fickian Difüzyon)"
    elif 0.45 < n < 0.89: return "(Anomalous)" if lang != "Türkçe" else "(Anomalous Transport)"
    return "(Super Case II)"

def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def hixson(t, k): return 100 * (1 - (1 - np.maximum(k * t, 0))**3)
def korsmeyer(t, k, n): return k * (t**np.clip(n, 0.1, 1.5))
def kopcha(t, a, b): return a * np.sqrt(t) + b * t
def peppas_sahlin(t, k1, k2, m): return 100 * (k1 * (t**m) + k2 * (t**(2*m)))
def gompertz(t, xmax, k, i): return xmax * np.exp(-np.exp(k * (t - i)))
def weibull_complex(t, alpha, beta, td): return 100 * (1 - np.exp(- (np.maximum(t - td, 0)**beta) / alpha))

def baker_lonsdale_for_fit(t_data, k):
    def bl_root(q_guess, t_single, k_fit):
        q_norm = np.clip(q_guess / 100.0, 0.0001, 0.9999)
        return 1.5 * (1 - (1 - q_norm)**(2/3)) - q_norm - k_fit * t_single
    return np.array([root(bl_root, 50.0, args=(ts, k)).x[0] for ts in t_data])

def calculate_aic(n, rss, p_count):
    if n <= p_count or rss <= 0: return 9999
    return n * np.log(rss/n) + 2 * p_count

# --- 3. ARAYÜZ ---
st.set_page_config(page_title="PharmTech Lab v15.4", layout="wide")

# Dil Seçimi (Sidebar'ın en üstünde)
st.sidebar.title("🌍 Language / Dil")
selected_lang = st.sidebar.selectbox("Select Language:", list(LANG_DICT.keys()))
L = LANG_DICT[selected_lang]

st.sidebar.divider()
st.sidebar.subheader(f"📂 {L['test']} Veri Girişi")
test_file = st.sidebar.file_uploader(f"{L['test']} (XLSX/CSV)", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader(f"{L['ref']} (Opsiyonel)", type=['xlsx', 'csv'])

menu = st.sidebar.radio("Menü:", [f"📈 1. {L['release']} Profilleri", f"🧮 2. {L['model_fit']}"])

def load_data(file):
    if file is None: return None
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    mask = ~np.isnan(t)
    return {"t": t[mask], "raw": v.iloc[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask]}

test = load_data(test_file)

if test:
    t_raw, q_raw = test["t"], test["mean"]

    if "1." in menu:
        st.subheader(f"📍 {L['release']} Profili & İstatistik")
        
        # 1. TABLO: Ortalama, SD, RSD Tablosu
        st.write("### 📊 Veri İstatistiği")
        rsd = (test["std"] / q_raw) * 100
        stats_df = pd.DataFrame({
            L['time']: t_raw,
            f"Ortalama {L['release']} (%)": q_raw,
            L['sd']: test["std"],
            L['rsd']: rsd
        })
        st.table(stats_df.format("{:.2f}").hide(axis="index"))

        # 2. GRAFİK VE İNDİRME
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(t_raw, q_raw, yerr=test["std"], fmt='-ok', label=L['test'], capsize=5)
        ax.set_xlabel(f"{L['time']} (min)", fontsize=12)
        ax.set_ylabel(f"{L['release']} (%)", fontsize=12)
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Grafik İndirme Butonu
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button(label=f"🖼️ Grafiği İndir (PNG)", data=buf.getvalue(), file_name="salim_profili.png", mime="image/png")

    elif "2." in menu:
        st.subheader(f"🔍 {L['model_fit']}")
        
        fit_mask = (t_raw > 0) & (q_raw > 0)
        tf, qf = t_raw[fit_mask], q_raw[fit_mask]
        
        model_defs = [
            ("Sıfır Derece", zero_order, [0.1], [0], [100]),
            ("Birinci Derece", first_order, [0.01], [0], [10]),
            ("Higuchi", higuchi, [1.0], [0], [500]),
            ("Korsmeyer-Peppas", korsmeyer, [1.0, 0.5], [0, 0.1], [500, 2.0]),
            ("Hixson-Crowell", hixson, [0.001], [0], [1]),
            ("Kopcha", kopcha, [1.0, 0.1], [0, -10], [500, 100]),
            ("Peppas-Sahlin", peppas_sahlin, [0.1, 0.1, 0.5], [0, 0, 0.1], [100, 100, 1.5]),
            ("Gompertz", gompertz, [100, 0.1, 10], [50, 0, 0], [110, 5, 500]),
            ("Weibull (w/ Td)", weibull_complex, [50, 1.0, 1.0], [1, 0.1, 0], [10000, 10.0, 100])
        ]
        
        results = []
        for name, func, p0, low, up in model_defs:
            try:
                popt, _ = curve_fit(func, tf, qf, p0=p0, bounds=(low, up), maxfev=15000)
                y_p = func(tf, *popt)
                r2 = r2_score(qf, y_p)
                aic = calculate_aic(len(tf), np.sum((qf-y_p)**2), len(p0))
                comment = f"n: {popt[1]:.3f} {interpret_peppas_n(popt[1], selected_lang)}" if name == "Korsmeyer-Peppas" else "-"
                results.append({"Model": name, "R²": r2, "AIC": aic, L['model_fit']: L['calc'], L['comment']: comment})
            except:
                results.append({"Model": name, "R²": 0, "AIC": 9999, L['model_fit']: L['unsuitable'], L['comment']: "-"})

        df_res = pd.DataFrame(results)
        best_idx = df_res[df_res[L['model_fit']] == L['calc']]["AIC"].idxmin()
        df_res.at[best_idx, L['comment']] += f" {L['best']}"

        st.table(df_res.style.apply(lambda x: ['font-weight: bold' if x.name == best_idx else '' for i in x], axis=1)
                 .format({"R²": "{:.4f}", "AIC": "{:.2f}"}).hide(axis="index"))
else:
    st.info("Please upload data to start / Lütfen veri yükleyiniz.")
