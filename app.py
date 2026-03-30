import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score
import io

# --- 1. MODEL BİLGİ BANKASI & AKADEMİK YORUMLAR ---
MODEL_KNOWLEDGE = {
    "Türkçe": {
        "Sıfır Derece": "Sıfır derece kinetiğine uymaktadır. Zamandan bağımsız sabit hızda salımı açıklar.",
        "Birinci Derece": "Birinci derece kinetiğine uymaktadır. Salım hızı, kalan ilaç konsantrasyonuna bağlıdır.",
        "Higuchi": "Higuchi kinetiğine uymaktadır. Matris sistemlerinden difüzyon temelli salımı açıklar.",
        "Korsmeyer-Peppas": "Korsmeyer-Peppas modeline uymaktadır. Mekanizma 'n' üsteli ile tanımlanır.",
        "Hixson-Crowell": "Hixson-Crowell kinetiğine uymaktadır. Yüzey alanı ve çapın zamanla küçüldüğü (erozyon) durumları açıklar.",
        "Hopfenberg": "Hopfenberg modeline uymaktadır. Yüzeyden aşınan (surface-eroding) polimerlerin geometrik (levha, silindir, küre) erozyonunu açıklar.",
        "Makoid-Banakar": "Makoid-Banakar modeline uymaktadır. Hem difüzyon nem de birinci dereceyi kapsar; başlangıçtaki 'burst release' (ani salım) etkisini ölçer.",
        "Square Root of Mass": "Kütle karekök modeline uymaktadır. Hixson-Crowell'e benzer ancak erozyonu kütle değişimi üzerinden hesaplar.",
        "Peppas-Sahlin": "Peppas-Sahlin modeline uymaktadır. Difüzyonel ve polimer relaksasyonu (erozyon) katkısını birbirinden ayırır.",
        "Gompertz": "Gompertz modeline uymaktadır. Gecikmeli başlayan sigmoid (S-tipi) profilleri açıklar.",
        "Weibull (w/ Td)": "Weibull modeline uymaktadır. Profilin ölçek, şekil ve gecikme süresini karakterize eder.",
        "Baker-Lonsdale": "Baker-Lonsdale modeline uymaktadır. Küresel matrislerden salımı açıklar.",
        "Kopcha": "Kopcha modeline uymaktadır. Difüzyon ve erozyon oranlarını ayrıştırır.",
        "Quadratic": "Quadratic modeline uymaktadır. Çok kısa süreli ve doğrusal olmayan salımları açıklar.",
        "Peppas-Rincon": "Peppas-Rincon modeline uymaktadır. Çok katmanlı veya karmaşık geometriler için geliştirilmiş versiyondur.",
        "Logistic": "Lojistik modele uymaktadır. Simetrik sigmoid (S-tipi) salımları açıklar."
    },
    "English": {
        "Sıfır Derece": "fits Zero-Order kinetics, describing a constant release rate independent of time.",
        "Birinci Derece": "fits First-Order kinetics. The release rate is concentration-dependent.",
        "Higuchi": "fits the Higuchi model, describing diffusion-based release from matrix systems.",
        "Korsmeyer-Peppas": "fits the Korsmeyer-Peppas model, where the mechanism is defined by the 'n' exponent.",
        "Hixson-Crowell": "fits Hixson-Crowell kinetics, explaining cases where surface area and particle diameter decrease (erosion) over time.",
        "Hopfenberg": "fits the Hopfenberg model, explaining surface-eroding polymers for specific geometries (slab, cylinder, sphere).",
        "Makoid-Banakar": "fits the Makoid-Banakar model, covering both diffusion and first-order release while accounting for initial 'burst release'.",
        "Square Root of Mass": "fits the Square Root of Mass model, similar to Hixson-Crowell but based on mass change erosion.",
        "Peppas-Sahlin": "fits the Peppas-Sahlin model, separating diffusion and relaxation contributions.",
        "Gompertz": "fits the Gompertz model, explaining sigmoid (S-type) lag-time profiles.",
        "Weibull (w/ Td)": "fits the Weibull model, characterizing profile scale, shape, and lag time.",
        "Baker-Lonsdale": "fits the Baker-Lonsdale model, explaining release from spherical matrices.",
        "Kopcha": "fits the Kopcha model, decoupling diffusion and erosion rates.",
        "Quadratic": "fits the Quadratic model, explaining short-term non-linear release.",
        "Peppas-Rincon": "fits the Peppas-Rincon model, developed for multi-layer or complex geometries.",
        "Logistic": "fits the Logistic model, explaining symmetric sigmoid (S-type) profiles."
    }
}

UNSUITABLE_DESC = {
    "Türkçe": "⚠️ Veri yapısı bu modelin matematiksel varsayımlarına (örneğin sigmoid yapı, erozyon hızı veya gecikme süresi) istatistiksel olarak uymuyor.",
    "English": "⚠️ Data structure does not statistically fit the model's mathematical assumptions (e.g., sigmoid shape, erosion rate, or lag-time)."
}

LANG_DICT = {
    "Türkçe": {
        "time": "Zaman (Dakika)", "release": "Kümülatif İlaç Salımı", "calc": "✅ Hesaplandı", "unsuitable": "❌ Uyumsuz", 
        "best": "🏆 En Uygun Model", "stats": "📊 Veri İstatistiği & Profil", "graph": "🛠️ Model Uyumu Grafiği", 
        "report": "📝 Akademik Değerlendirme", "model_title": "16 Kinetik Model Analizi", "unit": "dk"
    },
    "English": {
        "time": "Time (Minutes)", "release": "Cumulative Drug Release", "calc": "✅ Calculated", "unsuitable": "❌ Unsuitable", 
        "best": "🏆 Best Fit Model", "stats": "📊 Statistics & Profile", "graph": "🛠️ Model Fit Graph", 
        "report": "📝 Academic Evaluation", "model_title": "16 Kinetic Model Analysis", "unit": "min"
    }
}

# --- 2. MATEMATİKSEL FONKSİYONLAR ---
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def hixson(t, k): return 100 * (1 - (1 - np.maximum(k * t, 0))**3)
def korsmeyer(t, k, n): return k * (t**np.clip(n, 0.1, 1.5))
def kopcha(t, a, b): return a * np.sqrt(t) + b * t
def peppas_sahlin(t, k1, k2, m): return 100 * (k1 * (t**m) + k2 * (t**(2*m)))
def gompertz(t, xmax, k, i): return xmax * np.exp(-np.exp(k * (t - i)))
def weibull_complex(t, alpha, beta, td): return 100 * (1 - np.exp(- (np.maximum(t - td, 0)**beta) / alpha))
def hopfenberg(t, k, n): return 100 * (1 - (1 - np.maximum(k * t, 0))**n)
def makoid_banakar(t, k, n, c): return k * (t**n) * np.exp(-c * t)
def
