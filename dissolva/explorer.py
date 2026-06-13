"""DissolvA Academy — interactive kinetic model explorer.

A self-contained HTML/JS component (rendered via st.components.v1.html) that lets
the learner drag parameter sliders and watch the release curve redraw live, with
t50 / t80 / %@end read-outs and a "pin & compare" overlay. Time window is
switchable (60 / 120 / 360 min) so lag-time and extended-release models display
well.

IMPORTANT (IP): these are the STANDARD, published textbook equations implemented
in JS purely for teaching/visualisation. The proprietary fitting engine (62-model
optimisation, bootstrap) stays server-side in the private backend and is NOT used
here. So this module exposes nothing confidential.

Model keys match dissolva.models.MODEL_DEFS exactly, so academy.py can pre-select
the model whose detail page is open. EXPLORABLE lists the covered models.
"""
import json
import streamlit as st
import streamlit.components.v1 as components

# Theme (matches dissolva.theme)
_BG      = "#1C2541"   # workspace
_CARD    = "#16203F"   # surface
_AMBER   = "#FFCC00"
_TXT     = "#FFFFFF"
_TXT2    = "#9fb0d0"
_GRID    = "rgba(255,255,255,0.08)"
_AXIS    = "rgba(255,255,255,0.35)"

# Models with a closed-form F(t) we can render client-side. Each:
#   key: [ equation_label, [ [plabel,min,max,step,default], ... ], js_function_body ]
# js body is the EXPRESSION returning F given t and params p[0..]; helper fns
# erf/phi/bl (Baker-Lonsdale) are available in scope.
_M = {
  "Zero Order":        ["F = k₀·t", [["k₀ (%/min)",0.1,3,0.1,1.0]], "p[0]*t"],
  "First Order":       ["F = 100·(1−e^(−k₁t))", [["k₁ (1/min)",0.005,0.2,0.005,0.05]], "100*(1-Math.exp(-p[0]*t))"],
  "Higuchi":           ["F = k_H·√t", [["k_H (%/√min)",1,18,0.5,10]], "p[0]*Math.sqrt(t)"],
  "Hixson-Crowell":    ["F = 100·(1−(1−k_s·t)³)", [["k_s (1/min)",0.004,0.05,0.001,0.02]], "t<1/p[0]?100*(1-Math.pow(1-p[0]*t,3)):100"],
  "Korsmeyer-Peppas":  ["F = k·tⁿ", [["k",1,30,0.5,10],["n (exponent)",0.3,1.0,0.01,0.5]], "p[0]*Math.pow(t,p[1])"],
  "Hopfenberg":        ["F = 100·[1−(1−k·t)ⁿ]", [["k (1/min)",0.005,0.04,0.001,0.02],["n (geometry 1–3)",1,4,0.5,2]], "t<1/p[0]?100*(1-Math.pow(1-p[0]*t,p[1])):100"],
  "Baker-Lonsdale":    ["3/2[1−(1−F)^(2/3)]−F = k_BL·t", [["k_BL (1/min)",0.0005,0.01,0.0005,0.002]], "bl(t,p[0])"],
  "Makoid-Banakar":    ["F = k·tⁿ·e^(−b·t)", [["k",1,30,0.5,10],["n",0.3,1.0,0.01,0.5],["b (1/min)",0,0.05,0.005,0.01]], "p[0]*Math.pow(t,p[1])*Math.exp(-p[2]*t)"],
  "Peppas-Sahlin":     ["F = k₁·t^m + k₂·t^2m", [["k₁ (diffusion)",0,20,0.5,6],["k₂ (erosion)",-2,5,0.1,0.8],["m",0.3,0.8,0.01,0.5]], "p[0]*Math.pow(t,p[2])+p[1]*Math.pow(t,2*p[2])"],
  "Weibull":           ["F = 100·(1−e^(−(t−T_d)^b/a))", [["a (scale)",5,120,1,50],["b (shape)",0.4,2.5,0.05,1.0],["T_d (lag, min)",0,20,0.5,0]], "t<=p[2]?0:100*(1-Math.exp(-Math.pow(t-p[2],p[1])/p[0]))"],
  "Gompertz":          ["F = A·e^(−b·e^(−k·t))", [["A (%)",80,100,1,100],["b",1,8,0.1,5],["k (1/min)",0.02,0.3,0.01,0.1]], "p[0]*Math.exp(-p[1]*Math.exp(-p[2]*t))"],
  "Logistic":          ["F = A/(1+e^(−k(t−t₅₀)))", [["A (%)",80,100,1,100],["k (steepness)",0.05,0.5,0.01,0.15],["t₅₀ (min)",5,50,1,25]], "p[0]/(1+Math.exp(-p[1]*(t-p[2])))"],
  "Quadratic":         ["F = a·t² + b·t + c", [["a",-0.03,0.01,0.005,-0.01],["b",0.1,3,0.1,1.5],["c",0,15,1,0]], "p[0]*t*t+p[1]*t+p[2]"],
  "Probit":            ["F = A·Φ((t−μ)/σ)", [["μ (min)",5,50,1,30],["σ",5,30,1,15],["A (%)",80,100,1,100]], "p[2]*phi((t-p[0])/p[1])"],
  "Weibull (No Lag)":  ["F = 100·(1−e^(−t^b/a))", [["a (scale)",5,120,1,50],["b (shape)",0.4,2.5,0.05,1.0]], "100*(1-Math.exp(-Math.pow(t,p[1])/p[0]))"],
  "KP + Lag":          ["F = k·(t−t_lag)ⁿ", [["k",1,30,0.5,10],["n",0.3,1.0,0.01,0.5],["t_lag (min)",0,20,0.5,5]], "t<=p[2]?0:p[0]*Math.pow(t-p[2],p[1])"],
  "First Order + Lag": ["F = 100·(1−e^(−k₁(t−t_lag)))", [["k₁ (1/min)",0.005,0.2,0.005,0.06],["t_lag (min)",0,20,0.5,5]], "t<=p[1]?0:100*(1-Math.exp(-p[0]*(t-p[1])))"],
  "Zero Order + Lag":  ["F = k₀·(t−t_lag)", [["k₀ (%/min)",0.1,3,0.1,1.0],["t_lag (min)",0,20,0.5,5]], "t<=p[1]?0:p[0]*(t-p[1])"],
  "Higuchi + Lag":     ["F = k_H·√(t−t_lag)", [["k_H (%/√min)",1,18,0.5,10],["t_lag (min)",0,20,0.5,5]], "t<=p[1]?0:p[0]*Math.sqrt(t-p[1])"],
  "Probit Log":        ["F = A·Φ((log₁₀t−μ)/σ)", [["μ",0.5,2.5,0.1,1.5],["σ",0.1,1.0,0.05,0.5],["A (%)",80,100,1,100]], "t<=0?0:p[2]*phi((Math.log(t)/Math.LN10-p[0])/p[1])"],
  "Double Exponential":["F = A₁(1−e^−k₁t)+A₂(1−e^−k₂t)", [["A₁ (%)",0,100,1,60],["k₁ (1/min)",0.005,0.3,0.005,0.05],["A₂ (%)",0,100,1,40],["k₂ (1/min)",0.001,0.1,0.001,0.005]], "p[0]*(1-Math.exp(-p[1]*t))+p[2]*(1-Math.exp(-p[3]*t))"],
  "Triple Exponential":["F = Σ Aᵢ(1−e^−kᵢt)", [["A₁ (%)",0,80,1,40],["k₁",0.01,0.3,0.005,0.1],["A₂ (%)",0,80,1,40],["k₂",0.005,0.1,0.005,0.02],["A₃ (%)",0,60,1,20],["k₃",0.001,0.05,0.001,0.005]], "p[0]*(1-Math.exp(-p[1]*t))+p[2]*(1-Math.exp(-p[3]*t))+p[4]*(1-Math.exp(-p[5]*t))"],
  "Power-Exponential": ["F = A·(1−e^(−k·tⁿ))", [["A (%)",80,100,1,100],["k",0.01,0.3,0.01,0.05],["n",0.6,2.0,0.05,1.2]], "p[0]*(1-Math.exp(-p[1]*Math.pow(t,p[2])))"],
  "Biexp. Absorption": ["F = F_r·100·k_a/(k_a−k)·(e^−kt−e^−k_at)", [["F_r (fraction)",0.5,1.0,0.05,1.0],["k_a (1/min)",0.05,0.5,0.01,0.2],["k (1/min)",0.005,0.1,0.005,0.05]], "Math.abs(p[1]-p[2])<1e-6?0:p[0]*100*p[1]/(p[1]-p[2])*(Math.exp(-p[2]*t)-Math.exp(-p[1]*t))"],
  "Gallagher-Corrigan":["Biphasic: burst (k₁) + slow sigmoid (k₂)", [["A_max (burst %)",20,100,1,60],["k₁ (1/min)",0.02,0.4,0.01,0.1],["k₂ (1/min)",0.005,0.1,0.005,0.03],["t_max (min)",10,120,1,40]], "p[0]*(1-Math.exp(-p[1]*t))+(100-p[0])*(Math.exp(p[2]*(t-p[3]))/(1+Math.exp(p[2]*(t-p[3]))))"],
  "Combined Higuchi+FO":["F = α·k_H√t + (1−α)·100(1−e^−k₁t)", [["k_H (%/√min)",1,18,0.5,10],["k₁ (1/min)",0.005,0.2,0.005,0.05],["α (mix)",0,1,0.05,0.5]], "p[2]*p[0]*Math.sqrt(t)+(1-p[2])*100*(1-Math.exp(-p[1]*t))"],
  "Henriksen":         ["F = A·(e^−k₁t−e^−k₂t)", [["A (%)",40,150,1,90],["k₁ (1/min)",0.005,0.2,0.005,0.03],["k₂ (1/min)",0.05,0.6,0.01,0.2]], "p[0]*(Math.exp(-p[1]*t)-Math.exp(-p[2]*t))"],
  "Modified Gompertz": ["F = A_max·e^(−e^(μ·e/A_max·(λ−t)+1))", [["A_max (%)",80,100,1,100],["μ (rate)",0.05,0.5,0.01,0.15],["λ (lag, min)",0,30,1,8]], "p[0]*Math.exp(-Math.exp(p[1]*Math.E/p[0]*(p[2]-t)+1))"],
  "Richards":          ["F = A·(1+e^(−k(t−t₅₀)))^(−1/n)", [["A (%)",80,100,1,100],["k",0.05,0.5,0.01,0.15],["n (asymmetry)",0.3,3,0.1,1.0],["t₅₀ (min)",5,50,1,25]], "p[0]*Math.pow(1+Math.exp(-p[1]*(t-p[3])),-1/p[2])"],
  "4-Parameter Logistic":["F = A+(B−A)/(1+e^(−k(t−t₅₀)))", [["A (low %)",0,30,1,0],["B (high %)",70,100,1,100],["k",0.05,0.5,0.01,0.15],["t₅₀ (min)",5,50,1,25]], "p[0]+(p[1]-p[0])/(1+Math.exp(-p[2]*(t-p[3])))"],
  "Log-Normal":        ["F = A·Φ((ln t−μ)/σ)", [["μ",1.5,4.5,0.1,3.2],["σ",0.2,1.2,0.05,0.5],["A (%)",80,100,1,100]], "t<=0?0:p[2]*phi((Math.log(t)-p[0])/p[1])"],
  "Hill Equation":     ["F = A_max·tⁿ/(kⁿ+tⁿ)", [["A_max (%)",80,100,1,100],["k (min)",5,50,1,25],["n (cooperativity)",1,4,0.1,1.8]], "p[0]*Math.pow(t,p[2])/(Math.pow(p[1],p[2])+Math.pow(t,p[2]))"],
  "Dose-Response":     ["F = E_min+(E_max−E_min)·tⁿ/(EC₅₀ⁿ+tⁿ)", [["E_min (%)",0,30,1,0],["E_max (%)",70,100,1,100],["EC₅₀ (min)",5,50,1,25],["n",1,4,0.1,1.5]], "p[0]+(p[1]-p[0])*Math.pow(t,p[3])/(Math.pow(p[2],p[3])+Math.pow(t,p[3]))"],
  "Fractal First Order":["F = 100·(1−e^(−k·t^α))", [["k",0.01,0.3,0.01,0.05],["α (fractal)",0.4,1.2,0.05,0.8]], "100*(1-Math.exp(-p[0]*Math.pow(t,p[1])))"],
  "Stretched Exponential":["F = A·(1−e^(−(t/τ)^β))", [["A (%)",80,100,1,100],["β",0.4,1.5,0.05,0.8],["τ (min)",5,60,1,30]], "p[0]*(1-Math.exp(-Math.pow(t/p[2],p[1])))"],
  "Fractal Weibull":   ["F = 100·(1−e^(−((t−γ)/α)^β))", [["α (scale)",5,80,1,30],["β (shape)",0.4,2.5,0.05,1.2],["γ (lag, min)",0,20,0.5,0]], "t<=p[2]?0:100*(1-Math.exp(-Math.pow((t-p[2])/p[0],p[1])))"],
  "Exponential Assoc.":["F = A·(1−e^(−k·t))", [["A (%)",80,100,1,100],["k (1/min)",0.005,0.2,0.005,0.05]], "p[0]*(1-Math.exp(-p[1]*t))"],
  "Hyperbolic":        ["F = A_max·t/(k+t)", [["A_max (%)",80,100,1,100],["k (min)",5,50,1,20]], "p[0]*t/(p[1]+t)"],
  "Linear-Exponential":["F = A·t·e^(−k·t)+b", [["A",0.5,5,0.1,2.0],["k (1/min)",0.005,0.08,0.005,0.02],["b (%)",0,20,1,0]], "p[0]*t*Math.exp(-p[1]*t)+p[2]"],
  "Brody Growth":      ["F = A·(1−b·e^(−k·t))", [["A (%)",80,120,1,100],["k (1/min)",0.01,0.2,0.005,0.05],["b",0.5,1.0,0.05,0.9]], "p[0]*(1-p[2]*Math.exp(-p[1]*t))"],
  "Bertalanffy":       ["F = A·(1−e^(−k·t))ⁿ", [["A (%)",80,100,1,100],["k (1/min)",0.01,0.2,0.005,0.05],["n",1,4,0.1,3.0]], "p[0]*Math.pow(1-Math.exp(-p[1]*t),p[2])"],
  "Pade Approximation":["F = (a₀+a₁·t)/(1+b₁·t)", [["a₀ (%)",0,30,1,0],["a₁",0.5,5,0.1,2.0],["b₁ (1/min)",0.005,0.06,0.005,0.02]], "(p[0]+p[1]*t)/(1+p[2]*t)"],
  "hPLC Model":        ["F = 100·(1−(1+A·tⁿ)^(−B))", [["A",0.01,0.2,0.01,0.05],["B",0.5,3,0.1,1.5],["n",0.5,2,0.05,1.0]], "100*(1-Math.pow(1+p[0]*Math.pow(t,p[2]),-p[1]))"],
  "Compreg Model":     ["F = 100·(1−e^(−k·tⁿ))^m", [["k",0.01,0.2,0.005,0.05],["n",0.5,2,0.05,1.0],["m",0.5,4,0.1,2.0]], "100*Math.pow(1-Math.exp(-p[0]*Math.pow(t,p[1])),p[2])"],
  "KP Modified":       ["F = k·tⁿ/(1+b·t)", [["k",1,30,0.5,10],["n",0.3,1.2,0.01,0.5],["b (1/min)",0,0.05,0.005,0.01]], "p[0]*Math.pow(t,p[1])/(1+p[2]*t)"],
  "Makoid-Banakar Mod.":["F = k·tⁿ·e^(−b·t)+c", [["k",1,30,0.5,10],["n",0.3,1.0,0.01,0.5],["b (1/min)",0,0.05,0.005,0.01],["c (%)",0,20,1,0]], "p[0]*Math.pow(t,p[1])*Math.exp(-p[2]*t)+p[3]"],
  "Zero Order + F0":   ["F = F₀ + k₀·t", [["F₀ (burst %)",0,40,1,10],["k₀ (%/min)",0.1,3,0.1,1.0]], "p[0]+p[1]*t"],
  "First Order + Fmax":["F = F_max·(1−e^(−k₁t))", [["k₁ (1/min)",0.005,0.2,0.005,0.05],["F_max (%)",70,100,1,100]], "p[1]*(1-Math.exp(-p[0]*t))"],
  "First Order + Tlag + Fmax":["F = F_max·(1−e^(−k₁(t−t_lag)))", [["k₁ (1/min)",0.005,0.2,0.005,0.06],["t_lag (min)",0,20,0.5,5],["F_max (%)",70,100,1,100]], "t<=p[1]?0:p[2]*(1-Math.exp(-p[0]*(t-p[1])))"],
  "Higuchi + F0":      ["F = F₀ + k_H·√t", [["F₀ (burst %)",0,40,1,10],["k_H (%/√min)",1,18,0.5,8]], "p[0]+p[1]*Math.sqrt(t)"],
  "KP + F0":           ["F = F₀ + k·tⁿ", [["F₀ (burst %)",0,40,1,10],["k",1,25,0.5,8],["n",0.3,1.0,0.01,0.5]], "p[0]+p[1]*Math.pow(t,p[2])"],
  "Peppas-Sahlin 2":   ["F = k₁·√t + k₂·t", [["k₁ (diffusion)",0,20,0.5,5],["k₂ (erosion)",-1,3,0.1,1.0]], "p[0]*Math.sqrt(t)+p[1]*t"],
  "Second Order":      ["F = k·t²", [["k",0.005,0.1,0.005,0.03]], "p[0]*t*t"],
  "Third Order":       ["F = k·t³", [["k",0.0001,0.002,0.0001,0.0005]], "p[0]*t*t*t"],
  "Michaelis-Menten":  ["F = Q_max·t/(k_m+t)", [["Q_max (%)",80,100,1,100],["k_m (min)",5,60,1,20]], "p[0]*t/(p[1]+t)"],
  "Hixson-Crowell + Lag":["F = 100·(1−(1−k_s(t−t_lag))³)", [["k_s (1/min)",0.004,0.05,0.001,0.02],["t_lag (min)",0,20,0.5,5]], "t<=p[1]?0:((t-p[1])<1/p[0]?100*(1-Math.pow(1-p[0]*(t-p[1]),3)):100)"],
  "Logistic 1 (DDSolver)":["F = 100·e^(α+β·ln t)/(1+e^(α+β·ln t))", [["α",-5,2,0.1,-3],["β",0.5,4,0.1,1.5]], "t<=0?0:100*(function(z){return Math.exp(z)/(1+Math.exp(z));})(p[0]+p[1]*Math.log(t))"],
  "Logistic 2 (DDSolver)":["F = F_max·e^(α+β·ln t)/(1+e^(…))", [["α",-5,2,0.1,-3],["β",0.5,4,0.1,1.5],["F_max (%)",70,100,1,100]], "t<=0?0:p[2]*(function(z){return Math.exp(z)/(1+Math.exp(z));})(p[0]+p[1]*Math.log(t))"],
  "Gompertz 1 (DDSolver)":["F = 100·e^(−e^(α−β·ln t))", [["α",1,5,0.1,2.5],["β",0.5,4,0.1,1.5]], "t<=0?0:100*Math.exp(-Math.exp(p[0]-p[1]*Math.log(t)))"],
  "Gompertz 2 (DDSolver)":["F = F_max·e^(−e^(α−β·ln t))", [["α",1,5,0.1,2.5],["β",0.5,4,0.1,1.5],["F_max (%)",70,100,1,100]], "t<=0?0:p[2]*Math.exp(-Math.exp(p[0]-p[1]*Math.log(t)))"],
  "Probit 1 (DDSolver)":["F = 100·Φ(α+β·log₁₀t)", [["α",-3,1,0.1,-1.5],["β",0.5,4,0.1,1.5]], "t<=0?0:100*phi(p[0]+p[1]*Math.log(t)/Math.LN10)"],
}

EXPLORABLE = set(_M.keys())


def is_explorable(name: str) -> bool:
    return name in EXPLORABLE


_HTML = r"""
<div id="kmx-root" style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:__CARD__;border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:14px 16px;height:100%;box-sizing:border-box;display:flex;flex-direction:column;color:__TXT__;">
  <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px;">
    <select id="kmx-model" style="flex:1;min-width:200px;background:__BG__;color:__TXT__;border:1px solid rgba(255,255,255,0.15);border-radius:8px;padding:7px 9px;font-size:13px;"></select>
    <select id="kmx-tmax" style="background:__BG__;color:__TXT__;border:1px solid rgba(255,255,255,0.15);border-radius:8px;padding:7px 9px;font-size:13px;">
      <option value="60">0–60 min</option><option value="120">0–120 min</option><option value="360">0–360 min</option>
    </select>
    <button id="kmx-pin" style="background:transparent;color:__AMBER__;border:1px solid __AMBER__;border-radius:8px;padding:7px 11px;font-size:13px;cursor:pointer;white-space:nowrap;">Pin curve</button>
  </div>
  <div id="kmx-eq" style="font-family:ui-monospace,Menlo,monospace;font-size:13px;color:__AMBER__;background:__BG__;padding:7px 11px;border-radius:8px;margin-bottom:10px;"></div>
  <div id="kmx-sliders" style="margin-bottom:6px;"></div>
  <div style="flex:1;min-height:240px;position:relative;">
    <svg id="kmx-plot" viewBox="0 0 640 300" preserveAspectRatio="xMidYMid meet" style="position:absolute;top:0;left:0;width:100%;height:100%;display:block;"></svg>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px;">
    <div style="background:__BG__;border-radius:8px;padding:7px 10px;"><div style="font-size:11px;color:__TXT2__;">t₅₀</div><div id="kmx-t50" style="font-size:18px;font-weight:600;">—</div></div>
    <div style="background:__BG__;border-radius:8px;padding:7px 10px;"><div style="font-size:11px;color:__TXT2__;">t₈₀</div><div id="kmx-t80" style="font-size:18px;font-weight:600;">—</div></div>
    <div style="background:__BG__;border-radius:8px;padding:7px 10px;"><div style="font-size:11px;color:__TXT2__;">% @ end</div><div id="kmx-qend" style="font-size:18px;font-weight:600;">—</div></div>
  </div>
</div>
<style>
#kmx-root input[type=range]{-webkit-appearance:none;height:4px;border-radius:2px;background:rgba(255,255,255,0.18);outline:none;accent-color:__AMBER__;}
#kmx-root input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:__AMBER__;cursor:pointer;}
#kmx-root input[type=range]::-moz-range-thumb{width:16px;height:16px;border:none;border-radius:50%;background:__AMBER__;cursor:pointer;}
#kmx-root button:hover{background:__AMBER__;color:__BG__;}
</style>
<script>
(function(){
  var MODELS=__MODELS__, PRESELECT=__PRESELECT__;
  function erf(x){var s=x<0?-1:1;x=Math.abs(x);var a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;var t=1/(1+p*x);var y=1-(((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);return s*y;}
  function phi(z){return 0.5*(1+erf(z/Math.SQRT2));}
  function bl(t,k){var tg=k*t;if(tg<=0)return 0;if(tg>=0.5)return 100;var lo=0,hi=1;for(var i=0;i<46;i++){var m=(lo+hi)/2;var h=1.5*(1-Math.pow(1-m,2/3))-m;if(h<tg)lo=m;else hi=m;}return (lo+hi)/2*100;}
  function clamp(q){if(!isFinite(q))return 0;return Math.max(0,Math.min(100,q));}
  var sel=document.getElementById('kmx-model'),tmaxSel=document.getElementById('kmx-tmax'),sl=document.getElementById('kmx-sliders');
  var names=Object.keys(MODELS);
  names.forEach(function(n){var o=document.createElement('option');o.value=n;o.textContent=n;sel.appendChild(o);});
  if(MODELS[PRESELECT])sel.value=PRESELECT;
  var pinned=null,cur=null;
  function fval(key,t,p){try{return MODELS[key].f(t,p,erf,phi,bl);}catch(e){return 0;}}
  function curve(key,p,TM){var a=[];for(var t=0;t<=TM;t+=TM/180){a.push([t,clamp(fval(key,t,p))]);}return a;}
  function cross(key,p,TM,tg){for(var t=0;t<=TM;t+=TM/600){if(clamp(fval(key,t,p))>=tg)return t;}return null;}
  function buildSliders(){
    var m=MODELS[sel.value];sl.innerHTML='';cur=[];
    m.P.forEach(function(spec,i){
      cur.push(spec[4]);
      var row=document.createElement('div');row.style.cssText='display:flex;align-items:center;gap:10px;margin-bottom:6px;';
      var lab=document.createElement('label');lab.textContent=spec[0];lab.style.cssText='font-size:12px;color:__TXT2__;min-width:128px;';
      var inp=document.createElement('input');inp.type='range';inp.min=spec[1];inp.max=spec[2];inp.step=spec[3];inp.value=spec[4];inp.style.flex='1';
      var out=document.createElement('span');out.style.cssText='font-size:13px;font-weight:600;min-width:50px;text-align:right;';
      var dec=spec[3]<0.01?4:(spec[3]<0.1?3:(spec[3]<1?(spec[3]<0.5?2:1):0));
      out.textContent=Number(spec[4]).toFixed(dec);
      inp.addEventListener('input',function(){cur[i]=parseFloat(inp.value);out.textContent=Number(cur[i]).toFixed(dec);draw();});
      row.appendChild(lab);row.appendChild(inp);row.appendChild(out);sl.appendChild(row);
    });
    document.getElementById('kmx-eq').textContent=m.eq;
  }
  var PX0=46,PX1=624,PY0=266,PY1=14;
  function draw(){
    var key=sel.value,TM=parseFloat(tmaxSel.value);
    function sx(t){return PX0+(t/TM)*(PX1-PX0);}
    function sy(q){return PY0+(q/100)*(PY1-PY0);}
    function path(pts){return pts.map(function(p,i){return (i?'L':'M')+sx(p[0]).toFixed(1)+' '+sy(p[1]).toFixed(1);}).join(' ');}
    var g='';
    for(var q=0;q<=100;q+=25){g+='<line x1="'+PX0+'" y1="'+sy(q)+'" x2="'+PX1+'" y2="'+sy(q)+'" stroke="__GRID__" stroke-width="0.5"/>';g+='<text x="'+(PX0-7)+'" y="'+(sy(q)+4)+'" text-anchor="end" font-size="10" fill="__TXT2__">'+q+'</text>';}
    var step=TM/4;for(var t=0;t<=TM+0.01;t+=step){g+='<text x="'+sx(t)+'" y="'+(PY0+16)+'" text-anchor="middle" font-size="10" fill="__TXT2__">'+Math.round(t)+'</text>';}
    g+='<line x1="'+PX0+'" y1="'+PY0+'" x2="'+PX1+'" y2="'+PY0+'" stroke="__AXIS__" stroke-width="1"/>';
    g+='<line x1="'+PX0+'" y1="'+PY0+'" x2="'+PX0+'" y2="'+PY1+'" stroke="__AXIS__" stroke-width="1"/>';
    g+='<text x="'+((PX0+PX1)/2)+'" y="296" text-anchor="middle" font-size="11" fill="__TXT2__">Time (min)</text>';
    if(pinned){g+='<path d="'+path(pinned)+'" fill="none" stroke="__TXT2__" stroke-width="1.5" stroke-dasharray="4 4"/>';}
    g+='<path d="'+path(curve(key,cur,TM))+'" fill="none" stroke="__AMBER__" stroke-width="2.5"/>';
    document.getElementById('kmx-plot').innerHTML=g;
    var t50=cross(key,cur,TM,50),t80=cross(key,cur,TM,80),qe=clamp(fval(key,TM,cur));
    document.getElementById('kmx-t50').textContent=t50!==null?t50.toFixed(1)+' min':'> '+TM;
    document.getElementById('kmx-t80').textContent=t80!==null?t80.toFixed(1)+' min':'> '+TM;
    document.getElementById('kmx-qend').textContent=qe.toFixed(1)+' %';
  }
  sel.addEventListener('change',function(){pinned=null;buildSliders();draw();});
  tmaxSel.addEventListener('change',draw);
  document.getElementById('kmx-pin').addEventListener('click',function(){pinned=curve(sel.value,cur.slice(),parseFloat(tmaxSel.value));draw();});
  buildSliders();draw();
})();
</script>
"""


def _models_js() -> str:
    """Serialise the model map to a JS object literal (functions reconstructed
    client-side from the expression string)."""
    items = []
    for name, (eq, params, expr) in _M.items():
        pj = json.dumps(params, ensure_ascii=False)
        nm = json.dumps(name, ensure_ascii=False)
        eqj = json.dumps(eq, ensure_ascii=False)
        items.append(f"{nm}:{{eq:{eqj},P:{pj},f:function(t,p,erf,phi,bl){{return ({expr});}}}}")
    return "{" + ",".join(items) + "}"


def render(preselect: str = "Korsmeyer-Peppas", height: int = 560):
    if preselect not in EXPLORABLE:
        preselect = "Korsmeyer-Peppas"
    nparams = len(_M[preselect][1])
    h = height + nparams * 30
    html = (_HTML
            .replace("__MODELS__", _models_js())
            .replace("__PRESELECT__", json.dumps(preselect, ensure_ascii=False))
            .replace("__CARD__", _CARD).replace("__BG__", _BG)
            .replace("__AMBER__", _AMBER).replace("__TXT2__", _TXT2)
            .replace("__TXT__", _TXT).replace("__GRID__", _GRID)
            .replace("__AXIS__", _AXIS))
    components.html(html, height=h, scrolling=False)
