import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip()))

M("""
# ¿Cuándo conviene una Covered Call? Modelos de volatilidad vs. volatilidad implícita
**Finanzas Cuantitativas — FCEN/UBA · Trabajo integrador (boceto end-to-end)**

**Pregunta.** Si nuestro pronóstico de volatilidad del próximo ciclo mensual (σ̂) es *menor* que la volatilidad implícita (IV) que cobra el mercado, la call está cara y conviene venderla (Covered Call); si no, nos quedamos con el subyacente.

**Activos:** SPY (IV = VIX) y QQQ (IV = VXN). NVDA queda afuera: no hay serie histórica de IV. **Muestra:** 2007 → sep-2026 (2007 = warm-up; evaluación out-of-sample desde ene-2008).

**Reglas congeladas antes de mirar resultados** (protocolo de C8):
- Decisión con info hasta t0−1, ejecución al cierre de t0 (vencimiento mensual estándar, 3er viernes).
- Timing: vender si σ̂_ens < IV. Strike: K = S0·exp(z·σ̂_ens·√T), z = 0.5 (prob. de ejercicio según el modelo ≈ 31%).
- Precio: Black-Scholes con la IV del índice ajustada por skew (calibrada con calls de SPY de IBKR), costo = 2% de la prima.
""")
C("""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path.cwd().resolve()
while not (ROOT / "poster").exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
from poster.src.data import load_panel, load_ibkr, validate, build_cycles
from poster.src import vol_models as VM, strategy as ST, checks as CK
from poster.src.build import get_forecasts
FIG = ROOT / "poster" / "overleaf" / "figures"
pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
""")
M("## 1. Datos")
C("""
A = load_panel()
validate(A)
""")
M("""
Panel alineado: SPY y VIX salen del Excel del profe (hoja `SPY ` con filas de dividendos mezcladas, parseadas aparte); QQQ y ^VXN vienen de yfinance; la tasa libre de riesgo es ^IRX. El retorno total se arma con Close + dividendos y se contrasta contra Adj Close más abajo.
""")
M("""
## 2. Calibración del skew con las calls de IBKR (SPY, jun–sep 2026)
IBKR no da historia de opciones vencidas: solo hay calls vigentes (20–70 DTE). Invertimos Black-Scholes sobre el precio operado para estimar IV(K)/VIX en función de la moneyness m = ln(K/S). **Limitación:** una sola ventana de 3 meses, régimen de vol baja, precios de último trade (pueden ser viejos); para QQQ se asume el mismo skew relativo.
""")
C("""
from scipy.optimize import brentq
from src.pricing import black_scholes_call_price as bs
ib = load_ibkr(); s = ib[ib.symbol == "SPY"].copy()
s["date"] = pd.to_datetime(s["date"]); s["expiration"] = pd.to_datetime(s["expiration"])
s["T"] = (s.expiration - s.date).dt.days / 365.0
s["r"] = A["SPY"]["rf"].reindex(s.date).values
s = s[(s.days_to_expiration.between(20, 70)) & (s.volume >= 10)].copy()
def iv_of(row):
    f = lambda sg: bs(row.underlying_close, row.strike, row["T"], row.r, sg, 0.011) - row.close
    try: return brentq(f, 0.02, 2.0)
    except Exception: return np.nan
s["iv"] = s.apply(iv_of, axis=1)
s["m"] = np.log(s.strike / s.underlying_close); s["vix"] = A["SPY"]["iv"].reindex(s.date).values
s = s.dropna(subset=["iv", "vix"]); s["ratio"] = s.iv / s.vix
o = s[(s.m > -0.02) & (s.m < 0.06)]
b1, b0 = np.polyfit(o.m, o.ratio, 1)
print(f"IV(K)/VIX ≈ {b0:.2f} + ({b1:.2f})·m    n = {len(o)} opciones   (el pipeline usa A={ST.SKEW_A}, B={ST.SKEW_B})")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(o.m * 100, o.ratio, s=6, alpha=.25, color="#2a78d6")
xs = np.linspace(-2, 6, 50); ax.plot(xs, b0 + b1 * xs / 100, color="#eb6834", lw=2.5)
ax.set_xlabel("moneyness ln(K/S) (%)"); ax.set_ylabel("IV de la call / VIX"); ax.set_title("Skew de las calls de SPY (IBKR)")
plt.show()
""")
M("""
**Kalman sobre la IV de IBKR (donde el suavizado sí sirve).** La IV diaria ATM de IBKR sale de precios esporádicos y salta; un filtro de nivel local (MLE de la razón señal/ruido) la ordena. Es el uso donde suavizar es el objetivo, a diferencia de pronosticar vol donde el filtro se evalúa out-of-sample.
""")
C("""
atm = s[s.m.abs() < 0.01].groupby("date").iv.median()
y = np.log(atm.values)
th = VM.kalman_fit(y); mu, phi, q, rr = th[0], np.tanh(th[1]), np.exp(th[2]), np.exp(th[3])
_, af, _ = VM._kf(y, mu, phi, q, rr, ret_all=True)
print(f"phi={phi:.3f}   q/r={q/rr:.3f}")
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(atm.index, atm.values, ".", color="#9a9990", label="IV ATM (mediana diaria, IBKR)")
ax.plot(atm.index, np.exp(af), color="#1baf7a", lw=2.5, label="Filtro de Kalman (causal)")
ax.plot(atm.index, A["SPY"]["iv"].reindex(atm.index).values, color="#eb6834", lw=1.8, label="VIX")
ax.legend(frameon=False); plt.show()
""")
M("""
## 3. Modelos de volatilidad (walk-forward, ventana expansiva)
Rolling 21d · EWMA(0.94) · GJR-GARCH(1,1)-t · HAR-RV (Corsi) · Kalman (filtro, nunca smoother) · **Ensamble** = promedio de varianzas. La IV se mide como referencia del mercado, no como modelo propio. Métricas: RMSE (en vol), **QLIKE** (sobre varianza, robusta al ruido del proxy) y R² de Mincer-Zarnowitz.
""")
C("""
fcs = get_forecasts(A)
errs = {k: VM.forecast_table(f) for k, f in fcs.items()}
for k in errs:
    print(k, "— ciclos OOS:", len(fcs[k])); display(errs[k])
print("Último ajuste (SPY):", fcs["SPY"].attrs.get("garch_last"), fcs["SPY"].attrs.get("kalman_last"))
""")
C("""
for n in ["fig_vrp", "fig_forecast", "fig_errors"]: display(Image(str(FIG / f"{n}.png"), width=620))
""")
M("""
**Lectura.** (i) La IV promedia ~4 pp (SPY) y ~3 pp (QQQ) *por encima* de la volatilidad realizada: es la prima de riesgo de varianza que la Covered Call intenta cobrar. (ii) La IV es de los mejores predictores individuales (como dice la literatura) y el Kalman — estimado por MLE, con q/r ≈ 0.2, o sea que **no** suaviza de más — la iguala o la supera en QLIKE. (iii) Los modelos ingenuos (rolling, EWMA) son claramente peores. (iv) El ensamble queda entre los mejores y es más estable que cualquier modelo individual.
""")
M("## 4. Estrategias")
C("""
from poster.src.build import main as build_main
fcs, res, summ, errs, boot, extra = build_main()
for k in summ:
    print(k); display(summ[k])
""")
C("""
for n in ["fig_signal", "fig_equity", "fig_ablation"]: display(Image(str(FIG / f"{n}.png"), width=620))
""")
M("""
### Inferencia: ¿la diferencia es distinguible del azar?
Bootstrap estacionario (bloques de 3 ciclos) sobre la diferencia de retorno por ciclo, anualizada. `p_neg` = fracción de réplicas con media ≤ 0.
""")
C("""
for k in boot:
    print(k, "(CC modelo menos la comparación)")
    display(pd.DataFrame(boot[k]).T)
print({k: {a: round(b, 3) for a, b in v.items()} for k, v in extra.items()})
""")
M("""
## 5. Sensibilidades (solo notebook; **ojo: es multiple testing** — el póster usa los valores fijados de antemano z=0.5, m=0, costo=2%)
""")
C("""
cyc = {k: build_cycles(A[k]["px"].index) for k in A}
rows = []
for k in A:
    for z in [0, .25, .5, 1]:
        for m in [0, .05, .10]:
            d = ST.simulate(A[k], fcs[k], ST.standard_specs(z=z, m=m)["CC modelo"], cyc[k]); mt = ST.metrics(d)
            rows.append(dict(activo=k, z=z, m=m, CAGR=mt["CAGR"], sharpe=mt["sharpe"], maxdd=mt["maxdd"], pct_vendido=mt["pct_vendido"]))
sens = pd.DataFrame(rows)
for k in A:
    print(k, "— Sharpe por (z, m)"); display(sens[sens.activo == k].pivot(index="z", columns="m", values="sharpe"))
""")
C("""
rows = []
for k in A:
    for cost in [0, .02, .05]:
        for skew in [True, False]:
            d = ST.simulate(A[k], fcs[k], ST.standard_specs(cost=cost, skew=skew)["CC modelo"], cyc[k])
            e = ST.simulate(A[k], fcs[k], ST.standard_specs(cost=cost, skew=skew)["CC estático ATM"], cyc[k])
            rows.append(dict(activo=k, costo=cost, skew_IBKR=skew, CAGR_modelo=ST.metrics(d)["CAGR"], CAGR_estatico=ST.metrics(e)["CAGR"]))
pd.DataFrame(rows)
""")
M("### Una estrategia por modelo de volatilidad (mismo timing + strike, distinto σ̂)")
C("""
rows = []
for k in A:
    for m in VM.MODELS + ["Ensamble"]:
        d = ST.simulate(A[k], fcs[k], ST.standard_specs(sigma=m)["CC modelo"], cyc[k]); mt = ST.metrics(d)
        rows.append(dict(activo=k, sigma=m, CAGR=mt["CAGR"], sharpe=mt["sharpe"], maxdd=mt["maxdd"], pct_vendido=mt["pct_vendido"]))
pd.DataFrame(rows).pivot(index="sigma", columns="activo", values=["CAGR", "sharpe", "pct_vendido"])
""")
M("## 6. Chequeos de cordura")
C("""
print("Sin look-ahead (|Δ| tras corromper el futuro):", CK.check_no_lookahead(A["SPY"]))
print("Estático == 'siempre + z=0' (max |Δret|):", CK.check_static_equals_z0(A["SPY"], fcs["SPY"]))
bw = CK.check_vs_buywrite(A["SPY"], fcs["SPY"])
print(f"CC estático simulado vs ETF BuyWrite real (PBP): corr mensual {bw.attrs['corr']:.3f} | CAGR {bw.attrs['cagr_sim']:.2%} vs {bw.attrs['cagr_pbp']:.2%} (PBP cobra ~0.75% anual)")
bh = res["SPY"]["Buy & Hold"]; adj = A["SPY"]["px"]["Adj Close"]
print("B&H por ciclos vs Adj Close:", round((1 + bh.ret).prod() - 1, 4), round(adj.loc[bh.texp.iloc[-1]] / adj.loc[bh.index[0]] - 1, 4))
""")
M("""
## 7. Interpretación (provisoria — para discutir)
- **Contra el CC estático, el modelo gana** en ambos activos, con IC95% (bootstrap) que excluye 0.
- **Contra Buy & Hold, el CC modelo no gana en retorno** (SPY: el IC95% de la diferencia incluye 0; QQQ: pierde claramente), pero reduce volatilidad y drawdown. Hay un costo estructural: el techo al upside en el mercado alcista 2009-2026.
- **Ablación (timing vs strike):** en SPY el timing solo (σ̂<IV con strike ATM) *no* mejora al CC estático (Sharpe 0.52 vs 0.56); la mejora viene de fijar el strike con σ̂ (0.63). En QQQ el timing sí suma (0.65 con ambos vs 0.58 solo strike).
- **Los modelos sofisticados casi no se distinguen en la estrategia:** corriendo la misma regla con cada σ̂, los Sharpe quedan entre 0.62 y 0.66 (SPY) y 0.62 y 0.65 (QQQ); incluso el rolling ingenuo rinde parecido al ensamble. Mejor pronóstico (QLIKE) no se traduce en mucho mejor P&L.
- **El skew importa muchísimo:** con IV plana (sin ajuste IBKR) el CC estático parecería igualar al B&H (CAGR 11.2% vs 11.9% en SPY), un resultado irreal; con el skew calibrado el CC estático replica al ETF BuyWrite real (corr. 0.97).
- **Heterogeneidad por activo:** el efecto del *upside capping* pesa mucho más en QQQ por su mayor crecimiento.
- **Limitaciones:** IV del índice (no de la opción exacta), skew calibrado en un único régimen, precios BS (no bid/ask real), costos supuestos, dos activos, NVDA excluido.
""")
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "cc_vol_resultados.ipynb"
nbf.write(nb, out)
print(out)
