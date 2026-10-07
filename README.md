# Covered Calls condicionadas por volatilidad — Finanzas Cuantitativas (FCEN-UBA)

¿Conviene vender una covered call sobre SPY o QQQ cuando un pronóstico de volatilidad dice que la call está cara?
El proyecto pronostica la volatilidad con 5 modelos walk-forward, decide cada mes si vender y a qué strike, y compara contra Buy & Hold y covered calls estáticas (2008–2026).

**Resultados y conclusiones:** [`resultados/conclusiones.md`](resultados/conclusiones.md) · tablas en [`resultados/tablas.md`](resultados/tablas.md) · gráficos en `resultados/graficos/`.

## Datos y referencias (QR del póster)

### Datos

| Serie | Uso | Fuente |
|---|---|---|
| SPY (OHLC, dividendos) y VIX | subyacente e IV del índice | Excel de la cátedra (`data/raw/datos_finanzas.xlsx`) |
| QQQ (OHLC, dividendos) y VXN | subyacente e IV del índice | Yahoo Finance (yfinance) |
| T-bill 13 semanas (^IRX) | tasa libre de riesgo (convertida de tasa de descuento a tasa continua) | Yahoo Finance |
| ETF PBP (Invesco S&P 500 BuyWrite) | validación del simulador contra un covered call real | Yahoo Finance |
| Calls de SPX y SPY, jun–sep 2026 | calibración del skew IV(K)/VIX | Interactive Brokers (`data/raw/ibkr_calls_daily.parquet`) |
| VIX3M, NVDA, panel 2007–2026 | features de la fase 1 | Yahoo Finance (`data/panel_ohlcv.parquet`) |

Muestra del backtest: 224 ciclos mensuales fuera de muestra (ene-2008 a sep-2026), entre vencimientos estándar (tercer viernes), con 2007 como calentamiento.

### Referencias

- Bailey, D. y López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*.
- Bollerslev, T., Tauchen, G. y Zhou, H. (2009). Expected Stock Returns and Variance Risk Premia. *Review of Financial Studies*.
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*.
- Garman, M. y Klass, M. (1980). On the Estimation of Security Price Volatilities from Historical Data. *Journal of Business*.
- Glosten, L., Jagannathan, R. y Runkle, D. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance*.
- Hull, J. (2018). *Options, Futures and Other Derivatives*. Pearson.
- Israelov, R. y Nielsen, L. (2015). Covered Calls Uncovered. *Financial Analysts Journal*.
- J.P. Morgan/Reuters (1996). *RiskMetrics — Technical Document*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Politis, D. y Romano, J. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*.
- Whaley, R. (2002). Return and Risk of CBOE Buy Write Monthly Index. *Journal of Derivatives*.
- Yang, D. y Zhang, Q. (2000). Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices. *Journal of Business*.

## Estructura

```text
├── data/
│   ├── raw/                      # Excel de la cátedra (SPY, VIX), calls de IBKR (SPX/SPY, jun–sep 2026)
│   ├── panel_ohlcv.parquet       # panel yfinance 2007–2026 (fase 1)
│   └── features_phase1.parquet   # features de la fase 1 (generado)
├── src/                          # Fase 1: ingeniería de features (Mundo P)
│   ├── download_market_data.py   # descarga yfinance -> data/panel_*.parquet
│   ├── volatility_estimators.py  # close-to-close, Parkinson, Garman-Klass, Yang-Zhang
│   ├── garch_model.py            # GJR-GARCH(1,1): ajuste descriptivo y pronóstico walk-forward
│   ├── kalman_filter.py          # regresión con coeficientes variables (q, r por MLE)
│   ├── ou_process.py             # Ornstein-Uhlenbeck del spread ln(VIX) − ln(VIX3M), s-score
│   ├── fracdiff.py               # diferenciación fraccionaria (FFD) con d* sin lookahead
│   ├── build_features_phase1.py  # orquesta la fase 1
│   └── plot_phase1.py            # gráficos de diagnóstico de la fase 1
├── poster/
│   ├── src/                      # Backtest de covered calls
│   │   ├── data.py               # panel SPY/QQQ, IV (VIX/VXN), tasa, dividendos, ciclos de vencimiento
│   │   ├── vol_models.py         # Rolling, EWMA, GJR-GARCH, HAR-RV, Kalman, ensamble (walk-forward)
│   │   ├── strategy.py           # Black-Scholes, skew, señal, simulación, métricas, DSR
│   │   ├── figures.py            # gráficos del backtest
│   │   ├── checks.py             # chequeos de cordura (usados por los tests)
│   │   └── build.py              # corre todo -> resultados/
│   ├── data/                     # cache de yfinance y pronósticos
│   └── overleaf/, notebooks/     # póster y notebook de la versión ANTERIOR (no actualizados)
├── resultados/                   # salida del backtest: conclusiones, tablas, results.json, gráficos
├── tests/test_core.py            # tests de fórmulas y de ausencia de lookahead
└── docs/                         # consigna (esquema_tp.md) y material de clase
```

## Cómo correr

Python 3.10+.

```bash
pip install -r requirements.txt
python src/build_features_phase1.py      # fase 1 (usa data/panel_ohlcv.parquet)
python src/plot_phase1.py
python -m poster.src.build --refresh     # backtest; sin --refresh reutiliza los pronósticos cacheados
python -m pytest tests -q
```

`src/download_market_data.py` vuelve a descargar el panel de yfinance (requiere internet). El backtest usa el Excel de la cátedra para SPY y VIX, y el cache de `poster/data/` para QQQ, VXN, ^IRX y PBP.

## Convenciones del backtest

- **Decisión y ejecución:** la decisión usa información hasta la rueda anterior al vencimiento (t0−1). Se vende al cierre de t0 y se liquida al cierre del vencimiento siguiente.
- **Anualización:** volatilidades realizadas y pronosticadas en base 252 ruedas; IV tipo VIX en base 365 días corridos. Las comparaciones se hacen en desvío por ciclo.
- **Precio de la call:** Black-Scholes con IV(K) = IV_índice · (a + b·x), x = ln(K/S)/(IV·√T). El skew se calibra con calls de SPX de IBKR.
- **Señal:** se vende si σ̂ < IV de la call al strike elegido.
- **Ejercicio anticipado:** se modela antes de cada ex-dividendo (opciones americanas de SPY y QQQ).
- **Tasa libre de riesgo:** ^IRX convertida de tasa de descuento a tasa continua.
- **Validación:** el CC estático simulado se valida contra el ETF real PBP (BuyWrite).
