# Conclusiones: Covered Calls condicionadas por pronósticos de volatilidad (SPY y QQQ, 2008–2026)

Versión corregida del proyecto (rama `correcciones`). Todos los números salen de `results.json` y `tablas.md`, generados por `python -m poster.src.build`. Gráficos en `graficos/`.

---

## 1. Resumen

1. **La prima de volatilidad del VIX no está en las calls.** El VIX promedia 20,2% en SPY y la volatilidad realizada 16,3%, pero la call que se vende cotiza a una IV de **14,7%**, por debajo de la realizada. En QQQ: VXN 22,8%, realizada 19,7%, call vendida **16,7%**. El VIX incluye el skew de los puts. Ver `03_iv_call_vs_realizada.png`.
2. **Vender calls de forma sistemática destruye valor.** La pata de la opción le resta al Buy & Hold **−7,1 pp/año en SPY y −12,0 pp/año en QQQ** con calls ATM, y −2,5 / −6,5 pp/año con calls OTM. Ver `07_aporte_call.png`.
3. **Con la señal corregida, el modelo casi nunca vende:** 13% de los meses en SPY y 9% en QQQ. Termina siendo prácticamente Buy & Hold. La diferencia es indistinguible de cero: SPY +0,0 pp/año [−0,8; +0,7], QQQ −0,1 pp/año [−1,1; +0,8].
4. **Lo que sí demuestra el modelo es que evita una pérdida.** Le gana a vender siempre una call OTM (+2,5 pp/año en SPY, +6,5 en QQQ, ambos significativos). También le gana a la versión original del proyecto, que comparaba con el VIX entero (+2,0 y +5,4 pp/año).
5. **Conclusión práctica:** en índices y en un mercado alcista, la covered call no se justifica por la prima de riesgo de varianza. Un buen pronóstico de volatilidad sirve para *no* vender, no para ganarle al mercado.

---

## 2. Resultados del backtest (224 ciclos mensuales, ene-2008 a sep-2026)

### SPY

| Estrategia | CAGR | Vol. | Sharpe | Sortino | MaxDD diario | Vende | Aporte de la call |
|---|---|---|---|---|---|---|---|
| Buy & Hold | 11,9% | 18,5% | 0,63 | 0,88 | −51% | — | — |
| CC estático ATM | 5,2% | 12,2% | 0,37 | 0,41 | −38% | 100% | −7,1 pp/año |
| CC siempre OTM (strike con la IV, sin modelo) | 9,7% | 15,5% | 0,59 | 0,74 | −45% | 100% | −2,5 pp/año |
| CC modelo, señal original (σ̂ < VIX) | 10,0% | 16,7% | 0,58 | 0,77 | −47% | 85% | −2,0 pp/año |
| **CC modelo, señal corregida (σ̂ < IV de la call)** | **11,9%** | 18,3% | **0,64** | 0,88 | −51% | 13% | +0,0 pp/año |

### QQQ

| Estrategia | CAGR | Vol. | Sharpe | Sortino | MaxDD diario | Vende | Aporte de la call |
|---|---|---|---|---|---|---|---|
| Buy & Hold | 16,9% | 21,2% | 0,78 | 1,17 | −49% | — | — |
| CC estático ATM | 5,3% | 12,8% | 0,36 | 0,42 | −41% | 100% | −12,0 pp/año |
| CC siempre OTM (strike con la IV, sin modelo) | 10,5% | 16,7% | 0,60 | 0,78 | −47% | 100% | −6,5 pp/año |
| CC modelo, señal original (σ̂ < VXN) | 11,4% | 18,2% | 0,61 | 0,83 | −50% | 78% | −5,5 pp/año |
| **CC modelo, señal corregida** | **16,9%** | 21,0% | **0,79** | 1,17 | −49% | 9% | −0,1 pp/año |

### El CC modelo contra cada alternativa (bootstrap estacionario, IC 95%, pp/año)

| Contra | SPY | QQQ |
|---|---|---|
| Buy & Hold | +0,0 [−0,8; +0,7] | −0,1 [−1,1; +0,8] |
| CC estático ATM | +7,2 [+3,4; +11,0] | +11,9 [+7,3; +16,8] |
| CC siempre OTM (IV) | +2,5 [+0,2; +5,0] | +6,5 [+3,6; +9,6] |
| CC modelo con la señal original | +2,0 [+0,2; +4,0] | +5,4 [+3,0; +7,9] |

### Sobreajuste

Se probaron **36 configuraciones**: 6 pronósticos × 3 valores de z × 2 señales.
- El Deflated Sharpe del CC modelo da 0,96 (SPY) y 0,99 (QQQ). Su Sharpe es "real", pero porque replica al Buy & Hold.
- La pregunta relevante es si le gana al Buy & Hold. El PSR del retorno activo da **0,50 (SPY) y 0,45 (QQQ)**: no hay evidencia de que le gane.

### Robustez (Sharpe del CC modelo)

| Variante | SPY | QQQ |
|---|---|---|
| Base | 0,64 | 0,79 |
| Costo 5% de la prima (en lugar de 2%) | 0,63 | 0,79 |
| Skew calibrado con calls de SPY en lugar de SPX | 0,64 | 0,77 |
| σ̂ = Kalman en lugar del ensamble | 0,65 | 0,79 |
| z = 1 (strike más alejado) | 0,63 | 0,78 |
| **Sin skew (IV plana = VIX)** | **0,87** | **0,90** |

La conclusión es robusta a todo **salvo al skew**. Sin skew, hasta el CC estático tendría mejor Sharpe que el Buy & Hold (0,74 contra 0,63 en SPY). Por eso el skew se validó contra un ETF real: ver la sección 4.

---

## 3. Calidad de los pronósticos de volatilidad (fuera de muestra)

| QLIKE (menor = mejor) | SPY | QQQ |
|---|---|---|
| Kalman | **0,430** | 0,318 |
| IV (VIX / VXN) | 0,445 | **0,306** |
| HAR-RV | 0,442 | 0,334 |
| Ensamble (usado) | 0,469 | 0,350 |
| GJR-GARCH | 0,534 | 0,401 |
| EWMA (λ = 0,97) | 0,666 | 0,458 |
| Rolling 21d | 0,706 | 0,512 |

- El Kalman es el mejor modelo propio y empata con la volatilidad implícita.
- El ensamble equiponderado queda peor que el Kalman porque incluye modelos débiles. Se mantuvo como especificación fijada de antemano, y el Kalman se reporta como robustez: el resultado no cambia.

---

## 4. Validación del simulador contra el mercado real

- El CC estático ATM simulado sobre SPY replica al ETF **PBP** (Invesco S&P 500 BuyWrite, que replica el índice BXM) con **correlación mensual de 0,97**.
- **CAGR:** simulación con opciones europeas 6,3% anual, contra PBP 6,6% antes de comisiones (5,9% neto de 0,75%). Ver `10_validacion_pbp.png`.
- **El ejercicio anticipado cuesta ~1,1 pp/año al CC ATM sobre SPY:** con opciones americanas la simulación da 5,2% en lugar de 6,3%. SPY paga dividendo en la semana del vencimiento trimestral, y las calls en el dinero se ejercen la víspera, así que el vendedor pierde el dividendo. El BXM usa opciones de SPX, que son europeas, y no sufre esto.
- **El skew se calibró con 821 calls de SPX** (20–45 días, sesión de la tarde): IV(K)/VIX = 0,83 − 0,18·x, con x = ln(K/S)/(VIX·√T). Con calls de SPY sale casi idéntico: 0,84 − 0,19·x. Ver `02_skew_calibracion.png`.

---

## 5. Fase 1 (features)

| Tema | Antes | Ahora | Efecto |
|---|---|---|---|
| Prima de varianza ex-ante (SPY) | VIX − Garman-Klass | VIX − Yang-Zhang | Promedio de **6,6 a 3,2 pp**: Garman-Klass ignora el gap overnight y duplicaba la prima (`fase1_vrp_gk_vs_yz.png`) |
| GJR-GARCH como feature | Ajuste in-sample (miraba el futuro) | Pronóstico walk-forward | El pronóstico se actualiza todos los días. Antes quedaba congelado 21 días entre reajustes (`fase1_garch_walkforward.png`) |
| Half-life del spread VIX/VIX3M | MCO, ventana 60 | Ventana 252 con corrección de sesgo de Kendall | Mediana de **3,8 a 6,9 ruedas**: la versión anterior subestimaba la persistencia a la mitad |
| Regímenes OU | Cortes −1 / 0,5 / 2 | Cortes de la consigna (0 y 2) | Calma 58%, alerta 39%, estrés extremo 3,4% de los días (`fase1_ou_sscore.png`) |
| Kalman (VIX/VIX3M y CAPM) | q a mano, r con toda la muestra | (q, r) por máxima verosimilitud en el primer año | Sin información futura. En VIX/VIX3M: q = 8,4e−5, r = 1,9e−4 |
| Diferenciación fraccionaria | Precio en nivel, d* con toda la muestra | Log-precio, d* solo con 2007–2014 | d* = 0,50 (SPY, QQQ) y 0,10 (NVDA) |

**Advertencia sobre la diferenciación fraccionaria.** En SPY y QQQ, el test KPSS rechaza la estacionariedad para todo d < 1. Solo ADF la acepta desde d = 0,5, y a ese d la correlación con el nivel cae a ~0,7. La promesa de la técnica (estacionariedad con memoria, correlación > 0,9) **no se cumple para estos índices**.

---

## 6. Limitaciones que siguen

- **Skew de un solo régimen.** Se calibró con jun–sep 2026 (volatilidad baja) y se aplica a 2008–2026. Medir la moneyness en desvíos (x) atenúa el problema, pero no lo elimina. Es el supuesto del que más dependen los resultados.
- **QQQ usa el skew de SPX.** Las calls de Nasdaq podrían tener otro skew relativo al VXN.
- **Precios teóricos.** Black-Scholes sobre IV del índice, con costo fijo de 2% de la prima. Los precios de IBKR son de último trade, no bid/ask.
- **Solo 2 activos**, ambos índices en un período mayormente alcista. NVDA no tiene una IV histórica pública.
- **Desvíos de la consigna (`docs/esquema_tp.md`):** el backtest no usa meta-labeling ni triple barrera, y las features de la fase 1 (OU, fracdiff, CAPM dinámico) no alimentan la decisión de venta.

---

## 7. Qué se corrigió (detalle)

**Backtest (`poster/src/`)**
- **Señal:** σ̂ se compara con la IV de la call que se vende (al strike elegido), no con el VIX entero.
- **Convención de tiempo:** cada cantidad se anualiza en su base (252 ruedas para la volatilidad realizada y pronosticada, 365 días corridos para el VIX), y la comparación se hace en desvío por ciclo.
- **Skew:**
  - calibrado en el código, con calls europeas de SPX;
  - mismas convenciones que el backtest (T en días corridos, dividendo por fecha);
  - solo trades de la tarde;
  - moneyness estandarizada y extrapolación plana fuera del rango.
- **Ejercicio anticipado** antes de cada ex-dividendo (opciones americanas de SPY y QQQ).
- **MaxDD diario,** con la call marcada a mercado. Antes era mensual: −45% en lugar de −51% en SPY.
- **Métricas:** Sharpe sobre el desvío de los excesos, Sortino, PSR sobre excesos y Deflated Sharpe sobre las 36 configuraciones.
- **Comparación justa:** "CC siempre OTM (IV)", con el mismo tipo de strike pero sin modelo.
- **Tasa libre de riesgo:** ^IRX se convierte de tasa de descuento a tasa continua. Se eliminó el `bfill` de la tasa.
- **EWMA:** λ = 0,97, el valor de RiskMetrics para horizonte mensual.
- **Validación contra PBP** comparando parejo: opción europea, comisiones explícitas.
- **Datos:** `yfinance` solo se importa si falta el cache.

**Fase 1 (`src/`)**
- **GARCH:** walk-forward corregido y usado como feature, en lugar del ajuste in-sample.
- **Prima de varianza:** con Yang-Zhang.
- **Kalman:** varianzas por máxima verosimilitud en la ventana de calibración.
- **OU:** ventana 252, corrección de sesgo y regímenes de la consigna.
- **Diferenciación fraccionaria:** sobre el log-precio, d* sin información futura, KPSS como diagnóstico.
- **Descarga:** sin `bfill`.
- **Features regeneradas:** el archivo anterior estaba desactualizado (2020–2026); ahora cubre 2008–2026.

**Tests (`tests/test_core.py`, 13 tests):**
- Black-Scholes contra el ejemplo de Hull, inversión de IV y conversión de tasa.
- Recuperación de parámetros OU y efecto de la corrección de sesgo.
- Diferenciación fraccionaria con d = 1 igual a la primera diferencia.
- Causalidad de GARCH, Kalman y de los pronósticos del backtest.
- Consistencia entre la equity diaria y la mensual, y estático igual a z = 0.

---

## 8. Cómo reproducir

```bash
pip install -r requirements.txt
python src/build_features_phase1.py      # features de la fase 1 -> data/features_phase1.parquet
python src/plot_phase1.py                # gráficos fase1_*.png
python -m poster.src.build --refresh     # pronósticos + backtest -> resultados/
python -m pytest tests -q
```
