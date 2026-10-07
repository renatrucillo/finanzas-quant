# Tablas de resultados

### Estrategias — SPY

| SPY | Total | CAGR | Vol. | Sharpe | Sortino | MaxDD diario | MaxDD mensual | Vende | Aporte call/año | Ej. anticip. |
|---|---|---|---|---|---|---|---|---|---|---|
| Buy & Hold | 712% | 11.9% | 18.5% | 0.63 | 0.88 | -51% | -45% | 0% | +0.0 pp | 0 |
| CC estático ATM | 158% | 5.2% | 12.2% | 0.37 | 0.41 | -38% | -34% | 100% | -7.1 pp | 49 |
| CC siempre OTM (IV) | 461% | 9.7% | 15.5% | 0.59 | 0.74 | -45% | -40% | 100% | -2.5 pp | 31 |
| Siempre + strike modelo | 419% | 9.2% | 15.3% | 0.57 | 0.71 | -40% | -40% | 100% | -2.9 pp | 33 |
| Timing modelo + ATM | 516% | 10.2% | 17.5% | 0.57 | 0.78 | -44% | -44% | 32% | -1.7 pp | 17 |
| CC modelo (señal VIX) | 497% | 10.0% | 16.7% | 0.58 | 0.77 | -47% | -42% | 85% | -2.0 pp | 31 |
| CC modelo | 718% | 11.9% | 18.3% | 0.64 | 0.88 | -51% | -45% | 13% | +0.0 pp | 2 |

### CC modelo contra cada alternativa — SPY (bootstrap estacionario)

| CC modelo — SPY | Diferencia anual | IC 95% |
|---|---|---|
| vs Buy & Hold | +0.0 pp | [-0.8; +0.7] |
| vs CC estático ATM | +7.2 pp | [+3.4; +11.0] |
| vs CC siempre OTM (IV) | +2.5 pp | [+0.2; +5.0] |
| vs Siempre + strike modelo | +2.9 pp | [+0.7; +5.4] |
| vs CC modelo (señal VIX) | +2.0 pp | [+0.2; +4.0] |

### Error de pronóstico fuera de muestra — SPY (RMSE y sesgo en pp de vol.)

| Pronóstico — SPY | RMSE | QLIKE | sesgo | R2_MZ |
|---|---|---|---|---|
| Rolling21 | 10.0 | 0.706 | +0.2 | 0.38 |
| EWMA | 9.9 | 0.666 | +1.0 | 0.34 |
| GJR-GARCH | 10.1 | 0.534 | +2.0 | 0.43 |
| HAR-RV | 9.6 | 0.442 | +2.9 | 0.43 |
| Kalman | 8.9 | 0.430 | +1.7 | 0.40 |
| Ensamble | 9.1 | 0.469 | +1.8 | 0.44 |
| IV | 9.6 | 0.445 | +3.9 | 0.43 |

### Robustez (Sharpe) — SPY

| Sharpe — SPY | Buy & Hold | CC estático ATM | CC siempre OTM (IV) | CC modelo |
|---|---|---|---|---|
| costo 5% | 0.63 | 0.31 | 0.58 | 0.63 |
| skew de SPY | 0.63 | 0.39 | 0.60 | 0.64 |
| sin skew (IV plana) | 0.63 | 0.74 | 0.97 | 0.87 |
| σ̂ = Kalman | 0.63 | 0.37 | 0.59 | 0.65 |
| z = 1 | 0.63 | 0.37 | 0.63 | 0.63 |

### Estrategias — QQQ

| QQQ | Total | CAGR | Vol. | Sharpe | Sortino | MaxDD diario | MaxDD mensual | Vende | Aporte call/año | Ej. anticip. |
|---|---|---|---|---|---|---|---|---|---|---|
| Buy & Hold | 1749% | 16.9% | 21.2% | 0.78 | 1.17 | -49% | -47% | 0% | +0.0 pp | 0 |
| CC estático ATM | 161% | 5.3% | 12.8% | 0.36 | 0.42 | -41% | -38% | 100% | -12.0 pp | 28 |
| CC siempre OTM (IV) | 546% | 10.5% | 16.7% | 0.60 | 0.78 | -47% | -44% | 100% | -6.5 pp | 16 |
| Siempre + strike modelo | 495% | 10.0% | 16.4% | 0.58 | 0.74 | -44% | -44% | 100% | -7.0 pp | 16 |
| Timing modelo + ATM | 1320% | 15.3% | 20.1% | 0.74 | 1.09 | -47% | -47% | 21% | -1.7 pp | 6 |
| CC modelo (señal VIX) | 647% | 11.4% | 18.2% | 0.61 | 0.83 | -50% | -47% | 78% | -5.5 pp | 12 |
| CC modelo | 1745% | 16.9% | 21.0% | 0.79 | 1.17 | -49% | -47% | 9% | -0.1 pp | 0 |

### CC modelo contra cada alternativa — QQQ (bootstrap estacionario)

| CC modelo — QQQ | Diferencia anual | IC 95% |
|---|---|---|
| vs Buy & Hold | -0.1 pp | [-1.1; +0.8] |
| vs CC estático ATM | +11.9 pp | [+7.3; +16.8] |
| vs CC siempre OTM (IV) | +6.5 pp | [+3.6; +9.6] |
| vs Siempre + strike modelo | +6.9 pp | [+4.1; +10.1] |
| vs CC modelo (señal VIX) | +5.4 pp | [+3.0; +7.9] |

### Error de pronóstico fuera de muestra — QQQ (RMSE y sesgo en pp de vol.)

| Pronóstico — QQQ | RMSE | QLIKE | sesgo | R2_MZ |
|---|---|---|---|---|
| Rolling21 | 10.1 | 0.512 | +0.2 | 0.34 |
| EWMA | 9.9 | 0.458 | +0.9 | 0.30 |
| GJR-GARCH | 9.8 | 0.401 | +1.4 | 0.35 |
| HAR-RV | 9.0 | 0.334 | +1.5 | 0.36 |
| Kalman | 8.7 | 0.318 | +1.0 | 0.37 |
| Ensamble | 8.9 | 0.350 | +1.2 | 0.38 |
| IV | 9.4 | 0.306 | +3.2 | 0.40 |

### Robustez (Sharpe) — QQQ

| Sharpe — QQQ | Buy & Hold | CC estático ATM | CC siempre OTM (IV) | CC modelo |
|---|---|---|---|---|
| costo 5% | 0.78 | 0.30 | 0.59 | 0.79 |
| skew de SPY | 0.78 | 0.39 | 0.61 | 0.77 |
| sin skew (IV plana) | 0.78 | 0.77 | 1.01 | 0.90 |
| σ̂ = Kalman | 0.78 | 0.36 | 0.60 | 0.79 |
| z = 1 | 0.78 | 0.36 | 0.75 | 0.78 |
