"""
Módulo de Volatilidad Condicional Asimétrica GJR-GARCH(1,1) (Clase 4)
=====================================================================
Implementa el modelo GJR-GARCH(1,1) (Glosten, Jagannathan y Runkle, 1993):
    sigma_t^2 = omega + (alpha + gamma * I_{eps_{t-1} < 0}) * eps_{t-1}^2 + beta * sigma_{t-1}^2

Captura:
1. Volatility clustering (persistencia: alpha + beta + gamma/2 < 1, con innovaciones simétricas)
2. Leverage effect (asimetría ante shocks negativos: gamma > 0)
3. Pronósticos condicionales 1-step ahead sin lookahead bias (`rolling_gjr_garch_forecast`)

IMPORTANTE: `fit_gjr_garch` estima con TODA la muestra (in-sample). Su volatilidad condicional
sirve para describir la serie, nunca como feature de un modelo predictivo o de un backtest.
Para eso usar `rolling_gjr_garch_forecast`.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def _fit(r_pct: np.ndarray, dist: str):
    from arch import arch_model

    model = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=dist)
    return model.fit(disp="off", show_warning=False)


def fit_gjr_garch(
    returns: pd.Series,
    dist: str = "t",
    annualize: bool = True,
    trading_periods: int = 252,
) -> Tuple[Dict[str, float], pd.Series]:
    """
    Ajusta un GJR-GARCH(1,1) sobre toda la serie de log-retornos (IN-SAMPLE, solo descriptivo).

    Retorna los parámetros (omega en escala decimal^2) y la volatilidad condicional sigma_t.
    """
    r = returns.dropna()
    res = _fit(r.values * 100.0, dist)  # escala % para estabilidad numérica

    p = res.params
    alpha = float(p.get("alpha[1]", 0.0))
    gamma = float(p.get("gamma[1]", 0.0))
    beta = float(p.get("beta[1]", 0.0))
    params = {
        "omega": float(p.get("omega", 0.0)) / 1e4,
        "alpha": alpha,
        "gamma_leverage": gamma,
        "beta": beta,
        "persistence": alpha + beta + 0.5 * gamma,
        "aic": float(res.aic),
        "bic": float(res.bic),
    }

    sigma = pd.Series(np.asarray(res.conditional_volatility) / 100.0, index=r.index)
    if annualize:
        sigma = sigma * np.sqrt(trading_periods)
    sigma.name = f"{returns.name}_gjr_garch_vol_insample"
    return params, sigma


def rolling_gjr_garch_forecast(
    returns: pd.Series,
    min_obs: int = 252,
    refit_step: int = 21,
    dist: str = "t",
    trading_periods: int = 252,
) -> pd.Series:
    """
    Pronóstico ex-ante a 1 paso de la volatilidad (anualizada), estrictamente causal.

    - Cada `refit_step` ruedas se reestiman los parámetros con ventana expansiva (datos hasta t-1).
    - Entre reajustes los parámetros quedan fijos, pero la varianza se ACTUALIZA todos los días con
      la recursión GJR usando el retorno observado ayer. (La versión anterior pronosticaba siempre
      desde la fecha del último reajuste, dejando el pronóstico congelado durante 21 días.)

    El valor en la fecha t usa solo retornos hasta t-1.
    """
    r = returns.dropna()
    x = r.values * 100.0
    n = len(x)
    out = np.full(n, np.nan)

    for start in range(min_obs, n, refit_step):
        res = _fit(x[:start], dist)
        p = res.params
        mu, omega = p["mu"], p["omega"]
        alpha, gamma, beta = p["alpha[1]"], p["gamma[1]"], p["beta[1]"]
        # varianza condicional de la rueda `start` dada la info hasta start-1
        s2 = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])
        for t in range(start, min(start + refit_step, n)):
            out[t] = s2
            e = x[t] - mu
            s2 = omega + (alpha + gamma * (e < 0)) * e * e + beta * s2

    vol = np.sqrt(out) / 100.0 * np.sqrt(trading_periods)
    return pd.Series(vol, index=r.index, name=f"{returns.name}_gjr_forecast")
