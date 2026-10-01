"""
Módulo de Volatilidad Condicional Asimétrica GJR-GARCH(1,1) (Clase 4)
=====================================================================
Implementa el modelo GJR-GARCH(1,1) (Glosten, Jagannathan y Runkle, 1993):
    sigma_t^2 = omega + (alpha + gamma * I_{eps_{t-1} < 0}) * eps_{t-1}^2 + beta * sigma_{t-1}^2

Captura:
1. Volatility clustering (persistencia: alpha + beta + gamma/2 < 1)
2. Leverage effect (asimetría ante shocks negativos: gamma > 0)
3. Pronósticos condicionales 1-step ahead sin lookahead bias
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


def fit_gjr_garch(
    returns: pd.Series,
    dist: str = "StudentsT",
    annualize: bool = True,
    trading_periods: int = 252,
) -> Tuple[Dict[str, float], pd.Series]:
    """
    Ajusta un modelo GJR-GARCH(1,1) sobre una serie de log-retornos.

    Parámetros:
    -----------
    returns : pd.Series
        Log-retornos diarios (en escala decimal, ej: 0.01 = 1%).
    dist : str
        Distribución de los residuos: 'StudentsT', 'normal' o 'skewt'.
    annualize : bool
        Si True, devuelve la volatilidad condicional anualizada (* sqrt(252)).

    Retorna:
    --------
    params : Dict[str, float]
        Parámetros estimados (omega, alpha, gamma, beta, persistencia).
    cond_vol : pd.Series
        Serie de volatilidad condicional sigma_t.
    """
    from arch import arch_model

    r = returns.dropna() * 100.0  # Escalar a porcentaje para estabilidad numérica de optimización

    # p=1, o=1 (asimetría GJR), q=1
    model = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=dist)
    res = model.fit(disp="off", show_warning=False)

    p = res.params
    omega = float(p.get("omega", 0.0)) / 10000.0
    alpha = float(p.get("alpha[1]", 0.0))
    gamma = float(p.get("gamma[1]", 0.0))
    beta = float(p.get("beta[1]", 0.0))
    persistence = alpha + beta + 0.5 * gamma

    params = {
        "omega": omega,
        "alpha": alpha,
        "gamma_leverage": gamma,
        "beta": beta,
        "persistence": persistence,
        "aic": float(res.aic),
        "bic": float(res.bic),
    }

    # Volatilidad condicional estimada in-sample
    sigma_daily = (res.conditional_volatility / 100.0).copy()
    if annualize:
        sigma_cond = sigma_daily * np.sqrt(trading_periods)
    else:
        sigma_cond = sigma_daily

    sigma_cond.name = f"{returns.name}_gjr_garch_vol"
    return params, sigma_cond


def rolling_gjr_garch_forecast(
    returns: pd.Series,
    min_obs: int = 252,
    refit_step: int = 21,
    dist: str = "StudentsT",
    trading_periods: int = 252,
) -> pd.Series:
    """
    Genera pronósticos de volatilidad condicional ex-ante 1-paso hacia adelante (t+1)
    con reajuste periódico (walk-forward) estrictamente causal y sin lookahead bias.
    """
    from arch import arch_model

    r_clean = returns.dropna()
    r_pct = r_clean * 100.0
    n = len(r_pct)

    forecasted_vol = pd.Series(index=r_clean.index, dtype=float, name=f"{returns.name}_gjr_forecast")

    # Ajuste inicial y pronóstico recursivo
    res = None
    for t in range(min_obs, n):
        # Reestimar parámetros cada 'refit_step' ruedas (ej. mensualmente)
        if (t == min_obs) or ((t - min_obs) % refit_step == 0):
            train_data = r_pct.iloc[:t]
            model = arch_model(train_data, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=dist)
            res = model.fit(disp="off", show_warning=False)

        # Proyectar varianza condicional para t (usando información hasta t-1)
        # forecast horizon=1
        f = res.forecast(horizon=1, start=train_data.index[-1], reindex=False)
        var_next = f.variance.values[-1, 0]
        sigma_next_daily = np.sqrt(var_next) / 100.0
        forecasted_vol.iloc[t] = sigma_next_daily * np.sqrt(trading_periods)

    return forecasted_vol
