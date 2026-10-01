"""
Módulo del Filtro de Kalman Dinámico (Clase 7)
==============================================
Implementa el estimador recursivo en espacio de estados para rastrear
coeficientes variables en el tiempo sin lookahead bias ni rezago de fase:
1. Kalman 1D: Coeficiente beta o hedge ratio dinámico (idéntico a Clase 7, Ej. 3)
2. Kalman 2D: Intercepto alfa y beta dinámicos (y_t = alpha_t + beta_t * x_t + eps_t)
"""

from typing import Tuple
import numpy as np
import pandas as pd


def kalman_beta_1d(
    y: pd.Series,
    x: pd.Series,
    q: float = 1e-5,
    r: float = None,
    initial_beta: float = 1.0,
    initial_p: float = 1.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Filtro de Kalman 1D: Estima beta_t en y_t = beta_t * x_t + eps_t.
    Idéntico a la formulación de la Clase 7 (~15 líneas).

    Parámetros:
    -----------
    y : pd.Series
        Serie dependiente (ej: ln(VIX) o retornos del activo).
    x : pd.Series
        Serie independiente (ej: ln(VIX3M) o retornos de mercado).
    q : float
        Varianza del proceso de estado (cuánto puede moverse beta por día).
    r : float
        Varianza del ruido de medición. Si es None, usa 0.5 * var(y).

    Retorna:
    --------
    beta_series : pd.Series
        Serie temporal de beta_t estimada recursivamente día a día.
    spread_series : pd.Series
        Serie de residuos: e_t = y_t - beta_{t-1} * x_t (sin lookahead).
    """
    valid = (~y.isna()) & (~x.isna())
    y_aligned = y.loc[valid]
    x_aligned = x.loc[valid]
    common_idx = y_aligned.index

    if r is None:
        r = float(y_aligned.var() * 0.5)

    n = len(y_aligned)
    beta_k = np.zeros(n)
    b = initial_beta
    p = initial_p

    y_vals = y_aligned.values
    x_vals = x_aligned.values

    for t in range(n):
        x_t = x_vals[t]
        y_t = y_vals[t]

        # 1. PREDICT: el estado pudo moverse
        p = p + q

        # 2. GAIN: dial de confianza
        denom = (x_t ** 2) * p + r
        k = (p * x_t) / denom if denom != 0 else 0.0

        # 3. UPDATE: corregir por la sorpresa observada
        b = b + k * (y_t - b * x_t)
        p = (1.0 - k * x_t) * p

        beta_k[t] = b

    beta_series = pd.Series(beta_k, index=common_idx, name="beta_kalman").reindex(y.index).ffill()
    # Residuo con beta rezagado (shift 1) para garantizar CERO lookahead bias
    spread_series = y - beta_series.shift(1) * x
    spread_series.name = "spread_kalman"

    return beta_series, spread_series


def kalman_capm_2d(
    y: pd.Series,
    x: pd.Series,
    q_alpha: float = 1e-6,
    q_beta: float = 1e-5,
    r: float = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Filtro de Kalman 2D: Estima alpha_t y beta_t en:
    y_t = alpha_t + beta_t * x_t + eps_t

    Retorna:
    --------
    states_df : pd.DataFrame
        Columnas ['alpha_kalman', 'beta_kalman']
    resid_series : pd.Series
        Residuo sin lookahead: y_t - (alpha_{t-1} + beta_{t-1} * x_t)
    """
    valid = (~y.isna()) & (~x.isna())
    y_aligned = y.loc[valid]
    x_aligned = x.loc[valid]
    common_idx = y_aligned.index

    if r is None:
        r = float(y_aligned.var() * 0.5)

    n = len(y_aligned)
    theta = np.zeros(2)  # [alpha, beta]
    p_mat = np.eye(2) * 1.0
    q_mat = np.diag([q_alpha, q_beta])

    states = np.zeros((n, 2))
    y_vals = y_aligned.values
    x_vals = x_aligned.values

    for t in range(n):
        h_t = np.array([1.0, x_vals[t]])
        y_t = y_vals[t]

        # PREDICT
        p_mat = p_mat + q_mat

        # GAIN
        s = h_t @ p_mat @ h_t + r
        k_gain = (p_mat @ h_t) / s

        # UPDATE
        surprise = y_t - (h_t @ theta)
        theta = theta + k_gain * surprise
        p_mat = (np.eye(2) - np.outer(k_gain, h_t)) @ p_mat

        states[t] = theta

    states_df = pd.DataFrame(
        states,
        index=common_idx,
        columns=["alpha_kalman", "beta_kalman"]
    ).reindex(y.index).ffill()
    # Residuo rezagado (t-1)
    alpha_lag = states_df["alpha_kalman"].shift(1)
    beta_lag = states_df["beta_kalman"].shift(1)
    resid_series = y - (alpha_lag + beta_lag * x)
    resid_series.name = "idiosyncratic_resid"

    return states_df, resid_series
