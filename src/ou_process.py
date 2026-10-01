"""
Módulo de Modelado Ornstein-Uhlenbeck (Clase 7)
==============================================
Calibra el proceso estocástico continuo de reversión a la media:
    dS_t = kappa * (theta - S_t) * dt + sigma_ou * dW_t

Permite calcular:
- theta: Media de equilibrio de largo plazo
- kappa: Velocidad de reversión a la media
- half-life: t_{1/2} = ln(2) / kappa (días para disipar la mitad de un shock)
- s-score: Puntuación estandarizada para tipificar regímenes de volatilidad
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def calibrate_ou_ar1(
    series: pd.Series,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Calibra analíticamente los parámetros del proceso Ornstein-Uhlenbeck
    ajustando una regresión AR(1): S_t = a + b * S_{t-1} + eps_t.
    """
    s = series.dropna()
    if len(s) < 10:
        raise ValueError("Serie temporal demasiado corta para calibrar OU.")

    s_curr = s.iloc[1:].values
    s_prev = s.iloc[:-1].values

    # OLS cerrado
    x = np.column_stack([np.ones_like(s_prev), s_prev])
    beta_ols, residuals, _, _ = np.linalg.lstsq(x, s_curr, rcond=None)
    a, b = beta_ols[0], beta_ols[1]

    # Prevenir no-estacionariedad o divergencia (b debe ser < 1)
    b_clipped = np.clip(b, 1e-4, 0.9999)

    kappa = -np.log(b_clipped) / dt
    theta = a / (1.0 - b_clipped)
    residuals_arr = s_curr - (a + b_clipped * s_prev)
    sigma_eps = float(np.std(residuals_arr, ddof=2))

    # Varianza y desvío de equilibrio estacionario
    sigma_eq = np.sqrt(sigma_eps ** 2 / (1.0 - b_clipped ** 2))
    half_life = np.log(2.0) / kappa

    return {
        "a": float(a),
        "b": float(b),
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma_eps": float(sigma_eps),
        "sigma_eq": float(sigma_eq),
        "half_life": float(half_life),
    }


def compute_rolling_ou_scores(
    series: pd.Series,
    window: int = 60,
    dt: float = 1.0,
) -> pd.DataFrame:
    """
    Calcula de forma rodante (rolling) los parámetros de Ornstein-Uhlenbeck y el s-score
    sin lookahead bias (utilizando solo la información disponible hasta t-1 para calibrar).
    """
    s = series.dropna()
    n = len(s)
    
    thetas = np.full(n, np.nan)
    kappas = np.full(n, np.nan)
    half_lives = np.full(n, np.nan)
    sigma_eqs = np.full(n, np.nan)
    s_scores = np.full(n, np.nan)

    values = s.values

    for t in range(window, n):
        window_slice = s.iloc[t - window:t]
        try:
            params = calibrate_ou_ar1(window_slice, dt=dt)
            thetas[t] = params["theta"]
            kappas[t] = params["kappa"]
            half_lives[t] = params["half_life"]
            sigma_eqs[t] = params["sigma_eq"]

            # s-score del día actual comparado contra los parámetros calibrados hasta ayer
            current_val = values[t]
            if params["sigma_eq"] > 1e-8:
                s_scores[t] = (current_val - params["theta"]) / params["sigma_eq"]
        except Exception:
            continue

    df_ou = pd.DataFrame(
        {
            "spread_level": s,
            "ou_theta": thetas,
            "ou_kappa": kappas,
            "ou_half_life": half_lives,
            "ou_sigma_eq": sigma_eqs,
            "ou_s_score": s_scores,
        },
        index=s.index,
    )

    # Clasificación categórica del régimen según s-score
    # s < -1: Contango Fuerte / Muy Tranquilo
    # -1 <= s < 0.5: Calma Normal
    # 0.5 <= s < 2.0: Alerta / Backwardation Leve
    # s >= 2.0: Shock Extremo / Backwardation Agudo
    df_ou["regime_label"] = pd.cut(
        df_ou["ou_s_score"],
        bins=[-np.inf, -1.0, 0.5, 2.0, np.inf],
        labels=["deep_contango", "normal_calm", "alert_backwardation", "extreme_shock"]
    )

    return df_ou
