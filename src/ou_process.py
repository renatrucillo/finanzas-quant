"""
Módulo de Modelado Ornstein-Uhlenbeck (Clase 7)
==============================================
Calibra el proceso de reversión a la media:
    dS_t = kappa * (theta - S_t) * dt + sigma_ou * dW_t

Discretización exacta: S_t = a + b S_{t-1} + eps_t, con b = e^{-kappa dt}, theta = a / (1 - b).
- half-life: t_{1/2} = ln(2) / kappa
- s-score: (S_t - theta) / sigma_eq, sigma_eq = sigma_eps / sqrt(1 - b^2)

El estimador MCO de b está sesgado hacia abajo en muestras cortas (sesgo de Kendall ~ -(1+3b)/n),
lo que subestima el half-life. Por defecto se corrige: b_adj = b + (1 + 3b) / n.
"""

from typing import Dict
import numpy as np
import pandas as pd


def calibrate_ou_ar1(series: pd.Series, dt: float = 1.0, bias_correct: bool = True) -> Dict[str, float]:
    """Calibra OU vía AR(1) por MCO (con corrección de sesgo de Kendall opcional)."""
    s = series.dropna()
    if len(s) < 10:
        raise ValueError("Serie temporal demasiado corta para calibrar OU.")

    s_curr = s.iloc[1:].values
    s_prev = s.iloc[:-1].values
    n = len(s_curr)

    X = np.column_stack([np.ones_like(s_prev), s_prev])
    (a, b), *_ = np.linalg.lstsq(X, s_curr, rcond=None)
    b_raw = b
    if bias_correct:
        b = b + (1.0 + 3.0 * b) / n
    b = float(np.clip(b, 1e-4, 0.9999))  # estacionariedad
    # el intercepto se recalcula para conservar la media muestral con el b corregido
    a = float(np.mean(s_curr) - b * np.mean(s_prev))

    kappa = -np.log(b) / dt
    theta = a / (1.0 - b)
    resid = s_curr - (a + b * s_prev)
    sigma_eps = float(np.std(resid, ddof=2))
    sigma_eq = sigma_eps / np.sqrt(1.0 - b**2)

    return {
        "a": a,
        "b": b,
        "b_mco": float(b_raw),
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma_eps": sigma_eps,
        "sigma_eq": float(sigma_eq),
        "half_life": float(np.log(2.0) / kappa),
    }


def compute_rolling_ou_scores(
    series: pd.Series,
    window: int = 252,
    dt: float = 1.0,
    bias_correct: bool = True,
) -> pd.DataFrame:
    """
    Parámetros OU rodantes y s-score sin lookahead: en t se calibra con [t-window, t-1] y se evalúa S_t.

    Regímenes (según la especificación del TP):
        s < 0      -> calma (contango normal)
        0 <= s < 2 -> alerta
        s >= 2     -> estrés extremo (backwardation aguda)
    """
    s = series.dropna()
    n = len(s)
    cols = {k: np.full(n, np.nan) for k in ["ou_theta", "ou_kappa", "ou_half_life", "ou_sigma_eq", "ou_s_score"]}
    values = s.values

    for t in range(window, n):
        try:
            p = calibrate_ou_ar1(s.iloc[t - window:t], dt=dt, bias_correct=bias_correct)
        except ValueError:
            continue
        cols["ou_theta"][t] = p["theta"]
        cols["ou_kappa"][t] = p["kappa"]
        cols["ou_half_life"][t] = p["half_life"]
        cols["ou_sigma_eq"][t] = p["sigma_eq"]
        if p["sigma_eq"] > 1e-8:
            cols["ou_s_score"][t] = (values[t] - p["theta"]) / p["sigma_eq"]

    df = pd.DataFrame({"spread_level": s, **cols}, index=s.index)
    df["regime_label"] = pd.cut(df["ou_s_score"], bins=[-np.inf, 0.0, 2.0, np.inf], right=False,
                                labels=["calma", "alerta", "estres_extremo"])
    return df
