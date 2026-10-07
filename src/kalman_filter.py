"""
Módulo del Filtro de Kalman Dinámico (Clase 7)
==============================================
Regresión con coeficientes variables en el tiempo (estado = coeficientes, paseo aleatorio):
    y_t = H_t' theta_t + eps_t,      eps_t ~ N(0, r)
    theta_t = theta_{t-1} + w_t,     w_t ~ N(0, diag(q))

1. Kalman 1D: y_t = beta_t * x_t + eps_t (ratio dinámico, sin intercepto)
2. Kalman 2D: y_t = alpha_t + beta_t * x_t + eps_t (CAPM dinámico)

Las varianzas (q, r) se estiman por máxima verosimilitud (descomposición del error de predicción)
usando SOLO una ventana inicial de calibración (`fit_window`); el filtro corre luego hacia adelante.
Así el grado de suavizado lo decide la verosimilitud y no se usa información futura
(la versión anterior fijaba q a mano y tomaba r = 0.5 * var(y) de TODA la muestra).
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _filter(y: np.ndarray, H: np.ndarray, q: np.ndarray, r: float,
            theta0: np.ndarray, p0: float) -> Tuple[np.ndarray, float]:
    """Filtro de Kalman con estado en paseo aleatorio. Devuelve estados filtrados y log-verosimilitud."""
    n, k = H.shape
    theta = theta0.astype(float).copy()
    P = np.eye(k) * p0
    Q = np.diag(q)
    states = np.empty((n, k))
    ll = 0.0
    for t in range(n):
        h = H[t]
        P = P + Q                         # PREDICT
        F = h @ P @ h + r                 # varianza de la innovación
        v = y[t] - h @ theta              # sorpresa
        ll -= 0.5 * (np.log(2 * np.pi * F) + v * v / F)
        K = P @ h / F                     # GAIN
        theta = theta + K * v             # UPDATE
        P = P - np.outer(K, h) @ P
        states[t] = theta
    return states, ll


def _mle(y: np.ndarray, H: np.ndarray, theta0: np.ndarray, p0: float) -> Tuple[np.ndarray, float]:
    """MLE de (log q_1..q_k, log r) sobre la ventana de calibración."""
    k = H.shape[1]
    r0 = max(np.var(y), 1e-12)
    x0 = np.r_[np.full(k, np.log(r0 * 1e-3)), np.log(r0)]

    def nll(th):
        _, ll = _filter(y, H, np.exp(th[:k]), float(np.exp(th[k])), theta0, p0)
        return -ll if np.isfinite(ll) else 1e12

    res = minimize(nll, x0, method="L-BFGS-B", bounds=[(-30, 5)] * (k + 1))
    return np.exp(res.x[:k]), float(np.exp(res.x[k]))


def _align(y: pd.Series, x: pd.Series) -> Tuple[pd.Series, pd.Series]:
    idx = y.dropna().index.intersection(x.dropna().index)
    return y.loc[idx], x.loc[idx]


def kalman_beta_1d(
    y: pd.Series,
    x: pd.Series,
    q: Optional[float] = None,
    r: Optional[float] = None,
    initial_beta: float = 1.0,
    initial_p: float = 1.0,
    fit_window: int = 252,
) -> Tuple[pd.Series, pd.Series]:
    """
    Estima beta_t en y_t = beta_t * x_t + eps_t.

    Si q o r son None se estiman por MLE con las primeras `fit_window` observaciones.
    Devuelve (beta_t, spread_t) con spread_t = y_t - beta_{t-1} * x_t (sin lookahead).

    Nota: es un ratio dinámico sin intercepto, NO un test de cointegración.
    """
    y_a, x_a = _align(y, x)
    H = x_a.values.reshape(-1, 1)
    theta0 = np.array([initial_beta])
    if q is None or r is None:
        w = min(fit_window, len(y_a))
        q_hat, r_hat = _mle(y_a.values[:w], H[:w], theta0, initial_p)
        q = float(q_hat[0]) if q is None else q
        r = r_hat if r is None else r
    states, _ = _filter(y_a.values, H, np.array([q]), r, theta0, initial_p)

    beta = pd.Series(states[:, 0], index=y_a.index, name="beta_kalman")
    spread = (y_a - beta.shift(1) * x_a).rename("spread_kalman")
    beta.attrs.update(q=q, r=r)
    return beta, spread


def kalman_capm_2d(
    y: pd.Series,
    x: pd.Series,
    q_alpha: Optional[float] = None,
    q_beta: Optional[float] = None,
    r: Optional[float] = None,
    fit_window: int = 252,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Estima alpha_t y beta_t en y_t = alpha_t + beta_t * x_t + eps_t (CAPM dinámico).

    Si algún parámetro es None, (q_alpha, q_beta, r) se estiman por MLE con las primeras `fit_window` obs.
    Devuelve los estados y el residuo sin lookahead: y_t - (alpha_{t-1} + beta_{t-1} x_t).
    """
    y_a, x_a = _align(y, x)
    H = np.column_stack([np.ones(len(x_a)), x_a.values])
    theta0 = np.array([0.0, 1.0])
    p0 = 1.0
    if q_alpha is None or q_beta is None or r is None:
        w = min(fit_window, len(y_a))
        q_hat, r_hat = _mle(y_a.values[:w], H[:w], theta0, p0)
        q_alpha = q_hat[0] if q_alpha is None else q_alpha
        q_beta = q_hat[1] if q_beta is None else q_beta
        r = r_hat if r is None else r
    states, _ = _filter(y_a.values, H, np.array([q_alpha, q_beta]), r, theta0, p0)

    states_df = pd.DataFrame(states, index=y_a.index, columns=["alpha_kalman", "beta_kalman"])
    resid = y_a - (states_df["alpha_kalman"].shift(1) + states_df["beta_kalman"].shift(1) * x_a)
    resid.name = "idiosyncratic_resid"
    states_df.attrs.update(q_alpha=float(q_alpha), q_beta=float(q_beta), r=float(r))
    return states_df, resid
