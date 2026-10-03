"""Pronosticadores de volatilidad a horizonte de ciclo (≈21 ruedas), estrictamente causales.

En cada fecha de decisión t0 (cierre del vencimiento anterior) se usa SOLO información hasta
la rueda t0-1 (pos0-1) para pronosticar la vol anualizada promedio de las próximas h ruedas.

Modelos: rolling 21d, EWMA(0.94), GJR-GARCH(1,1)-t, HAR-RV (Corsi), Kalman (filtro, nunca smoother)
y ensamble (promedio de varianzas). La IV (VIX/VXN) se evalúa como referencia, no como modelo propio.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ANN = 252
MODELS = ["Rolling21", "EWMA", "GJR-GARCH", "HAR-RV", "Kalman"]
MIN_OBS = 252


# ---------------------------------------------------------------- insumos
def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def gk_daily_var(px: pd.DataFrame) -> pd.Series:
    """Varianza diaria de Garman-Klass (1980) a partir de OHLC (sin gap overnight)."""
    o, h, l, c = px["Open"], px["High"], px["Low"], px["Close"]
    v = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return v.clip(lower=0.0)


# ---------------------------------------------------------------- modelos simples
def f_rolling(r: np.ndarray, h: int, win: int = 21) -> float:
    return float(np.sqrt(np.var(r[-win:], ddof=1) * ANN))


def f_ewma(r: np.ndarray, h: int, lam: float = 0.94) -> float:
    """RiskMetrics: sigma2_{t+1} = lam*sigma2_t + (1-lam)*r_t^2 (pronóstico plano en h)."""
    s2 = np.var(r[:30], ddof=1)
    for x in r:
        s2 = lam * s2 + (1 - lam) * x * x
    return float(np.sqrt(s2 * ANN))


# ---------------------------------------------------------------- GJR-GARCH
def f_gjr(r: np.ndarray, h: int, cache: dict) -> float:
    """GJR-GARCH(1,1)-t, ventana expansiva; pronóstico analítico a h pasos (promedio de varianza)."""
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(r * 100.0, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        try:
            res = am.fit(disp="off", show_warning=False, starting_values=cache.get("sv"))
        except Exception:
            res = am.fit(disp="off", show_warning=False)
        cache["sv"] = res.params.values
        cache["params"] = res.params.to_dict()
        fc = res.forecast(horizon=h, reindex=False).variance.values[-1]
    return float(np.sqrt(fc.mean() / 1e4 * ANN))


# ---------------------------------------------------------------- HAR-RV
def _har_features(rv: np.ndarray) -> np.ndarray:
    s = pd.Series(rv)
    return np.column_stack([np.ones(len(s)), s.values, s.rolling(5).mean().values, s.rolling(22).mean().values])


def f_har(r: np.ndarray, gk: np.ndarray, h: int, horizon_fit: int = 21) -> float:
    """HAR-RV (Corsi 2009): E[var close-to-close próximos 21d] ~ RV_d + RV_w + RV_m (RV = Garman-Klass).

    Se entrena con muestras cuyo target ya es observable en t0-1 (sin solapamiento con el futuro).
    """
    n = len(r)
    X = _har_features(gk)
    r2 = r**2
    # target: media de r^2 en (t+1 .. t+horizon_fit)
    cs = np.concatenate([[0.0], np.cumsum(r2)])
    tgt = np.full(n, np.nan)
    for t in range(n - horizon_fit):
        tgt[t] = (cs[t + 1 + horizon_fit] - cs[t + 1]) / horizon_fit
    ok = ~np.isnan(tgt) & ~np.isnan(X).any(axis=1)
    beta, *_ = np.linalg.lstsq(X[ok], tgt[ok], rcond=None)
    pred = float(X[-1] @ beta)
    floor = 0.25 * np.mean(r2[-252:])
    return float(np.sqrt(max(pred, floor) * ANN))


# ---------------------------------------------------------------- Kalman (espacio de estados)
def _kf(y: np.ndarray, mu: float, phi: float, q: float, rr: float, ret_all: bool = False):
    """Filtro de Kalman para y_t = x_t + e_t (var rr), x_t - mu = phi (x_{t-1} - mu) + w_t (var q).

    Devuelve log-verosimilitud y el estado filtrado (a_t|t, P_t|t): usa SOLO información hasta t.
    """
    n = len(y)
    a = mu
    P = q / max(1 - phi * phi, 1e-6)
    ll = 0.0
    af = np.empty(n)
    Pf = np.empty(n)
    for t in range(n):
        # predicción
        if t > 0:
            a = mu + phi * (a - mu)
            P = phi * phi * P + q
        v = y[t] - a
        F = P + rr
        ll -= 0.5 * (np.log(2 * np.pi * F) + v * v / F)
        K = P / F
        a = a + K * v
        P = (1 - K) * P
        af[t] = a
        Pf[t] = P
    return (ll, af, Pf) if ret_all else ll


def kalman_fit(y: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
    """MLE de (mu, phi, q, r). El grado de suavizado lo decide la verosimilitud (q/r), no se fija a mano."""
    if x0 is None:
        x0 = np.array([np.mean(y), 2.0, np.log(0.05), np.log(0.8)])  # phi = tanh(2) ≈ 0.96

    def unpack(th):
        return th[0], np.tanh(th[1]), np.exp(th[2]), np.exp(th[3])

    obj = lambda th: -_kf(y, *unpack(th))
    res = minimize(obj, x0, method="L-BFGS-B", options=dict(maxiter=80))
    return res.x


def f_kalman(r: np.ndarray, gk: np.ndarray, h: int, cache: dict, refit: bool) -> float:
    """Pronóstico a h pasos del filtro (no del smoother) sobre log-varianza GK.

    Calibración de nivel: K = mean(r^2)/mean(exp(estado filtrado)) en la ventana de entrenamiento
    (corrige el sesgo de Jensen/log-chi2 y la diferencia GK vs close-to-close).
    """
    y = np.log(np.maximum(gk, 1e-9))
    y = np.maximum(y, np.percentile(y, 1))  # winsorizo días de rango ~0
    if refit or "th" not in cache:
        cache["th"] = kalman_fit(y, cache.get("th"))
    th = cache["th"]
    mu, phi, q, rr = th[0], np.tanh(th[1]), np.exp(th[2]), np.exp(th[3])
    _, af, Pf = _kf(y, mu, phi, q, rr, ret_all=True)
    K = np.mean(r**2) / np.mean(np.exp(af))
    # varianza esperada del proceso latente: lognormal
    ks = np.arange(1, h + 1)
    path = mu + phi**ks * (af[-1] - mu)
    var_path = phi ** (2 * ks) * Pf[-1] + q * (1 - phi ** (2 * ks)) / (1 - phi**2)
    daily = np.exp(path + 0.5 * var_path) * K
    cache["last"] = dict(mu=mu, phi=phi, q=q, r=rr, q_over_r=q / rr)
    return float(np.sqrt(daily.mean() * ANN))


# ---------------------------------------------------------------- orquestación
def forecast_cycles(px: pd.DataFrame, iv: pd.Series, cycles: pd.DataFrame, kalman_every: int = 12,
                    min_obs: int = MIN_OBS, verbose: bool = True) -> pd.DataFrame:
    """Un pronóstico por modelo para cada ciclo, con datos hasta pos0-1. Devuelve DataFrame indexado por t0."""
    r_all = log_returns(px["Close"])
    gk_all = gk_daily_var(px)
    out = []
    gcache: dict = {}
    kcache: dict = {}
    for i, c in cycles.iterrows():
        end = c.pos0  # r_all.iloc[:end] -> retornos hasta la rueda pos0-1
        r = r_all.iloc[1:end].values
        gk = gk_all.iloc[1:end].values
        if len(r) < min_obs:
            continue
        h = int(c.h)
        row = dict(t0=c.t0, texp=c.texp, h=h)
        row["Rolling21"] = f_rolling(r, h)
        row["EWMA"] = f_ewma(r, h)
        row["GJR-GARCH"] = f_gjr(r, h, gcache)
        row["HAR-RV"] = f_har(r, gk, h)
        row["Kalman"] = f_kalman(r, gk, h, kcache, refit=(len(out) % kalman_every == 0))
        # realizada del ciclo (close-to-close, cero media) e IV observada en t0-1
        rr = r_all.iloc[c.pos0 + 1 : c.posT + 1].values
        row["IV_sig"] = float(iv.iloc[c.pos0 - 1])  # IV observable al decidir (rueda t0-1)
        row["IV_exec"] = float(iv.iloc[c.pos0])     # IV a la que se vende (cierre de t0)
        row["RV"] = float(np.sqrt(np.sum(rr**2) / h * ANN))
        out.append(row)
        if verbose and len(out) % 40 == 0:
            print(f"  ciclo {len(out)} / {len(cycles)}  ({c.t0.date()})", flush=True)
    df = pd.DataFrame(out).set_index("t0")
    df.attrs["garch_last"] = gcache.get("params")
    df.attrs["kalman_last"] = kcache.get("last")
    return add_ensemble(df)


def add_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Ensamble"] = np.sqrt((df[MODELS] ** 2).mean(axis=1))
    return df


# ---------------------------------------------------------------- métricas de pronóstico
def qlike(rv: np.ndarray, fc: np.ndarray) -> float:
    """QLIKE sobre varianzas (menor es mejor, robusta a ruido en el proxy)."""
    a, b = rv**2, fc**2
    return float(np.mean(a / b - np.log(a / b) - 1))


def mincer_zarnowitz(rv: np.ndarray, fc: np.ndarray) -> dict:
    X = np.column_stack([np.ones(len(fc)), fc])
    beta, *_ = np.linalg.lstsq(X, rv, rcond=None)
    res = rv - X @ beta
    r2 = 1 - res.var() / rv.var()
    return dict(a=beta[0], b=beta[1], r2=r2)


def forecast_table(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """RMSE (vol), QLIKE, sesgo medio y R² de Mincer-Zarnowitz por modelo + IV de referencia."""
    cols = cols or (MODELS + ["Ensamble"])
    d = df.copy()
    d["IV"] = d["IV_sig"]
    rows = []
    for m in cols + ["IV"]:
        f, y = d[m].values, d["RV"].values
        mz = mincer_zarnowitz(y, f)
        rows.append(dict(modelo=m, RMSE=float(np.sqrt(np.mean((f - y) ** 2))), QLIKE=qlike(y, f),
                         sesgo=float(np.mean(f - y)), R2_MZ=mz["r2"]))
    return pd.DataFrame(rows).set_index("modelo")
