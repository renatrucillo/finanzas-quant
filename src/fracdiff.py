"""
Módulo de Diferenciación Fraccionaria (Clase 6 - ML Financiero)
===============================================================
López de Prado (Advances in Financial Machine Learning, cap. 5), método de ventana fija (FFD).

Se aplica sobre el LOGARITMO del precio. El orden d* es el mínimo d a partir del cual ADF rechaza la
raíz unitaria para ESE d y para todos los mayores de la grilla (el p-valor de ADF no es monótono en d,
y tomar el primer rechazo aislado es frágil). Se usa SOLO un período de entrenamiento (`train_end`):
elegir d con toda la muestra usa información futura. KPSS se reporta como diagnóstico.
"""

from typing import Optional, Tuple
import warnings
import numpy as np
import pandas as pd


def get_fractional_weights(d: float, size: int, threshold: float = 1e-4) -> np.ndarray:
    """Pesos de (1 - B)^d: w_0 = 1, w_k = -w_{k-1} (d - k + 1) / k, truncados cuando |w_k| < threshold."""
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1.0) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w[::-1])  # invertido para el producto con la ventana [t-L+1, ..., t]


def fractional_diff(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """Diferenciación fraccionaria FFD, causal (cada valor usa solo observaciones pasadas)."""
    s = series.dropna()
    n = len(s)
    w = get_fractional_weights(d, size=n, threshold=threshold)
    L = len(w)
    out = np.full(n, np.nan)
    if L <= n:
        v = s.values
        for i in range(L - 1, n):
            out[i] = np.dot(w, v[i - L + 1:i + 1])
    return pd.Series(out, index=s.index, name=f"{series.name}_fracdiff_d{d:.2f}")


def find_optimal_d(
    series: pd.Series,
    d_values: Optional[np.ndarray] = None,
    threshold: float = 1e-4,
    significance: float = 0.05,
    train_end=None,
) -> Tuple[float, pd.DataFrame, pd.Series]:
    """
    Busca el mínimo d tal que, en el período de entrenamiento, ADF rechaza raíz unitaria para d y para
    todo d mayor de la grilla. Devuelve (d*, tabla de diagnóstico con ADF y KPSS, serie diferenciada con d*).

    `series` debería ser el log-precio. `train_end` (fecha o None = toda la muestra) limita los tests.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    if d_values is None:
        d_values = np.linspace(0.0, 1.0, 21)
    s = series.dropna()
    s_train = s.loc[:train_end] if train_end is not None else s

    rows = []
    for d in d_values:
        x = s_train if d == 0 else fractional_diff(s_train, d, threshold).dropna()
        if len(x) < 100:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # avisos de API de statsmodels y de p-valores fuera de tabla en KPSS
            adf_p = float(adfuller(x.values, autolag="AIC")[1])
            kpss_p = float(kpss(x.values, regression="c", nlags="auto")[1])
        corr = float(np.corrcoef(s_train.loc[x.index], x)[0, 1])
        rows.append(dict(d=round(float(d), 2), adf_p=adf_p, kpss_p=kpss_p, correlation=corr, n=len(x),
                         adf_rechaza=adf_p < significance, kpss_no_rechaza=kpss_p > significance))

    table = pd.DataFrame(rows)
    best_d = 1.0
    for i in range(len(table)):
        if table["adf_rechaza"].iloc[i:].all():
            best_d = float(table["d"].iloc[i])
            break
    best = fractional_diff(s, best_d, threshold)
    best.name = f"{series.name}_fracdiff_opt"
    return best_d, table, best
