"""
Módulo de Diferenciación Fraccionaria (Clase 6 - ML Financiero)
===============================================================
Basado en Marcos López de Prado (Advances in Financial Machine Learning, Cap. 5).
Halla el orden mínimo de diferenciación d* in (0, 1) que logra estacionariedad
(test ADF p-value < 0.05) preservando la máxima memoria histórica posible.
"""

from typing import Dict, Tuple
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


def get_fractional_weights(d: float, size: int, threshold: float = 1e-4) -> np.ndarray:
    """
    Genera los pesos binomiales w_k para el operador de diferenciación fraccionaria:
    (1 - B)^d = sum_{k=0}^infty w_k B^k
    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1.0) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w[::-1])  # Invertir para convolución con rezagos


def fractional_diff(
    series: pd.Series,
    d: float,
    threshold: float = 1e-4,
) -> pd.Series:
    """
    Aplica diferenciación fraccionaria de orden d con memoria acotada por umbral.
    """
    s = series.dropna()
    n = len(s)
    weights = get_fractional_weights(d, size=n, threshold=threshold)
    k_len = len(weights)

    if k_len > n:
        return pd.Series(index=s.index, dtype=float)

    values = s.values
    diff_vals = np.full(n, np.nan)

    # Convolución causal sin lookahead
    for i in range(k_len - 1, n):
        window = values[i - k_len + 1 : i + 1]
        diff_vals[i] = np.dot(weights, window)

    res = pd.Series(diff_vals, index=s.index, name=f"{series.name}_fracdiff_d{d:.2f}")
    return res


def find_optimal_d(
    series: pd.Series,
    d_values: np.ndarray = None,
    threshold: float = 1e-4,
    adf_significance: float = 0.05,
) -> Tuple[float, pd.DataFrame, pd.Series]:
    """
    Busca el orden d* mínimo óptimo que hace a la serie estacionaria
    según el test Augmented Dickey-Fuller (ADF).

    Retorna:
    --------
    best_d : float
        Mínimo orden d que logra p-value < adf_significance.
    summary_df : pd.DataFrame
        Tabla con columnas ['d', 'adf_stat', 'p_value', 'correlation']
    best_series : pd.Series
        Serie fraccionariamente diferenciada con d*.
    """
    # Import diferido de statsmodels para evitar dependencias circulares
    from statsmodels.tsa.stattools import adfuller

    if d_values is None:
        d_values = np.linspace(0.0, 1.0, 21)

    s = series.dropna()
    results = []
    best_d = 1.0
    found_stationary = False

    for d in d_values:
        if d == 0.0:
            diff_s = s
        else:
            diff_s = fractional_diff(s, d=d, threshold=threshold).dropna()

        if len(diff_s) < 30:
            continue

        # Test ADF
        adf_res = adfuller(diff_s.values, autolag="AIC")
        adf_stat = float(adf_res[0])
        p_val = float(adf_res[1])

        # Correlación con la serie en nivel original
        common_idx = s.index.intersection(diff_s.index)
        corr = float(np.corrcoef(s.loc[common_idx], diff_s.loc[common_idx])[0, 1])

        results.append({
            "d": round(float(d), 2),
            "adf_stat": adf_stat,
            "p_value": p_val,
            "correlation": corr,
            "is_stationary": p_val < adf_significance,
        })

        if not found_stationary and (p_val < adf_significance):
            best_d = float(d)
            found_stationary = True

    df_results = pd.DataFrame(results)
    best_series = fractional_diff(s, d=best_d, threshold=threshold)
    best_series.name = f"{series.name}_fracdiff_opt"

    return best_d, df_results, best_series
