"""
Módulo de Estimadores de Volatilidad Realizada (Mundo P)
=======================================================
Implementa estimadores de rango OHLC y Close-to-Close anualizados:
- Close-to-Close tradicional
- Parkinson (1980): basado en High-Low
- Garman-Klass (1980): basado en Open, High, Low, Close
- Yang-Zhang (2000): robusto a drift y saltos overnight
"""

from typing import Union
import numpy as np
import pandas as pd


def close_to_close_volatility(
    close: Union[pd.Series, pd.DataFrame],
    window: int = 21,
    trading_periods: int = 252,
    clean_nans: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calcula la volatilidad histórica tradicional Close-to-Close móvil.
    sigma = std(ln(C_t / C_{t-1})) * sqrt(trading_periods)
    """
    log_ret = np.log(close / close.shift(1))
    rolling_std = log_ret.rolling(window=window).std()
    ann_vol = rolling_std * np.sqrt(trading_periods)
    return ann_vol.dropna() if clean_nans else ann_vol


def parkinson_volatility(
    high: Union[pd.Series, pd.DataFrame],
    low: Union[pd.Series, pd.DataFrame],
    window: int = 21,
    trading_periods: int = 252,
    clean_nans: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Estimador de Parkinson (1980).
    Aproximadamente 5x más eficiente que Close-to-Close.
    sigma^2 = (1 / (4 * ln(2))) * ln(H / L)^2
    """
    hl_ratio = np.log(high / low)
    factor = 1.0 / (4.0 * np.log(2.0))
    daily_var = factor * (hl_ratio ** 2)
    rolling_var = daily_var.rolling(window=window).mean()
    ann_vol = np.sqrt(rolling_var * trading_periods)
    return ann_vol.dropna() if clean_nans else ann_vol


def garman_klass_volatility(
    open_: Union[pd.Series, pd.DataFrame],
    high: Union[pd.Series, pd.DataFrame],
    low: Union[pd.Series, pd.DataFrame],
    close: Union[pd.Series, pd.DataFrame],
    window: int = 21,
    trading_periods: int = 252,
    clean_nans: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Estimador de Garman-Klass (1980).
    Hasta 8x más eficiente que Close-to-Close.
    sigma^2 = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    term1 = 0.5 * (log_hl ** 2)
    term2 = (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    daily_var = term1 - term2

    daily_var = np.maximum(daily_var, 0.0)
    rolling_var = pd.DataFrame(daily_var).rolling(window=window).mean()
    ann_vol = np.sqrt(rolling_var * trading_periods)

    if isinstance(close, pd.Series):
        ann_vol = ann_vol.iloc[:, 0]

    return ann_vol.dropna() if clean_nans else ann_vol


def yang_zhang_volatility(
    open_: Union[pd.Series, pd.DataFrame],
    high: Union[pd.Series, pd.DataFrame],
    low: Union[pd.Series, pd.DataFrame],
    close: Union[pd.Series, pd.DataFrame],
    window: int = 21,
    trading_periods: int = 252,
    clean_nans: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Estimador de Yang-Zhang (2000).
    Independiente del drift y con ajuste por saltos overnight (Close_prev -> Open).
    """
    close_prev = close.shift(1)
    log_oc = np.log(open_ / close_prev)

    log_co = np.log(close / open_)
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)

    rs_daily = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    var_overnight = log_oc.rolling(window=window).var()
    var_open_to_close = log_co.rolling(window=window).var()
    var_rs = rs_daily.rolling(window=window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    total_var = var_overnight + k * var_open_to_close + (1.0 - k) * var_rs
    total_var = np.maximum(total_var, 0.0)
    ann_vol = np.sqrt(total_var * trading_periods)

    return ann_vol.dropna() if clean_nans else ann_vol
