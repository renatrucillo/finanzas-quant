"""
Módulo de Pricing y Griegas Black-Scholes (Fase 2 - Mundo Q)
=============================================================
Marco teórico: Clase 2 (No-Arbitraje y Black-Scholes), Clase 3 (Smile & Skew)
y Clase 4 (Modelado de VRP).

Funcionalidades:
1. Valuación analítica Black-Scholes-Merton (con tasa libre de riesgo y dividendos continuos).
2. Cálculo analítico vectorizado de Griegas de primer y segundo orden (Delta, Gamma, Vega, Theta, Rho).
3. Inversión analítica de Strike por Delta objetivo: K = f(S, r, q, sigma, T, Delta).
4. Estimación de Volatilidad Implícita para activos individuales y ETFs a partir del VIX y Betas CAPM.
5. Descomposición teórica del P&L de Covered Calls y cosecha de Variance Risk Premium (VRP).
"""

from typing import Union, Tuple, Dict
import numpy as np
import pandas as pd
from scipy.stats import norm


def black_scholes_call_price(
    S: Union[float, np.ndarray, pd.Series],
    K: Union[float, np.ndarray, pd.Series],
    T: Union[float, np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[float, np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0,
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calcula el precio analítico de una opción Call europea bajo Black-Scholes-Merton.

    Parámetros:
    -----------
    S : float o array
        Precio spot del subyacente.
    K : float o array
        Precio de ejercicio (strike).
    T : float o array
        Tiempo hasta el vencimiento en años (ej. 30/252 o 21/252).
    r : float o array
        Tasa libre de riesgo anualizada continua (ej. 0.04 para 4%).
    sigma : float o array
        Volatilidad anualizada (en decimal, ej. 0.20 para 20%).
    q : float o array, default=0.0
        Tasa de dividendos continua anualizada.

    Retorna:
    --------
    call_price : float o array
        Precio teórico de la opción Call.
    """
    S = np.maximum(np.asarray(S, dtype=float), 1e-6)
    K = np.maximum(np.asarray(K, dtype=float), 1e-6)
    T = np.maximum(np.asarray(T, dtype=float), 1e-6)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    call_price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    return np.maximum(call_price, 0.0)


def black_scholes_greeks(
    S: Union[float, np.ndarray, pd.Series],
    K: Union[float, np.ndarray, pd.Series],
    T: Union[float, np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[float, np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0,
) -> Dict[str, Union[float, np.ndarray, pd.Series]]:
    """
    Calcula el vector completo de Griegas de una Call europea analíticamente.

    Griegas calculadas:
    - Delta: dC / dS
    - Gamma: d^2C / dS^2
    - Theta: dC / dt (expresada por día calendario / 365 o día hábil / 252)
    - Vega: dC / dsigma (cambio de prima por variación de 1 punto porcentual de vol)
    - Rho: dC / dr
    """
    S = np.maximum(np.asarray(S, dtype=float), 1e-6)
    K = np.maximum(np.asarray(K, dtype=float), 1e-6)
    T = np.maximum(np.asarray(T, dtype=float), 1e-6)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    delta = disc_q * cdf_d1
    gamma = (disc_q * pdf_d1) / (S * sigma * sqrt_T)
    vega = S * disc_q * sqrt_T * pdf_d1  # Por unidad de vol (dividir por 100 para 1% de vol)
    
    # Theta anualizada (negativa por decaimiento temporal)
    theta_annual = (
        - (S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
        - r * K * disc_r * cdf_d2
        + q * S * disc_q * cdf_d1
    )
    theta_daily = theta_annual / 252.0  # Decaimiento por rueda de mercado
    
    rho = K * T * disc_r * cdf_d2

    return {
        "delta": delta,
        "gamma": gamma,
        "theta_annual": theta_annual,
        "theta_daily": theta_daily,
        "vega": vega,
        "rho": rho,
        "d1": d1,
        "d2": d2,
    }


def strike_from_delta(
    S: Union[float, np.ndarray, pd.Series],
    delta_target: float,
    T: Union[float, np.ndarray, pd.Series],
    r: Union[float, np.ndarray, pd.Series],
    sigma: Union[float, np.ndarray, pd.Series],
    q: Union[float, np.ndarray, pd.Series] = 0.0,
) -> Union[float, np.ndarray, pd.Series]:
    """
    Despeja analíticamente el Strike K que produce exactamente el Delta objetivo.

    Derivación teórica:
    Delta = e^(-q T) * N(d1)  =>  N(d1) = Delta * e^(q T)
    d1 = N^(-1)(Delta * e^(q T))
    d1 = [ln(S/K) + (r - q + 0.5 * sigma^2)*T] / (sigma * sqrt(T))
    ln(S/K) = d1 * sigma * sqrt(T) - (r - q + 0.5 * sigma^2)*T
    ln(K/S) = (r - q + 0.5 * sigma^2)*T - d1 * sigma * sqrt(T)
    K = S * exp[(r - q + 0.5 * sigma^2)*T - d1 * sigma * sqrt(T)]
    """
    S = np.maximum(np.asarray(S, dtype=float), 1e-6)
    T = np.maximum(np.asarray(T, dtype=float), 1e-6)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)

    # Acotar delta objetivo para no generar divergencias en norm.ppf
    adjusted_delta = np.clip(delta_target * np.exp(q * T), 1e-4, 1.0 - 1e-4)
    d1_target = norm.ppf(adjusted_delta)

    exponent = (r - q + 0.5 * sigma**2) * T - d1_target * sigma * np.sqrt(T)
    strike = S * np.exp(exponent)
    return strike


def estimate_asset_implied_vol(
    vix_level: Union[float, np.ndarray, pd.Series],
    asset_garch_vol: Union[float, np.ndarray, pd.Series],
    spy_garch_vol: Union[float, np.ndarray, pd.Series],
    kalman_beta: Union[float, np.ndarray, pd.Series] = 1.0,
    idiosyncratic_vol: Union[float, np.ndarray, pd.Series] = 0.0,
) -> Union[float, np.ndarray, pd.Series]:
    """
    Estima la volatilidad implícita específica del activo calibrada contra el VIX.
    
    Para SPY: sigma_impl = VIX / 100.
    Para QQQ / NVDA:
    Combina el ratio VRP del mercado (VIX / SPY_GARCH) con la volatilidad propia del activo:
    sigma_impl,i = asset_garch_vol * (VIX / SPY_GARCH)
    
    Esto asegura que:
    1. Se capture la prima de riesgo de varianza (VRP) sistémica del mercado.
    2. La vol implícita de activos volátiles (NVDA, QQQ) sea coherente con su riesgo real
       y mayor a la del SPY cuando su beta o volatilidad idiosincrática son elevadas.
    """
    vix_dec = np.maximum(np.asarray(vix_level, dtype=float) / 100.0, 0.05)
    spy_vol = np.maximum(np.asarray(spy_garch_vol, dtype=float), 0.05)
    asset_vol = np.maximum(np.asarray(asset_garch_vol, dtype=float), 0.05)

    market_vrp_ratio = vix_dec / spy_vol
    # Acotamos el ratio para evitar distorsiones extremas
    clamped_ratio = np.clip(market_vrp_ratio, 0.6, 2.5)

    asset_implied = asset_vol * clamped_ratio
    return asset_implied


def covered_call_pnl_at_expiry(
    S_entry: float,
    S_expiry: float,
    strike: float,
    premium_received: float,
    dividend: float = 0.0,
) -> Dict[str, float]:
    """
    Calcula la descomposición exacta del P&L a vencimiento de una posición Covered Call (+S - C(K)).

    Fórmulas:
    Stock P&L = S_expiry - S_entry + dividend
    Call Payoff a vencimiento = max(S_expiry - strike, 0)
    Call P&L = premium_received - Call Payoff
    Covered Call P&L = Stock P&L + Call P&L
    Asignación (Early o Expiry): True si S_expiry > strike.
    """
    stock_pnl = S_expiry - S_entry + dividend
    call_payoff = max(S_expiry - strike, 0.0)
    call_pnl = premium_received - call_payoff
    cc_pnl = stock_pnl + call_pnl
    is_assigned = S_expiry > strike

    # Rendimientos porcentuales sobre capital inicial invertido (S_entry)
    stock_ret = stock_pnl / S_entry
    cc_ret = cc_pnl / S_entry
    upside_capped_amount = max(S_expiry - strike, 0.0)

    return {
        "stock_pnl": stock_pnl,
        "call_pnl": call_pnl,
        "covered_call_pnl": cc_pnl,
        "stock_return": stock_ret,
        "covered_call_return": cc_ret,
        "is_assigned": bool(is_assigned),
        "upside_capped_loss": upside_capped_amount,
        "premium_buffer_pct": premium_received / S_entry,
    }


def compute_variance_risk_premium(
    S: float,
    strike: float,
    T: float,
    r: float,
    sigma_market: float,
    sigma_garch: float,
    q: float = 0.0,
) -> Dict[str, float]:
    """
    Calcula el Variance Risk Premium (VRP) en términos de prima de opción:
    VRP_dollar = C(sigma_market) - C(sigma_garch)
    Representa el margen monetario esperado que captura el vendedor de volatilidad
    debido a que el mercado cotiza vol implícita por encima de la vol esperada.
    """
    c_market = black_scholes_call_price(S, strike, T, r, sigma_market, q=q)
    c_real = black_scholes_call_price(S, strike, T, r, sigma_garch, q=q)
    vrp_dollar = c_market - c_real
    vrp_vol_diff = sigma_market - sigma_garch

    return {
        "c_market": float(c_market),
        "c_garch": float(c_real),
        "vrp_dollar": float(vrp_dollar),
        "vrp_dollar_pct_spot": float(vrp_dollar / S),
        "vrp_vol_spread": float(vrp_vol_diff),
    }
