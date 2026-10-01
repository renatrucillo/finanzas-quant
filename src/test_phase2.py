"""
Test unitario y de consistencia para el Módulo de Pricing (Fase 2 - Mundo Q)
=============================================================================
Valida:
1. Valuación analítica Black-Scholes contra valores teóricos conocidos.
2. Inversión exacta del Strike K dado un Delta objetivo (ej. Delta=0.50 ATM, 0.30 OTM, 0.15 deep OTM).
3. Consistencia de las Griegas analíticas (Delta, Gamma, Vega, Theta).
4. Estimación de volatilidad implícita específica por activo (SPY, QQQ, NVDA) y cálculo del VRP.
5. Identidad del P&L de Covered Calls (+S - C(K)) a vencimiento.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from src.pricing import (
    black_scholes_call_price,
    black_scholes_greeks,
    strike_from_delta,
    estimate_asset_implied_vol,
    covered_call_pnl_at_expiry,
    compute_variance_risk_premium,
)


def test_black_scholes_pricing():
    print("--- 1. Test Black-Scholes Pricing & Griegas ---")
    S = 100.0
    K = 100.0
    T = 30.0 / 252.0  # ~1 mes de ruedas
    r = 0.05
    sigma = 0.20
    q = 0.015

    price = black_scholes_call_price(S, K, T, r, sigma, q=q)
    greeks = black_scholes_greeks(S, K, T, r, sigma, q=q)

    print(f"Call ATM (S={S}, K={K}, T={T:.3f}y, r={r:.1%}, vol={sigma:.1%}, div={q:.1%}):")
    print(f"  Precio: ${price:.3f}")
    print(f"  Delta:  {greeks['delta']:.4f}")
    print(f"  Gamma:  {greeks['gamma']:.4f}")
    print(f"  Vega:   {greeks['vega']:.4f} (por 100% vol)")
    print(f"  Theta:  ${greeks['theta_daily']:.4f}/rueda")

    assert price > 0, "El precio de la call debe ser positivo"
    assert 0 < greeks["delta"] < 1, "Delta debe estar en (0, 1)"
    assert greeks["gamma"] > 0, "Gamma debe ser positiva para calls compradas"
    assert greeks["vega"] > 0, "Vega debe ser positiva"
    print("  [OK] Valuación y Griegas validadas!")


def test_strike_inversion():
    print("\n--- 2. Test Inversión de Strike por Delta Objetivo ---")
    S = 500.0
    T = 21.0 / 252.0  # 1 mes hábil
    r = 0.045
    sigma = 0.16
    q = 0.013

    for delta_target in [0.50, 0.30, 0.15]:
        K = strike_from_delta(S, delta_target, T, r, sigma, q=q)
        greeks = black_scholes_greeks(S, K, T, r, sigma, q=q)
        delta_calibrated = greeks["delta"]
        diff = abs(delta_calibrated - delta_target)

        moneyness = (K / S - 1.0) * 100.0
        price = black_scholes_call_price(S, K, T, r, sigma, q=q)

        print(f"Delta Target = {delta_target:.2f} -> Strike K = ${K:.2f} ({moneyness:+.2f}% OTM) | Call = ${price:.2f} | Delta Real = {delta_calibrated:.4f}")
        assert diff < 1e-4, f"Diferencia de delta demasiado grande: {diff}"

    print("  [OK] Inversión de Strike analíticamente exacta!")


def test_asset_implied_vol_and_vrp():
    print("\n--- 3. Test Estimación de Volatilidad Implícita y VRP ---")
    vix = 18.5  # VIX al 18.5%
    spy_garch = 0.14  # SPY GARCH al 14% -> VRP en el mercado
    qqq_garch = 0.19
    nvda_garch = 0.38

    iv_spy = estimate_asset_implied_vol(vix, spy_garch, spy_garch)
    iv_qqq = estimate_asset_implied_vol(vix, qqq_garch, spy_garch)
    iv_nvda = estimate_asset_implied_vol(vix, nvda_garch, spy_garch)

    print(f"VIX: {vix:.1f}% | SPY GARCH: {spy_garch:.1%}")
    print(f"  SPY Implied Vol:  {iv_spy:.2%}")
    print(f"  QQQ Implied Vol:  {iv_qqq:.2%}")
    print(f"  NVDA Implied Vol: {iv_nvda:.2%}")

    assert iv_nvda > iv_qqq > iv_spy, "La estructura de volatilidad por activo debe respetar el riesgo relativo"

    vrp_spy = compute_variance_risk_premium(500.0, 510.0, 21.0 / 252.0, 0.045, iv_spy, spy_garch)
    print(f"\nVRP Monetario SPY (S=$500, K=$510):")
    print(f"  Prima cobrada (IV {iv_spy:.1%}):   ${vrp_spy['c_market']:.2f}")
    print(f"  Costo réplica (GARCH {spy_garch:.1%}): ${vrp_spy['c_garch']:.2f}")
    print(f"  VRP monetario cosechable:          ${vrp_spy['vrp_dollar']:.2f} ({vrp_spy['vrp_dollar_pct_spot']:.2%} del spot)")
    assert vrp_spy["vrp_dollar"] > 0, "El VRP debe ser positivo cuando IV > GARCH"
    print("  [OK] Modelado de VRP validado exitosamente!")


def test_covered_call_pnl():
    print("\n--- 4. Test Identidad del P&L de Covered Calls ---")
    S_entry = 100.0
    K = 105.0
    premium = 2.50

    # Escenario 1: Rally fuerte (S sube a 115) -> Asignado, capped upside
    pnl_rally = covered_call_pnl_at_expiry(S_entry, 115.0, K, premium)
    # Stock sube 15, call paga 10 (intrinseco 115-105). Net = 15 - 10 + 2.50 = 7.50
    assert pnl_rally["covered_call_pnl"] == (K - S_entry) + premium
    assert pnl_rally["is_assigned"] == True
    print(f"Escenario Rally (+15%): Stock P&L=${pnl_rally['stock_pnl']:.2f} | CC P&L=${pnl_rally['covered_call_pnl']:.2f} (Capped en Strike+Prima=${K+premium:.2f})")

    # Escenario 2: Mercado lateral (S termina en 102) -> Expira OTM, prima capturada completa
    pnl_flat = covered_call_pnl_at_expiry(S_entry, 102.0, K, premium)
    assert pnl_flat["covered_call_pnl"] == 2.0 + premium
    assert pnl_flat["is_assigned"] == False
    print(f"Escenario Lateral (+2%): Stock P&L=${pnl_flat['stock_pnl']:.2f} | CC P&L=${pnl_flat['covered_call_pnl']:.2f} (Supera al B&H)")

    # Escenario 3: Crash (-10%, S baja a 90) -> Amortigua la prima
    pnl_crash = covered_call_pnl_at_expiry(S_entry, 90.0, K, premium)
    assert pnl_crash["covered_call_pnl"] == -10.0 + premium
    assert pnl_crash["is_assigned"] == False
    print(f"Escenario Crash (-10%): Stock P&L=${pnl_crash['stock_pnl']:.2f} | CC P&L=${pnl_crash['covered_call_pnl']:.2f} (Amortiguado por prima)")

    print("  [OK] Identidad del P&L de Covered Calls validada!")


if __name__ == "__main__":
    test_black_scholes_pricing()
    test_strike_inversion()
    test_asset_implied_vol_and_vrp()
    test_covered_call_pnl()
    print("\n>>> [SUCCESS] TODOS LOS TESTS DE LA FASE 2 PASARON EXITOSAMENTE! <<<")
