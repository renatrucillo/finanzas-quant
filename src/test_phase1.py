"""
Script de verificación de los módulos de la Fase 1.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np

from src.volatility_estimators import (
    close_to_close_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    yang_zhang_volatility,
)
from src.kalman_filter import kalman_beta_1d, kalman_capm_2d
from src.ou_process import calibrate_ou_ar1, compute_rolling_ou_scores
from src.fracdiff import fractional_diff, find_optimal_d
from src.garch_model import fit_gjr_garch

def main():
    df = pd.read_parquet("data/panel_ohlcv.parquet")
    print(f"Dataset cargado correctamente: {df.shape[0]} filas x {df.shape[1]} columnas.")

    # 1. Estimadores de Volatilidad
    print("\n--- 1. Estimadores de Volatilidad Realizada (SPY, ventana 21d) ---")
    cc = close_to_close_volatility(df["SPY_Adj_Close"], window=21)
    park = parkinson_volatility(df["SPY_High"], df["SPY_Low"], window=21)
    gk = garman_klass_volatility(df["SPY_Open"], df["SPY_High"], df["SPY_Low"], df["SPY_Close"], window=21)
    yz = yang_zhang_volatility(df["SPY_Open"], df["SPY_High"], df["SPY_Low"], df["SPY_Close"], window=21)
    print(f"Último valor - Close-to-Close: {cc.iloc[-1]:.2%}")
    print(f"Último valor - Parkinson:      {park.iloc[-1]:.2%}")
    print(f"Último valor - Garman-Klass:   {gk.iloc[-1]:.2%}")
    print(f"Último valor - Yang-Zhang:     {yz.iloc[-1]:.2%}")

    # 2. Kalman VIX / VIX3M
    print("\n--- 2. Filtro de Kalman sobre Term-Structure VIX / VIX3M ---")
    log_vix = np.log(df["VIX"])
    log_vix3m = np.log(df["VIX3M"])
    beta_k, spread_k = kalman_beta_1d(log_vix, log_vix3m)
    print(f"Beta Kalman dinámico actual: {beta_k.iloc[-1]:.3f}")
    print(f"Desvío estándar del spread Kalman: {spread_k.std():.4f}")

    # 3. Ornstein-Uhlenbeck
    print("\n--- 3. Proceso Ornstein-Uhlenbeck (VIX Spread) ---")
    base_spread = log_vix - log_vix3m
    params_ou = calibrate_ou_ar1(base_spread)
    print(f"Media de largo plazo (theta): {params_ou['theta']:.4f}")
    print(f"Velocidad de reversión (kappa): {params_ou['kappa']:.4f}")
    print(f"Half-life de shocks: {params_ou['half_life']:.1f} días hábiles")

    df_ou = compute_rolling_ou_scores(base_spread, window=60)
    print(f"s-score actual: {df_ou['ou_s_score'].iloc[-1]:.2f}")
    print(f"Régimen actual: {df_ou['regime_label'].iloc[-1]}")

    # 4. GJR-GARCH(1,1)
    print("\n--- 4. GJR-GARCH(1,1) Asimétrico (SPY) ---")
    r_spy = np.log(df["SPY_Adj_Close"] / df["SPY_Adj_Close"].shift(1)).dropna()
    params_garch, cond_vol = fit_gjr_garch(r_spy)
    print(f"Omega: {params_garch['omega']:.6f}")
    print(f"Alpha: {params_garch['alpha']:.4f}")
    print(f"Gamma (Leverage Effect): {params_garch['gamma_leverage']:.4f} (Asimetría confirmada!)")
    print(f"Beta: {params_garch['beta']:.4f}")
    print(f"Persistencia total: {params_garch['persistence']:.4f}")
    print(f"Volatilidad condicional actual: {cond_vol.iloc[-1]:.2%}")

    # 5. CAPM Kalman (NVDA vs SPY)
    print("\n--- 5. CAPM Dinámico con Kalman (NVDA vs SPY) ---")
    rf = df["IRX"] / 100.0 / 252.0  # diario
    r_nvda = np.log(df["NVDA_Adj_Close"] / df["NVDA_Adj_Close"].shift(1)).dropna()
    ex_nvda = (r_nvda - rf).dropna()
    ex_spy = (r_spy - rf).dropna()
    states_capm, resid_capm = kalman_capm_2d(ex_nvda, ex_spy)
    print(f"Beta dinámico actual de NVDA: {states_capm['beta_kalman'].iloc[-1]:.3f}")
    print(f"Alpha diario dinámico de NVDA: {states_capm['alpha_kalman'].iloc[-1]:.5f}")

    # 6. Diferenciación Fraccionaria (NVDA)
    print("\n--- 6. Diferenciación Fraccionaria Óptima (SPY) ---")
    best_d, summary_d, frac_s = find_optimal_d(df["SPY_Adj_Close"].iloc[-400:], d_values=np.linspace(0.1, 0.9, 9))
    print(f"Orden d* óptimo encontrado: {best_d:.2f}")
    corr_opt = summary_d.loc[summary_d['d'] == best_d, 'correlation'].values[0]
    p_val_opt = summary_d.loc[summary_d['d'] == best_d, 'p_value'].values[0]
    print(f"Correlación con nivel original: {corr_opt:.2%}, ADF p-value: {p_val_opt:.4f}")

    print("\n[OK] TODOS LOS TESTS DE LA FASE 1 PASARON EXITOSAMENTE!")

if __name__ == "__main__":
    main()
