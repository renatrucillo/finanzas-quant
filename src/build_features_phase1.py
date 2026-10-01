"""
Pipeline Orquestador de la Fase 1: Ingeniería de Features (Mundo P)
==================================================================
Integra todos los módulos cuantitativos de la Fase 1:
1. Ingesta y alineación de datos diarios OHLCV + macro.
2. Estimadores de volatilidad de rango (Parkinson, Garman-Klass, Yang-Zhang, Close-to-Close).
3. Modelado de volatilidad condicional asimétrica GJR-GARCH(1,1) (Clase 4).
4. Term-Structure del VIX y proceso Ornstein-Uhlenbeck con Filtro de Kalman (Clase 7).
5. Diferenciación fraccionaria con memoria óptima d* (Clase 6).
6. CAPM dinámico con Filtro de Kalman 2D para el panel de activos (Clase 4 y 7).
7. Cálculo del Variance Risk Premium (VRP) implícito vs realizado.
8. Exportación consolidada a data/features_phase1.parquet sin lookahead bias.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.volatility_estimators import (
    close_to_close_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    yang_zhang_volatility,
)
from src.kalman_filter import kalman_beta_1d, kalman_capm_2d
from src.ou_process import compute_rolling_ou_scores
from src.fracdiff import find_optimal_d
from src.garch_model import fit_gjr_garch


def build_phase1_dataset(
    ohlcv_path: str = str(ROOT_DIR / "data" / "panel_ohlcv.parquet"),
    output_path: str = str(ROOT_DIR / "data" / "features_phase1.parquet"),
    assets: list[str] = ["SPY", "QQQ", "NVDA"],
    vol_window: int = 21,
    ou_window: int = 60,
) -> pd.DataFrame:
    """
    Construye y exporta la matriz completa de features para la Fase 1.
    """
    print(f"--- Iniciando Pipeline de Features Fase 1 ---")
    print(f"Cargando dataset base desde: {ohlcv_path}")
    raw_df = pd.read_parquet(ohlcv_path)

    features = pd.DataFrame(index=raw_df.index)

    # 1. Variables Macroeconómicas Base
    rf_annual = raw_df["IRX"] / 100.0
    rf_daily = rf_annual / 252.0
    vix = raw_df["VIX"] / 100.0      # En decimal
    vix3m = raw_df["VIX3M"] / 100.0  # En decimal

    features["rf_annual"] = rf_annual
    features["rf_daily"] = rf_daily
    features["vix_level"] = raw_df["VIX"]
    features["vix3m_level"] = raw_df["VIX3M"]

    # 2. Term-Structure del VIX y Modelado Ornstein-Uhlenbeck (Clase 7)
    print("\n[Fase 1.1] Procesando Estructura Temporal VIX y Proceso Ornstein-Uhlenbeck...")
    log_vix = np.log(raw_df["VIX"])
    log_vix3m = np.log(raw_df["VIX3M"])

    # Spread logarítmico base
    base_spread = log_vix - log_vix3m
    features["vix_log_spread"] = base_spread

    # Cointegración dinámica mediante Filtro de Kalman (Clase 7)
    vix_beta_k, vix_spread_k = kalman_beta_1d(log_vix, log_vix3m)
    features["vix_kalman_beta"] = vix_beta_k
    features["vix_kalman_spread"] = vix_spread_k

    # Calibración rodante Ornstein-Uhlenbeck sobre el spread
    df_ou = compute_rolling_ou_scores(base_spread, window=ou_window)
    features["ou_theta"] = df_ou["ou_theta"]
    features["ou_kappa"] = df_ou["ou_kappa"]
    features["ou_half_life"] = df_ou["ou_half_life"]
    features["ou_sigma_eq"] = df_ou["ou_sigma_eq"]
    features["ou_s_score"] = df_ou["ou_s_score"]
    features["ou_regime"] = df_ou["regime_label"]

    # Retornos del mercado (SPY) para el CAPM dinámico
    spy_ret = np.log(raw_df["SPY_Adj_Close"] / raw_df["SPY_Adj_Close"].shift(1))
    spy_excess_ret = spy_ret - rf_daily

    # 3. Procesamiento por Activo del Panel (SPY, QQQ, NVDA)
    for sym in assets:
        print(f"\n[Fase 1.2] Procesando activo: {sym}...")
        adj_close = raw_df[f"{sym}_Adj_Close"]
        open_ = raw_df[f"{sym}_Open"]
        high = raw_df[f"{sym}_High"]
        low = raw_df[f"{sym}_Low"]
        close = raw_df[f"{sym}_Close"]
        volume = raw_df[f"{sym}_Volume"]

        # Variables de precio y retornos
        features[f"{sym}_adj_close"] = adj_close
        features[f"{sym}_volume"] = volume
        log_ret = np.log(adj_close / adj_close.shift(1))
        features[f"{sym}_log_ret"] = log_ret

        # Estimadores de Volatilidad Realizada (Mundo P)
        features[f"{sym}_vol_cc_21d"] = close_to_close_volatility(adj_close, window=vol_window)
        features[f"{sym}_vol_parkinson_21d"] = parkinson_volatility(high, low, window=vol_window)
        features[f"{sym}_vol_gk_21d"] = garman_klass_volatility(open_, high, low, close, window=vol_window)
        features[f"{sym}_vol_yz_21d"] = yang_zhang_volatility(open_, high, low, close, window=vol_window)

        # Volatilidad Condicional GJR-GARCH(1,1) Asimétrica
        print(f"  Ajustando GJR-GARCH(1,1) para {sym}...")
        garch_params, cond_vol = fit_gjr_garch(log_ret)
        features[f"{sym}_gjr_cond_vol"] = cond_vol
        print(f"  Parámetros {sym}: alpha={garch_params['alpha']:.4f}, leverage_gamma={garch_params['gamma_leverage']:.4f}, beta={garch_params['beta']:.4f}")

        # CAPM Dinámico con Filtro de Kalman (Clase 4 y 7)
        if sym == "SPY":
            features[f"{sym}_kalman_beta"] = 1.0
            features[f"{sym}_kalman_alpha"] = 0.0
            features[f"{sym}_idiosyncratic_vol_21d"] = 0.0
        else:
            asset_excess_ret = log_ret - rf_daily
            states_capm, resid_capm = kalman_capm_2d(asset_excess_ret, spy_excess_ret)
            features[f"{sym}_kalman_alpha"] = states_capm["alpha_kalman"]
            features[f"{sym}_kalman_beta"] = states_capm["beta_kalman"]
            # Volatilidad idiosincrática anualizada rodante
            idio_vol = resid_capm.rolling(window=vol_window).std() * np.sqrt(252)
            features[f"{sym}_idiosyncratic_vol_21d"] = idio_vol

        # Diferenciación Fraccionaria (Clase 6 - Memoria vs Estacionariedad)
        print(f"  Calculando d* óptimo para {sym}...")
        best_d, summary_d, frac_series = find_optimal_d(
            adj_close,
            d_values=np.linspace(0.1, 0.9, 9),
            adf_significance=0.05
        )
        features[f"{sym}_fracdiff_dopt"] = frac_series
        features[f"{sym}_d_optimal"] = best_d
        print(f"  d* óptimo para {sym}: {best_d:.2f}")

    # 4. Variance Risk Premium (VRP) Indicators (Implied vs Realized)
    print("\n[Fase 1.3] Calculando primas de riesgo de varianza (VRP)...")
    # VRP para el benchmark SPY
    features["vrp_vix_minus_spy_gk"] = vix - features["SPY_vol_gk_21d"]
    features["vrp_vix_minus_spy_garch"] = vix - features["SPY_gjr_cond_vol"]
    features["vrp_ratio_vix_garch"] = vix / np.maximum(features["SPY_gjr_cond_vol"], 1e-4)

    # 5. Limpieza de filas iniciales por ventanas móviles
    initial_valid_idx = features["ou_s_score"].dropna().index[0]
    features_clean = features.loc[initial_valid_idx:].copy()

    # 6. Exportación a Parquet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    features_clean.to_parquet(output_path, engine="pyarrow")
    print(f"\n[OK] Pipeline Fase 1 completado exitosamente!")
    print(f"Dataset exportado a: {output_path}")
    print(f"Dimensiones finales: {features_clean.shape[0]} filas x {features_clean.shape[1]} columnas.")
    print(f"Período: {features_clean.index.min().date()} a {features_clean.index.max().date()}")

    return features_clean


if __name__ == "__main__":
    df_feat = build_phase1_dataset()
    print("\nColumnas generadas:")
    for col in df_feat.columns:
        print(f"  - {col}")
