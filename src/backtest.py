"""
Motor de Backtesting Sistemático Walk-Forward y Auditoría Anti-Overfitting (Fase 4)
==================================================================================
Marco teórico:
- Clase 5: Teoría de Portafolios, Simulación Walk-Forward y Métricas de Rendimiento.
- Clase 6: Métodos Anti-Overfitting: Probabilistic Sharpe Ratio (PSR) y Deflated Sharpe Ratio (DSR)
  (Bailey & López de Prado, 2014).
- Clase 4: Análisis comparativo en panel CAPM (SPY, QQQ, NVDA).

Estrategias evaluadas:
1. Buy & Hold (B&H): Portafolio pasivo 100% invertido en el subyacente.
2. Covered Call Estático (BXM-style): Venta incondicional mensual a Delta = 0.30.
3. Covered Call Adaptativo (Meta-Modelo): Venta condicionada al filtro de régimen P(y=1|X_t) >= tau.
4. Covered Call Adaptativo Dinámico: Selección adaptativa de Delta (0.30 si P alto, 0.15 si P moderado, 0.0 si P bajo).
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

from src.pricing import (
    black_scholes_call_price,
    strike_from_delta,
    estimate_asset_implied_vol,
    covered_call_pnl_at_expiry,
)
from src.meta_labeling import (
    generate_covered_call_triple_barrier_labels,
    prepare_meta_features,
    train_and_evaluate_meta_models,
)


# ==============================================================================
# 1. MÉTRICAS ESTADÍSTICAS Y AUDITORÍA ANTI-OVERFITTING (Clase 5 & 6)
# ==============================================================================

def calculate_performance_metrics(
    series_returns: pd.Series,
    rf_daily: Union[float, pd.Series] = 0.0,
    periods_per_year: int = 12,  # Retornos mensuales (o 252 si diarios)
) -> Dict[str, float]:
    """
    Calcula el panel completo de métricas financieras de rendimiento y riesgo.
    """
    r = series_returns.dropna()
    if len(r) < 2:
        return {}

    if isinstance(rf_daily, pd.Series):
        rf_aligned = rf_daily.loc[r.index].fillna(0.0)
    else:
        rf_aligned = rf_daily

    excess_ret = r - rf_aligned
    cum_equity = (1.0 + r).cumprod()

    total_return = cum_equity.iloc[-1] - 1.0
    n_years = len(r) / float(periods_per_year)
    cagr = (cum_equity.iloc[-1]) ** (1.0 / max(n_years, 1e-4)) - 1.0

    mean_ret_annual = r.mean() * periods_per_year
    vol_annual = r.std() * np.sqrt(periods_per_year)

    # Sharpe Ratio Anualizado
    sr = (mean_ret_annual - (rf_aligned.mean() * periods_per_year if isinstance(rf_aligned, pd.Series) else rf_aligned * periods_per_year)) / max(vol_annual, 1e-6)

    # Sortino Ratio Anualizado (Downside Deviation)
    downside_returns = r[r < 0.0]
    downside_vol = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 1 else 1e-6
    sortino = (mean_ret_annual - (rf_aligned.mean() * periods_per_year if isinstance(rf_aligned, pd.Series) else rf_aligned * periods_per_year)) / max(downside_vol, 1e-6)

    # Maximum Drawdown y Duración
    peak = cum_equity.cummax()
    drawdown = (cum_equity - peak) / peak
    max_dd = drawdown.min()

    # Calmar Ratio
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else np.nan

    # Asimetría y Kurtosis
    sk = float(skew(r))
    kt = float(kurtosis(r, fisher=False))  # Kurtosis no centrada (normal = 3)

    # Win rate
    win_rate = (r > 0).mean()

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_vol": float(vol_annual),
        "sharpe_ratio": float(sr),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar_ratio": float(calmar),
        "skewness": sk,
        "kurtosis": kt,
        "win_rate": float(win_rate),
        "n_periods": len(r),
        "n_years": float(n_years),
    }


def calculate_probabilistic_sharpe_ratio(
    strategy_returns: pd.Series,
    benchmark_sr: float,
    rf_daily: Union[float, pd.Series] = 0.0,
    periods_per_year: int = 12,
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR) - Bailey & López de Prado (2012, Clase 6).
    
    Evalúa la probabilidad de que el Sharpe observado sea estrictamente mayor al del benchmark:
    PSR(SR*) = N( (SR_hat - SR*) * sqrt(N - 1) / sqrt(1 - gamma3 * SR_hat + ((gamma4 - 1)/4) * SR_hat^2) )
    """
    r = strategy_returns.dropna()
    N = len(r)
    if N < 5:
        return 0.50

    if isinstance(rf_daily, pd.Series):
        rf_aligned = rf_daily.loc[r.index].fillna(0.0)
    else:
        rf_aligned = rf_daily

    excess_r = r - rf_aligned
    sk = float(skew(excess_r))
    kt = float(kurtosis(excess_r, fisher=True)) + 3.0  # Kurtosis Pearson (normal=3)

    # Desestandarizar SR anualizado a escala de frecuencia
    sr_per_period = excess_r.mean() / max(excess_r.std(), 1e-6)
    sr_star_period = benchmark_sr / np.sqrt(periods_per_year)

    denom = 1.0 - sk * sr_per_period + ((kt - 1.0) / 4.0) * (sr_per_period ** 2)
    denom = max(denom, 1e-6)
    z_stat = (sr_per_period - sr_star_period) * np.sqrt(N - 1) / np.sqrt(denom)

    psr = float(norm.cdf(z_stat))
    return psr


def calculate_deflated_sharpe_ratio(
    strategy_returns: pd.Series,
    num_trials: int = 10,
    var_sharpe_trials: float = 0.05,
    rf_daily: Union[float, pd.Series] = 0.0,
    periods_per_year: int = 12,
) -> Tuple[float, float]:
    """
    Deflated Sharpe Ratio (DSR) - Bailey & López de Prado (2014, Clase 6, diapositiva 50).
    
    Ajusta el Sharpe por la hipótesis nula de selección múltiple (data snooping / multiple testing).
    Calcula el Sharpe esperado del azar:
    SR_0 = sqrt(2 * ln(M)) * (1 - gamma_euler / (2*ln(M)))
    donde M es el número de combinaciones probadas.
    
    Retorna:
    --------
    dsr_pvalue : float
        Probabilidad de que la estrategia no sea producto del sobreajuste (valor en [0, 1]).
    sr_expected_null : float
        Sharpe ratio anualizado esperado por mero azar tras M pruebas.
    """
    r = strategy_returns.dropna()
    N = len(r)
    if N < 5 or num_trials <= 1:
        return 1.0, 0.0

    euler_mascheroni = 0.5772156649
    m = max(num_trials, 2)
    log_m = np.log(m)

    # Sharpe esperado máximo bajo la hipótesis nula tras M pruebas (anualizado)
    sr_0_annual = np.sqrt(var_sharpe_trials) * (
        (1.0 - euler_mascheroni / np.sqrt(2.0 * log_m)) * np.sqrt(2.0 * log_m)
        + euler_mascheroni / np.sqrt(2.0 * log_m)
    )

    # Calcular PSR contra el umbral del azar SR_0
    dsr_prob = calculate_probabilistic_sharpe_ratio(
        strategy_returns,
        benchmark_sr=sr_0_annual,
        rf_daily=rf_daily,
        periods_per_year=periods_per_year,
    )

    return float(dsr_prob), float(sr_0_annual)


# ==============================================================================
# 2. MOTOR DE SIMULACIÓN WALK-FORWARD DE PORTAFOLIOS
# ==============================================================================

def run_walk_forward_backtest(
    df_features: pd.DataFrame,
    df_trades: pd.DataFrame,
    oof_probabilities: pd.Series,
    asset: str = "SPY",
    roll_frequency: int = 21,
    prob_threshold: float = 0.50,
    option_fee_pct: float = 0.0005,  # 5 bps sobre spot por slippage/comisión de opción
    stock_fee_pct: float = 0.0005,   # 5 bps sobre transacciones de acciones
) -> pd.DataFrame:
    """
    Ejecuta la simulación walk-forward comparando los 4 portafolios de forma idéntica:
    1. Buy & Hold (B&H)
    2. Covered Call Estático (Venta incondicional Delta = 0.30)
    3. Covered Call Adaptativo (Filtro binario con Meta-Modelo)
    4. Covered Call Dinámico (Ajuste adaptativo de Delta por confianza de régimen)
    """
    idx = df_trades.index.intersection(oof_probabilities.index)
    trades = df_trades.loc[idx].copy()
    probs = oof_probabilities.loc[idx]

    results = []

    for t0, trade in trades.iterrows():
        S0 = trade["S0"]
        S1 = trade["exit_price"]
        K = trade["K"]
        C0 = trade["C0"]
        r_f_period = df_features.loc[t0, "rf_daily"] * trade["holding_days"]

        # 1. Rendimiento Buy & Hold
        ret_bh = (S1 - S0) / S0

        # 2. Rendimiento Covered Call Estático
        # P&L = S1 - S0 + C0 - max(S1 - K, 0) - costos
        call_payoff = max(S1 - K, 0.0)
        pnl_static = (S1 - S0) + C0 - call_payoff - (option_fee_pct * S0)
        ret_static = pnl_static / S0

        # 3. Rendimiento Covered Call Adaptativo (Meta-Modelo Binario)
        p_favorable = probs.loc[t0]
        sell_call_binary = p_favorable >= prob_threshold

        if sell_call_binary:
            pnl_adaptive = pnl_static
            ret_adaptive = ret_static
            action_adaptive = "Sell_Call_30D"
        else:
            pnl_adaptive = S1 - S0  # 100% libre de capping
            ret_adaptive = ret_bh
            action_adaptive = "Hold_Uncapped"

        # 4. Rendimiento Covered Call Dinámico (Multi-Delta)
        # Alta confianza (P >= 0.65): Delta 0.30
        # Confianza moderada (0.50 <= P < 0.65): Delta 0.15 defensivo
        # Baja confianza (P < 0.50): Veto (Hold Uncapped)
        if p_favorable >= 0.65:
            delta_dyn = 0.30
            pnl_dyn = pnl_static
            ret_dyn = ret_static
            action_dyn = "Delta_0.30"
        elif p_favorable >= 0.50:
            delta_dyn = 0.15
            # Recalcular strike y prima defensiva para Delta 0.15
            T_exp = roll_frequency / 252.0
            r_ann = df_features.loc[t0, "rf_annual"]
            s_iv = trade["sigma_iv"]
            K_def = strike_from_delta(S0, 0.15, T_exp, r_ann, s_iv, q=0.015)
            C_def = black_scholes_call_price(S0, K_def, T_exp, r_ann, s_iv, q=0.015)
            call_payoff_def = max(S1 - K_def, 0.0)
            pnl_dyn = (S1 - S0) + C_def - call_payoff_def - (option_fee_pct * S0)
            ret_dyn = pnl_dyn / S0
            action_dyn = "Delta_0.15_Defensive"
        else:
            delta_dyn = 0.0
            pnl_dyn = S1 - S0
            ret_dyn = ret_bh
            action_dyn = "Delta_0.0_Uncapped"

        # Registro del período
        results.append({
            "date": t0,
            "expiry_date": trade["t1"],
            "holding_days": trade["holding_days"],
            "S0": S0,
            "S1": S1,
            "K_030": K,
            "C_030": C0,
            "p_model": float(p_favorable),
            "rf_period": float(r_f_period),
            "ret_bh": float(ret_bh),
            "ret_static": float(ret_static),
            "ret_adaptive": float(ret_adaptive),
            "ret_dynamic": float(ret_dyn),
            "action_adaptive": action_adaptive,
            "action_dynamic": action_dyn,
            "is_assigned_030": bool(S1 > K),
            "capping_loss_static": max(S1 - K, 0.0) / S0,
        })

    df_backtest = pd.DataFrame(results).set_index("date")

    # Curvas de Equity Acumuladas
    df_backtest["equity_bh"] = (1.0 + df_backtest["ret_bh"]).cumprod()
    df_backtest["equity_static"] = (1.0 + df_backtest["ret_static"]).cumprod()
    df_backtest["equity_adaptive"] = (1.0 + df_backtest["ret_adaptive"]).cumprod()
    df_backtest["equity_dynamic"] = (1.0 + df_backtest["ret_dynamic"]).cumprod()

    return df_backtest


# ==============================================================================
# 3. ANÁLISIS DE SUB-PERÍODOS DE ESTRÉS (Clase 5, diapositiva 72)
# ==============================================================================

def analyze_stress_subperiods(
    df_backtest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Desglosa el rendimiento cuantitativo en los 5 regímenes macroeconómicos históricos clave:
    1. Crisis Subprime (2007 - 2009)
    2. Bull Market Post-Crisis (2010 - 2019)
    3. Covid Crash & Shock (2020)
    4. Shock Inflacionario & Suba de Tasas (2021 - 2022)
    5. AI & Tech Rally (2023 - 2026)
    """
    subperiods = {
        "1. Crisis Subprime (2007-2009)": ("2007-01-01", "2009-12-31"),
        "2. Bull Market (2010-2019)": ("2010-01-01", "2019-12-31"),
        "3. Covid Crash (2020)": ("2020-01-01", "2020-12-31"),
        "4. Bear Inflacionario (2021-2022)": ("2021-01-01", "2022-12-31"),
        "5. AI Bull Run (2023-2026)": ("2023-01-01", "2026-12-31"),
    }

    records = []

    for name, (start, end) in subperiods.items():
        sub_df = df_backtest.loc[start:end]
        if len(sub_df) < 3:
            continue

        for strat_col, label in [
            ("ret_bh", "Buy & Hold"),
            ("ret_static", "CC Estático"),
            ("ret_adaptive", "CC Adaptativo"),
            ("ret_dynamic", "CC Dinámico"),
        ]:
            mets = calculate_performance_metrics(sub_df[strat_col], rf_daily=sub_df["rf_period"])
            records.append({
                "Período": name,
                "Estrategia": label,
                "CAGR": mets.get("cagr", np.nan),
                "Vol Anual": mets.get("annualized_vol", np.nan),
                "Sharpe": mets.get("sharpe_ratio", np.nan),
                "Sortino": mets.get("sortino_ratio", np.nan),
                "Max Drawdown": mets.get("max_drawdown", np.nan),
                "Win Rate": mets.get("win_rate", np.nan),
            })

    return pd.DataFrame(records)


# ==============================================================================
# 4. ORQUESTADOR PRINCIPAL DE LA FASE 4
# ==============================================================================

def run_phase4_backtest(
    features_path: str = "data/features_phase1.parquet",
    assets: List[str] = ["SPY", "QQQ", "NVDA"],
    num_dsr_trials: int = 20,
) -> Dict[str, any]:
    """
    Ejecuta el backtest completo walk-forward y auditoría anti-overfitting para todo el panel.
    """
    print("=================================================================")
    print("     INICIANDO PIPELINE DE FASE 4: BACKTESTING & AUDITORÍA DSR   ")
    print("=================================================================")
    df_features = pd.read_parquet(features_path)
    all_backtests = {}

    for asset in assets:
        print(f"\n>>> [Fase 4] Ejecutando simulación Walk-Forward para {asset}...")

        # 1. Generar trades con Triple Barrera y entrenar Meta-Modelo
        df_trades = generate_covered_call_triple_barrier_labels(df_features, asset=asset)
        X, y, t1 = prepare_meta_features(df_features, df_trades, asset=asset)
        cv_res = train_and_evaluate_meta_models(X, y, t1, n_splits=5, embargo_pct=0.02)

        # Usar probabilidades del mejor meta-modelo OOF (Random Forest o HistGBM)
        model_name = "RandomForest" if cv_res["metrics"]["RandomForest"]["roc_auc"] >= cv_res["metrics"]["LogisticRegression"]["roc_auc"] else "LogisticRegression"
        oof_probs = cv_res["oof_probabilities"][model_name].fillna(0.50)

        # 2. Correr simulación Walk-Forward
        df_bt = run_walk_forward_backtest(
            df_features=df_features,
            df_trades=df_trades,
            oof_probabilities=oof_probs,
            asset=asset,
            prob_threshold=0.50,
        )

        # 3. Métricas de Rendimiento Totales
        metrics_summary = {}
        for strat, col in [
            ("Buy & Hold", "ret_bh"),
            ("CC Estático", "ret_static"),
            ("CC Adaptativo", "ret_adaptive"),
            ("CC Dinámico", "ret_dynamic"),
        ]:
            m = calculate_performance_metrics(df_bt[col], rf_daily=df_bt["rf_period"])
            # PSR respecto a Buy & Hold
            sr_bh = calculate_performance_metrics(df_bt["ret_bh"], rf_daily=df_bt["rf_period"])["sharpe_ratio"]
            m["psr_vs_bh"] = calculate_probabilistic_sharpe_ratio(df_bt[col], benchmark_sr=sr_bh, rf_daily=df_bt["rf_period"])
            
            # Deflated Sharpe Ratio
            dsr_val, sr_null = calculate_deflated_sharpe_ratio(df_bt[col], num_trials=num_dsr_trials, rf_daily=df_bt["rf_period"])
            m["dsr_prob"] = dsr_val
            m["sr_null_expected"] = sr_null
            metrics_summary[strat] = m

        # 4. Análisis de Regímenes de Estrés
        df_subperiods = analyze_stress_subperiods(df_bt)

        # Reporte de consola
        print(f"\n--- Métricas Consolidadas 2007-2026 ({asset}) ---")
        df_mets = pd.DataFrame(metrics_summary).T[
            ["cagr", "annualized_vol", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio", "psr_vs_bh", "dsr_prob"]
        ]
        df_mets["cagr"] = df_mets["cagr"].map("{:.2%}".format)
        df_mets["annualized_vol"] = df_mets["annualized_vol"].map("{:.2%}".format)
        df_mets["max_drawdown"] = df_mets["max_drawdown"].map("{:.2%}".format)
        df_mets["sharpe_ratio"] = df_mets["sharpe_ratio"].map("{:.2f}".format)
        df_mets["sortino_ratio"] = df_mets["sortino_ratio"].map("{:.2f}".format)
        df_mets["calmar_ratio"] = df_mets["calmar_ratio"].map("{:.2f}".format)
        df_mets["psr_vs_bh"] = df_mets["psr_vs_bh"].map("{:.2%}".format)
        df_mets["dsr_prob"] = df_mets["dsr_prob"].map("{:.2%}".format)
        print(df_mets.to_string())

        all_backtests[asset] = {
            "df_backtest": df_bt,
            "metrics": metrics_summary,
            "df_subperiods": df_subperiods,
            "model_used": model_name,
        }

    print("\n[OK] Pipeline de Fase 4 finalizado con éxito!")
    return all_backtests


if __name__ == "__main__":
    run_phase4_backtest()
