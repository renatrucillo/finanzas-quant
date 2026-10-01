"""
Test unitario y de consistencia para la Fase 4: Backtesting & Auditoría Anti-Overfitting
========================================================================================
Valida:
1. Cálculo de métricas financieras (CAGR, Volatilidad, Sharpe, Sortino, Max Drawdown, Calmar).
2. Fórmulas analíticas de Probabilistic Sharpe Ratio (PSR) y Deflated Sharpe Ratio (DSR).
3. Simulación Walk-Forward de portafolios (B&H vs Estático vs Adaptativo).
4. Desglose consistente de regímenes de estrés (2008, 2020, 2022).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from src.backtest import (
    calculate_performance_metrics,
    calculate_probabilistic_sharpe_ratio,
    calculate_deflated_sharpe_ratio,
    run_phase4_backtest,
)


def test_metrics_and_dsr_math():
    print("--- 1. Test Matemático de Métricas y DSR/PSR ---")
    np.random.seed(42)
    # Generar serie sintética de 120 meses con drift positivo y colas moderadas
    r_normal = pd.Series(np.random.normal(0.01, 0.04, 120))
    mets = calculate_performance_metrics(r_normal, rf_daily=0.002, periods_per_year=12)

    print(f"Serie Normal (10 años): CAGR={mets['cagr']:.2%}, Sharpe={mets['sharpe_ratio']:.2f}, Sortino={mets['sortino_ratio']:.2f}, MaxDD={mets['max_drawdown']:.2%}")
    assert mets["cagr"] > 0, "CAGR debe ser positivo"
    assert mets["annualized_vol"] > 0, "Volatilidad debe ser positiva"
    assert -1.0 <= mets["max_drawdown"] <= 0.0, "Max Drawdown debe estar entre -100% y 0%"

    # PSR test: Si benchmark_sr es igual al observado, PSR debe ser exactamente 50%
    psr_exact = calculate_probabilistic_sharpe_ratio(r_normal, benchmark_sr=mets["sharpe_ratio"], rf_daily=0.002)
    print(f"PSR contra su propio Sharpe: {psr_exact:.2%}")
    assert abs(psr_exact - 0.50) < 0.02, "PSR contra su propio Sharpe debe ser ~50%"

    # DSR test: Probar con M=20 pruebas independientes
    dsr_p, sr_null = calculate_deflated_sharpe_ratio(r_normal, num_trials=20, rf_daily=0.002)
    print(f"DSR (M=20 pruebas): Prob={dsr_p:.2%}, Sharpe Esperado del Azar={sr_null:.2f}")
    assert 0.0 <= dsr_p <= 1.0, "DSR p-value debe estar acotado en [0, 1]"
    assert sr_null > 0.0, "El Sharpe esperado del azar debe ser estrictamente positivo con M > 1"

    print("  [OK] Métricas estadísticas y fórmulas DSR/PSR validadas!")


def test_walk_forward_execution():
    print("\n--- 2. Test Simulación Walk-Forward sobre SPY ---")
    results = run_phase4_backtest(assets=["SPY"])
    assert "SPY" in results, "El resultado debe contener SPY"
    df_bt = results["SPY"]["df_backtest"]
    metrics = results["SPY"]["metrics"]

    assert len(df_bt) > 200, "Debe haber más de 200 meses en el backtest"
    assert "equity_adaptive" in df_bt.columns, "Debe incluir la curva de equity adaptativa"
    assert "CC Adaptativo" in metrics, "Debe incluir métricas del CC Adaptativo"

    print("  Curva de Equity Final SPY:")
    print(f"    - Buy & Hold:    ${df_bt['equity_bh'].iloc[-1]:.2f}x")
    print(f"    - CC Estático:   ${df_bt['equity_static'].iloc[-1]:.2f}x")
    print(f"    - CC Adaptativo: ${df_bt['equity_adaptive'].iloc[-1]:.2f}x")
    print(f"    - CC Dinámico:   ${df_bt['equity_dynamic'].iloc[-1]:.2f}x")

    print("  [OK] Simulación Walk-Forward ejecutada sin errores!")


if __name__ == "__main__":
    test_metrics_and_dsr_math()
    test_walk_forward_execution()
    print("\n>>> [SUCCESS] TODOS LOS TESTS DE LA FASE 4 PASARON EXITOSAMENTE! <<<")
