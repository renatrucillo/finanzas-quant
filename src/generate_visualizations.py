"""
Generador de Visualizaciones Cuantitativas de Alta Resolución (Fase 4 & Póster)
================================================================================
Genera gráficos estilizados de grado académico listos para el póster y reporte final:
1. Curvas de Equity Acumulado comparativas (SPY, QQQ, NVDA).
2. Gráficos de Drawdowns subacuáticos en crisis históricas (2008, 2020, 2022).
3. Dinámica del VRP (VIX vs GARCH) y s-score Ornstein-Uhlenbeck.
4. Importancia de Features del Meta-Modelo Supervisado.
5. Rendimiento por Sub-períodos de Estrés (Subprime, Covid, Inflación, AI Boom).
6. El dilema de NVDA: Capping de upside en rallies explosivos vs Desbloqueo Adaptativo.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.backtest import run_phase4_backtest

# Configuración estética profesional
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.titlesize": 16,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIG_DIR = ROOT_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_equity_curves(backtest_results: dict):
    """
    Gráfico 1: Curvas de Equity en escala logarítmica para SPY, QQQ y NVDA.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    colors = {
        "equity_bh": "#1f77b4",        # Azul
        "equity_static": "#ff7f0e",    # Naranja
        "equity_adaptive": "#2ca02c",  # Verde esmeralda
        "equity_dynamic": "#9467bd",   # Púrpura
    }
    labels = {
        "equity_bh": "Buy & Hold Pasivo",
        "equity_static": "Covered Call Estático (Δ=0.30)",
        "equity_adaptive": "Covered Call Adaptativo (Meta-Modelo)",
        "equity_dynamic": "Covered Call Adaptativo Dinámico (Multi-Δ)",
    }

    for idx, asset in enumerate(["SPY", "QQQ", "NVDA"]):
        ax = axes[idx]
        df_bt = backtest_results[asset]["df_backtest"]
        metrics = backtest_results[asset]["metrics"]

        metrics_key_map = {
            "equity_bh": "Buy & Hold",
            "equity_static": "CC Estático",
            "equity_adaptive": "CC Adaptativo",
            "equity_dynamic": "CC Dinámico",
        }

        for col, c in colors.items():
            strat_label = labels[col]
            m_key = metrics_key_map[col]
            cagr_val = metrics[m_key]["cagr"]
            sr_val = metrics[m_key]["sharpe_ratio"]
            ax.plot(
                df_bt.index,
                df_bt[col],
                label=f"{strat_label} (CAGR: {cagr_val:.1%}, SR: {sr_val:.2f})",
                color=c,
                linewidth=1.8 if "Adaptativo" in strat_label else 1.3,
                alpha=0.95 if "Adaptativo" in strat_label else 0.75,
            )

        ax.set_yscale("log")
        ax.set_ylabel("Crecimiento de $1 (Escala Log)")
        ax.set_title(f"Panel {asset}: Evolución Patrimonial Walk-Forward 2007-2026", fontweight="bold")
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Fecha de Negociación")
    fig.suptitle("Comparativa de Curvas de Equity: Buy & Hold vs Covered Calls (2007 - 2026)", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = FIG_DIR / "fig1_equity_curves_panel.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Gráfico 1 Guardado] -> {out_path}")


def plot_drawdowns(backtest_results: dict):
    """
    Gráfico 2: Drawdowns subacuáticos comparativos destacando crisis (2008, 2020, 2022).
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    for idx, asset in enumerate(["SPY", "QQQ", "NVDA"]):
        ax = axes[idx]
        df_bt = backtest_results[asset]["df_backtest"]

        # Calcular series de drawdown
        dd_bh = (df_bt["equity_bh"] - df_bt["equity_bh"].cummax()) / df_bt["equity_bh"].cummax()
        dd_static = (df_bt["equity_static"] - df_bt["equity_static"].cummax()) / df_bt["equity_static"].cummax()
        dd_adapt = (df_bt["equity_adaptive"] - df_bt["equity_adaptive"].cummax()) / df_bt["equity_adaptive"].cummax()

        ax.fill_between(df_bt.index, dd_bh * 100, 0, color="#1f77b4", alpha=0.25, label=f"B&H MaxDD: {dd_bh.min():.1%}")
        ax.plot(df_bt.index, dd_static * 100, color="#ff7f0e", linewidth=1.2, label=f"CC Estático MaxDD: {dd_static.min():.1%}")
        ax.plot(df_bt.index, dd_adapt * 100, color="#2ca02c", linewidth=1.8, label=f"CC Adaptativo MaxDD: {dd_adapt.min():.1%}")

        ax.set_ylabel("Drawdown (%)")
        ax.set_title(f"Perfil de Caídas Máximas ({asset}): Amortiguación de Colas en Crisis", fontweight="bold")
        ax.legend(loc="lower left", frameon=True, framealpha=0.9)
        ax.set_ylim(-65, 5)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Destacar Crisis 2008 y Covid 2020
        ax.axvspan(pd.Timestamp("2007-10-01"), pd.Timestamp("2009-03-31"), color="red", alpha=0.08)
        ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"), color="red", alpha=0.08)

    axes[-1].set_xlabel("Fecha")
    fig.suptitle("Auditoría de Riesgo de Cola: Drawdowns Subacuáticos (2007 - 2026)", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = FIG_DIR / "fig2_drawdowns_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Gráfico 2 Guardado] -> {out_path}")


def plot_vrp_and_regimes():
    """
    Gráfico 3: Dinámica del Variance Risk Premium (VIX vs GARCH) y proceso Ornstein-Uhlenbeck.
    """
    df_feat = pd.read_parquet(ROOT_DIR / "data" / "features_phase1.parquet")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1.3]})

    # Panel Superior: VIX vs SPY GARCH condicional
    vix_dec = df_feat["vix_level"] / 100.0
    garch_vol = df_feat["SPY_gjr_cond_vol"]

    ax1.plot(df_feat.index, vix_dec * 100, label="Volatilidad Implícita VIX (Mundo Q)", color="#d62728", linewidth=1.2, alpha=0.85)
    ax1.plot(df_feat.index, garch_vol * 100, label="Volatilidad Condicional GJR-GARCH SPY (Mundo P)", color="#1f77b4", linewidth=1.4)
    ax1.fill_between(df_feat.index, garch_vol * 100, vix_dec * 100, where=(vix_dec >= garch_vol), color="#2ca02c", alpha=0.25, label="VRP Positivo (Prima Cosechable: IV > RV)")
    ax1.fill_between(df_feat.index, garch_vol * 100, vix_dec * 100, where=(vix_dec < garch_vol), color="#d62728", alpha=0.35, label="VRP Negativo (Vol Realizada Supera Implícita)")

    ax1.set_ylabel("Volatilidad Anualizada (%)")
    ax1.set_title("Estructura del Variance Risk Premium (VRP): Mundo Q (VIX) vs Mundo P (GJR-GARCH)", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel Inferior: s-score Ornstein-Uhlenbeck
    s_score = df_feat["ou_s_score"]
    ax2.plot(df_feat.index, s_score, color="#4b0082", linewidth=1.1, label="s-score Ornstein-Uhlenbeck (Spread VIX/VIX3M)")
    ax2.axhline(0.0, color="gray", linestyle="-", linewidth=0.8)
    ax2.axhline(1.5, color="red", linestyle="--", linewidth=1.0, label="Umbral Estrés Severo (s > 1.5)")
    ax2.axhline(-1.0, color="green", linestyle="--", linewidth=1.0, label="Régimen Calma / Contango (s < -1.0)")
    ax2.fill_between(df_feat.index, 1.5, s_score, where=(s_score >= 1.5), color="red", alpha=0.3)

    ax2.set_ylabel("s-score Normalizado")
    ax2.set_xlabel("Fecha")
    ax2.set_title("Detección de Régimen de Term-Structure mediante Proceso Ornstein-Uhlenbeck", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = FIG_DIR / "fig3_regimes_and_vrp_dynamics.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Gráfico 3 Guardado] -> {out_path}")


def plot_feature_importances(backtest_results: dict):
    """
    Gráfico 4: Importancia de Variables en el Meta-Modelo para SPY, QQQ y NVDA.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

    from src.meta_labeling import prepare_meta_features, train_and_evaluate_meta_models, generate_covered_call_triple_barrier_labels
    df_feat = pd.read_parquet(ROOT_DIR / "data" / "features_phase1.parquet")

    feature_name_map = {
        "drawdown_ath": "Drawdown desde Máximo",
        "vrp_ratio": "Ratio VIX / GARCH",
        "vrp_spread": "Spread VRP (VIX - GARCH)",
        "gjr_cond_vol": "Vol GJR-GARCH",
        "vol_yz": "Vol Realizada Yang-Zhang",
        "vol_gk": "Vol Garman-Klass",
        "ou_s_score": "OU s-score VIX",
        "ou_half_life": "OU Half-Life",
        "vix_level": "Nivel VIX",
        "ou_kappa": "OU Kappa Reversión",
        "vix_kalman_spread": "Spread Kalman VIX",
        "fracdiff_ret": "Fracdiff Retornos d*",
        "premium_yield": "Yield Prima Cobrada",
        "moneyness": "Moneyness Strike",
        "kalman_beta": "Beta Dinámico Kalman",
        "idiosyncratic_vol": "Vol Idiosincrática",
    }

    for idx, asset in enumerate(["SPY", "QQQ", "NVDA"]):
        ax = axes[idx]
        trades = generate_covered_call_triple_barrier_labels(df_feat, asset=asset)
        X, y, t1 = prepare_meta_features(df_feat, trades, asset=asset)
        res = train_and_evaluate_meta_models(X, y, t1)
        imp = res["metrics"]["RandomForest"]["feature_importances"].sort_values(ascending=True)

        # Traducir nombres y tomar top 10
        imp_top10 = imp.tail(10)
        readable_labels = [feature_name_map.get(k, k) for k in imp_top10.index]

        y_pos = np.arange(len(imp_top10))
        bars = ax.barh(y_pos, imp_top10.values, color="#1f77b4" if asset=="SPY" else ("#ff7f0e" if asset=="QQQ" else "#2ca02c"), alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(readable_labels)
        ax.set_xlabel("Importancia Relativa (Gini / MDI)")
        ax.set_title(f"Panel {asset}: Top Features", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Anotación del valor
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.002, bar.get_y() + bar.get_height()/2, f"{w:.1%}", va="center", fontsize=8)

    fig.suptitle("Importancia de Features del Meta-Modelo Supervisado (Random Forest)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = FIG_DIR / "fig4_feature_importances_meta_models.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Gráfico 4 Guardado] -> {out_path}")


def plot_stress_periods(backtest_results: dict):
    """
    Gráfico 5: Desglose de Rendimiento Cuantitativo en las 5 Crisis y Regímenes Macro.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    for idx, asset in enumerate(["SPY", "QQQ", "NVDA"]):
        ax = axes[idx]
        df_sub = backtest_results[asset]["df_subperiods"]

        periods = df_sub["Período"].unique()
        strategies = ["Buy & Hold", "CC Estático", "CC Adaptativo", "CC Dinámico"]
        bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

        x = np.arange(len(periods))
        width = 0.20

        for s_idx, strat in enumerate(strategies):
            sub_strat = df_sub[df_sub["Estrategia"] == strat]
            sharpes = [sub_strat[sub_strat["Período"] == p]["Sharpe"].values[0] if len(sub_strat[sub_strat["Período"] == p]) > 0 else 0.0 for p in periods]
            ax.bar(x + s_idx * width - 0.30, sharpes, width, label=strat if idx == 0 else "", color=bar_colors[s_idx], alpha=0.9)

        ax.set_ylabel("Sharpe Ratio Anualizado")
        ax.set_title(f"Panel {asset}: Sharpe Ratio por Régimen Macroeconómico de Estrés", fontweight="bold")
        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xticks(np.arange(len(periods)))
    axes[-1].set_xticklabels([p.split(". ")[-1] for p in periods], rotation=15, ha="right")
    axes[0].legend(loc="upper left", ncol=4, frameon=True, framealpha=0.9)

    fig.suptitle("Evaluación de Resiliencia en Regímenes de Estrés (Clase 5, Diapositiva 72)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = FIG_DIR / "fig5_stress_periods_performance.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Gráfico 5 Guardado] -> {out_path}")


def plot_nvda_dilemma(backtest_results: dict):
    """
    Gráfico 6: El dilema del Covered Call en activos con alto beta y rallies explosivos (NVDA).
    """
    df_bt = backtest_results["NVDA"]["df_backtest"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # Curvas de NVDA
    ax1.plot(df_bt.index, df_bt["equity_bh"], label="NVDA Buy & Hold Pasivo", color="#1f77b4", linewidth=2.0)
    ax1.plot(df_bt.index, df_bt["equity_static"], label="NVDA CC Estático (Upside Amputado por Venta Incondicional)", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax1.plot(df_bt.index, df_bt["equity_adaptive"], label="NVDA CC Adaptativo (Desbloqueo de Upside con Meta-Modelo)", color="#2ca02c", linewidth=2.2)

    ax1.set_yscale("log")
    ax1.set_ylabel("Crecimiento de $1 (Escala Log)")
    ax1.set_title("La Prueba de Fuego: Rendimiento en un Activo de Alto Beta (NVDA)", fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # Panel inferior: Probabilidad del Meta-Modelo y Acción tomada
    probs = df_bt["p_model"]
    ax2.plot(df_bt.index, probs, color="#4b0082", linewidth=1.2, label="Probabilidad P(y=1|X_t) Meta-Modelo")
    ax2.axhline(0.50, color="red", linestyle="--", linewidth=1.0, label="Umbral Decisión τ = 0.50")

    # Colorear zonas donde el modelo vetó la venta de calls para capturar el rally
    uncapped_mask = probs < 0.50
    ax2.fill_between(df_bt.index, 0, 1, where=uncapped_mask, color="#2ca02c", alpha=0.25, label="Veto de Call: 100% Upside Libre (Hold)")

    ax2.set_ylabel("Probabilidad de Régimen")
    ax2.set_xlabel("Fecha")
    ax2.set_title("Decisiones de Bet Sizing y Filtro de Régimen sobre NVDA", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    out_path = FIG_DIR / "fig6_nvda_upside_capping_dilemma.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Gráfico 6 Guardado] -> {out_path}")


def main():
    print("Iniciando generación de gráficos para el póster y reporte...")
    backtest_results = run_phase4_backtest(assets=["SPY", "QQQ", "NVDA"])

    plot_equity_curves(backtest_results)
    plot_drawdowns(backtest_results)
    plot_vrp_and_regimes()
    plot_feature_importances(backtest_results)
    plot_stress_periods(backtest_results)
    plot_nvda_dilemma(backtest_results)

    print("\n[OK] TODOS LOS GRÁFICOS GENERADOS EXITOSAMENTE EN reports/figures/!")


if __name__ == "__main__":
    main()
