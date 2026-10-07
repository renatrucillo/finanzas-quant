"""Figuras del backtest (PNG). Paleta dataviz: azul = Buy & Hold, naranja = CC estático, gris = benchmarks, aqua = CC modelo."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "resultados" / "graficos"

C_BH, C_STAT, C_MOD = "#2a78d6", "#eb6834", "#1baf7a"
C_OTM = "#8a63d2"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
GREY = "#9a9990"

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "lines.linewidth": 2.0, "savefig.facecolor": SURF, "pdf.fonttype": 42,
})

STRAT_COLORS = {"Buy & Hold": C_BH, "CC estático ATM": C_STAT, "CC siempre OTM (IV)": C_OTM, "CC modelo": C_MOD,
                "CC modelo (señal VIX)": GREY}


POSTER = False
BASE_RC = {k: plt.rcParams[k] for k in ["font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
                                        "ytick.labelsize", "legend.fontsize", "lines.linewidth"]}
POSTER_RC = {"font.size": 17, "axes.titlesize": 19, "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15,
             "legend.fontsize": 15, "lines.linewidth": 2.4}


def set_output(path: Path, poster: bool = False):
    """poster=True: letra más grande y PDF vectorial además del PNG (para el póster A0)."""
    global OUT, POSTER
    OUT = Path(path)
    OUT.mkdir(parents=True, exist_ok=True)
    POSTER = poster
    plt.rcParams.update(POSTER_RC if poster else BASE_RC)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    if POSTER:
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _years(ax, step=4):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def fig_payoff():
    S = np.linspace(70, 130, 300)
    K, prem = 105, 2.5
    stock = S - 100
    cc = stock - np.maximum(S - K, 0) + prem
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(S, stock, color=C_BH, label="Solo acción")
    ax.plot(S, cc, color=C_MOD, label="Covered Call (K = 105)")
    ax.axhline(0, color=INK2, lw=1)
    ax.axvline(K, color=GREY, lw=1.2, ls="--")
    ax.annotate("Techo:\nprima + (K − S₀)", xy=(122, cc[-1]), xytext=(112, -18), color=INK2,
                arrowprops=dict(arrowstyle="-", color=GREY))
    ax.annotate("La prima amortigua\nla caída", xy=(80, cc[np.argmin(abs(S - 80))]), xytext=(71, 6), color=INK2,
                arrowprops=dict(arrowstyle="-", color=GREY))
    ax.set_xlabel("Precio al vencimiento $S_T$  (S₀ = 100)")
    ax.set_ylabel("Resultado ($)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Payoff de una Covered Call", loc="left")
    _save(fig, "01_payoff")


def fig_skew(sample: pd.DataFrame, skew):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(sample.x, sample.ratio, s=8, alpha=0.3, color=C_BH, label=f"Calls {skew.source} (IBKR, jun–sep 2026)")
    xs = np.linspace(sample.x.min(), sample.x.max(), 100)
    ys = [skew.ratio(x * 0.2 * np.sqrt(0.08), 0.2, 0.08) for x in xs]  # ratio en función de x
    ax.plot(xs, ys, color=C_STAT, lw=2.5, label=f"Ajuste: {skew.a:.2f} {skew.b:+.2f}·x (plano fuera del rango)")
    ax.axvspan(skew.x_lo, skew.x_hi, color=GREY, alpha=0.08)
    ax.axhline(1, color=INK2, lw=1, ls=":")
    ax.set_xlabel("Moneyness estandarizada  x = ln(K/S) / (VIX·√T)")
    ax.set_ylabel("IV de la call / VIX")
    ax.set_title("Las calls cotizan por debajo del VIX", loc="left")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "02_skew_calibracion")


def fig_iv_call(fcs: dict):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    for ax, (k, fc) in zip(axes, fcs.items()):
        idx_name = "VIX" if k == "SPY" else "VXN"
        ax.plot(fc.index, fc["IV_sig"] * 100, color=C_STAT, lw=1.4, label=f"{idx_name} (índice)")
        ax.plot(fc.index, fc["IV_call"] * 100, color=C_MOD, lw=1.4, label="IV de la call vendida")
        ax.plot(fc.index, fc["RV"] * 100, color=C_BH, lw=1.4, alpha=0.8, label="Vol. realizada del ciclo")
        ax.set_title(f"{k}:  {idx_name} {fc['IV_sig'].mean()*100:.1f}%  |  call vendida {fc['IV_call'].mean()*100:.1f}%"
                     f"  |  realizada {fc['RV'].mean()*100:.1f}%  (medias)", loc="left")
        ax.set_ylabel("Vol. anualizada (%)")
        ax.set_ylim(0, 85)
        _years(ax)
    axes[0].legend(frameon=False, loc="upper right", ncol=3)
    fig.tight_layout()
    _save(fig, "03_iv_call_vs_realizada")


def fig_errors(tables: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    order = pd.concat([t["QLIKE"] for t in tables.values()], axis=1).mean(axis=1).sort_values().index
    for i, (ax, (k, t)) in enumerate(zip(axes, tables.items())):
        t = t.loc[order]
        colors = [C_MOD if m == "Ensamble" else (C_STAT if m == "IV" else C_BH) for m in t.index]
        bars = ax.barh(t.index, t["QLIKE"], color=colors, height=0.62)
        if i == 0:
            ax.invert_yaxis()
        ax.set_title(k, loc="left")
        ax.set_xlabel("QLIKE (menor = mejor)")
        for b, v in zip(bars, t["QLIKE"]):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=11, color=INK2)
        ax.set_xlim(0, t["QLIKE"].max() * 1.2)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "04_error_pronostico")


def fig_signal(fcs: dict, results: dict):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True)
    for ax, (k, fc) in zip(axes, fcs.items()):
        R = results[k]["CC modelo"]
        sell = R["sell"].values.astype(bool)
        # en desvío por ciclo, expresado como vol anualizada base 365 de la call
        sd_model = fc["Ensamble"] * np.sqrt(fc["h"] / 252) / np.sqrt(fc["cal"] / 365)
        spread = (fc["IV_call"] - sd_model) * 100
        ax.bar(fc.index[sell], spread[sell], width=22, color=C_MOD, label="Vende (σ̂ < IV de la call)")
        ax.bar(fc.index[~sell], spread[~sell], width=22, color=C_STAT, label="No vende")
        ax.axhline(0, color=INK2, lw=1)
        ax.set_title(f"{k}   (vende {sell.mean()*100:.0f}% de los meses)", loc="left")
        ax.set_ylabel("IV(K) − σ̂  (pp)")
        _years(ax)
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    _save(fig, "05_senal")


def fig_equity(results: dict, names: list):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4) if POSTER else (12, 5.2))
    for ax, (k, R) in zip(axes, results.items()):
        for name in names:
            eq = (1 + R[name]["ret"]).cumprod()
            x = pd.DatetimeIndex([R[name].index[0]] + list(pd.to_datetime(R[name]["texp"])))
            y = np.concatenate([[1.0], eq.values])
            ax.plot(x, y, color=STRAT_COLORS[name], label=f"{name} ({y[-1]:.1f}×)", lw=1.8)
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 5, 10, 20])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(k, loc="left")
        ax.set_ylabel("Valor de $1 (escala log)")
        _years(ax, 3)
        ax.legend(frameon=False, loc="upper left", fontsize=None if POSTER else 10)
    if not POSTER:
        fig.suptitle("Crecimiento de $1, retorno total, 2008–2026", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    _save(fig, "06_crecimiento")


def fig_option_leg(results: dict):
    """Aporte acumulado de vender la call: suma de (retorno estrategia − retorno B&H) por ciclo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (k, R) in zip(axes, results.items()):
        for name in ["CC estático ATM", "CC siempre OTM (IV)", "CC modelo (señal VIX)", "CC modelo"]:
            d = (R[name]["ret"] - R[name]["ret_bh"]).cumsum() * 100
            ax.plot(pd.to_datetime(R[name]["texp"]), d.values, color=STRAT_COLORS[name], label=name, lw=1.8)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_title(k, loc="left")
        ax.set_ylabel("Aporte acumulado de la call (pp)")
        _years(ax, 3)
    axes[0].legend(frameon=False, loc="lower left", fontsize=10)
    fig.suptitle("Vender la call resta: aporte acumulado de la pata de la opción", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    _save(fig, "07_aporte_call")


def fig_drawdown(daily: dict):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True)
    for ax, (k, D) in zip(axes, daily.items()):
        for name in ["Buy & Hold", "CC estático ATM", "CC modelo"]:
            s = D[name]
            dd = (s / s.cummax() - 1) * 100
            ax.plot(dd.index, dd, color=STRAT_COLORS[name], lw=1.2, label=f"{name} (mín {dd.min():.0f}%)")
        ax.set_title(f"{k}: drawdown diario (call marcada a mercado)", loc="left")
        ax.set_ylabel("Drawdown (%)")
        ax.legend(frameon=False, loc="lower right", fontsize=10)
        _years(ax)
    fig.tight_layout()
    _save(fig, "08_drawdown_diario")


def fig_sharpe(summary: dict):
    names = ["Buy & Hold", "CC estático ATM", "CC siempre OTM (IV)", "Siempre + strike modelo", "Timing modelo + ATM",
             "CC modelo (señal VIX)", "CC modelo"]
    labels = ["B&H", "Estático\nATM", "Siempre\nOTM (IV)", "Siempre\nstrike mod.", "Timing\n+ ATM", "Modelo\nseñal VIX", "CC\nmodelo"]
    colors = [C_BH, C_STAT, C_OTM, GREY, GREY, GREY, C_MOD]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (k, s) in zip(axes, summary.items()):
        v = [s.loc[n, "sharpe"] for n in names]
        bars = ax.bar(range(len(names)), v, color=colors, width=0.66)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(k, loc="left")
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x + 0.01, f"{x:.2f}", ha="center", fontsize=10, color=INK2)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Sharpe (sobre excesos)")
    fig.tight_layout()
    _save(fig, "09_sharpe_estrategias")


def fig_pbp(static: pd.DataFrame, r_pbp: np.ndarray):
    x = pd.DatetimeIndex([static.index[0]] + list(pd.to_datetime(static["texp"])))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(x, np.concatenate([[1], (1 + static["ret"]).cumprod().values]), color=C_STAT, label="CC estático ATM simulado (SPY, opción europea)")
    ax.plot(x, np.concatenate([[1], np.cumprod(1 + r_pbp)]), color=INK2, label="ETF BuyWrite real (PBP, neto de comisión 0,75%)")
    corr = np.corrcoef(static["ret"].values, r_pbp)[0, 1]
    ax.set_title(f"Validación del simulador: correlación mensual {corr:.2f}", loc="left")
    ax.set_ylabel("Valor de $1")
    ax.legend(frameon=False, loc="upper left")
    _years(ax, 3)
    _save(fig, "10_validacion_pbp")


def fig_early_exercise(european: pd.DataFrame, american: pd.DataFrame):
    """CC estático ATM sobre SPY: opción europea vs americana (con ejercicio anticipado antes del dividendo)."""
    x = pd.DatetimeIndex([european.index[0]] + list(pd.to_datetime(european["texp"])))
    eq_eu = np.concatenate([[1], (1 + european["ret"]).cumprod().values])
    eq_am = np.concatenate([[1], (1 + american["ret"]).cumprod().values])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    axes[0].plot(x, eq_eu, color=GREY, label=f"Opción europea ({eq_eu[-1]:.2f}×)")
    axes[0].plot(x, eq_am, color=C_STAT, label=f"Opción americana, con ejercicio anticipado ({eq_am[-1]:.2f}×)")
    ex = american[american["early"] == 1]
    axes[0].scatter(pd.to_datetime(ex["texp"]), (1 + american["ret"]).cumprod().loc[ex.index], color=INK, s=14, zorder=3,
                    label=f"Ciclos con ejercicio anticipado ({len(ex)})")
    axes[0].set_ylabel("Valor de $1")
    axes[0].set_title("SPY, CC estático ATM: el dividendo trimestral se pierde cuando la call está en el dinero", loc="left")
    axes[0].legend(frameon=False, loc="upper left", fontsize=10)
    loss = ((american["ret"] - european["ret"]).cumsum() * 100)
    axes[1].plot(pd.to_datetime(american["texp"]), loss.values, color=C_STAT)
    axes[1].axhline(0, color=INK2, lw=1)
    axes[1].set_ylabel("Costo acumulado (pp)")
    _years(axes[1], 3)
    fig.tight_layout()
    _save(fig, "11_ejercicio_anticipado")


def fig_overfitting(grids: dict, summary: dict, extra: dict, n_py: dict):
    """Sharpe de las 36 configuraciones probadas vs Buy & Hold y el umbral del Deflated Sharpe."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (k, g) in zip(axes, grids.items()):
        rng = np.random.default_rng(0)
        for sig, col, lab in [("index", GREY, "Señal VIX (original)"), ("strike", C_MOD, "Señal IV de la call (corregida)")]:
            v = g[g.signal == sig]["sharpe"].values
            ax.scatter(v, rng.uniform(-0.25, 0.25, len(v)) + (0 if sig == "index" else 1), color=col, s=30, alpha=0.8, label=lab)
        bh = summary[k].loc["Buy & Hold", "sharpe"]
        ax.axvline(bh, color=C_BH, lw=2, label=f"Buy & Hold ({bh:.2f})")
        sr0 = extra[k]["dsr"]["sr0_periodo"] * np.sqrt(n_py[k])
        ax.axvline(sr0, color=C_STAT, lw=1.5, ls="--", label=f"Máximo esperado por azar ({sr0:.2f})")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Señal\nVIX", "Señal\ncorregida"])
        ax.set_xlabel("Sharpe anual")
        ax.set_title(f"{k}: 36 configuraciones (6 modelos × 3 z × 2 señales)", loc="left", fontsize=12)
        ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    _save(fig, "12_sobreajuste")


def fig_signal_by_year(results: dict):
    """Porcentaje de meses con venta por año: señal original (VIX) vs corregida (IV de la call)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True)
    for ax, (k, R) in zip(axes, results.items()):
        old = R["CC modelo (señal VIX)"]["sell"].groupby(R["CC modelo (señal VIX)"].index.year).mean() * 100
        new = R["CC modelo"]["sell"].groupby(R["CC modelo"].index.year).mean() * 100
        yrs = old.index.values
        ax.bar(yrs - 0.2, old.values, width=0.4, color=GREY, label=f"Señal VIX (vende {R['CC modelo (señal VIX)']['sell'].mean()*100:.0f}%)")
        ax.bar(yrs + 0.2, new.values, width=0.4, color=C_MOD, label=f"Señal corregida (vende {R['CC modelo']['sell'].mean()*100:.0f}%)")
        ax.set_ylabel("% de meses con venta")
        ax.set_ylim(0, 105)
        ax.set_title(k, loc="left")
        ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(range(2008, 2027, 2))
    fig.tight_layout()
    _save(fig, "13_senal_antes_despues")
