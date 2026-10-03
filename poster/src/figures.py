"""Figuras del póster (PDF vectorial + PNG). Paleta de referencia dataviz, slots 1-3 (validan all-pairs)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "overleaf" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_BH, C_STAT, C_MOD = "#2a78d6", "#eb6834", "#1baf7a"   # azul, naranja, aqua
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
GREY = "#9a9990"

plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 19, "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 15, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "lines.linewidth": 2.4, "savefig.facecolor": SURF, "pdf.fonttype": 42,
})

STRAT_COLORS = {"Buy & Hold": C_BH, "CC estático ATM": C_STAT, "CC modelo": C_MOD}


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _years(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def fig_payoff():
    S = np.linspace(70, 130, 300)
    K, prem = 100, 4
    stock = S - 100
    call = -np.maximum(S - K, 0) + prem
    cc = stock + call
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(S, stock, color=C_BH, label="Solo acción")
    ax.plot(S, cc, color=C_MOD, label="Covered Call")
    ax.axhline(0, color=INK2, lw=1)
    ax.axvline(K, color=GREY, lw=1.2, ls="--")
    ax.annotate("Techo: prima − (K − S₀)", xy=(118, cc[-1]), xytext=(94, 17), color=INK2, fontsize=15,
                arrowprops=dict(arrowstyle="-", color=GREY))
    ax.annotate("La prima amortigua\nla caída", xy=(80, cc[np.argmin(abs(S - 80))]), xytext=(70, 8), color=INK2,
                fontsize=15, arrowprops=dict(arrowstyle="-", color=GREY))
    ax.set_xlabel("Precio al vencimiento $S_T$")
    ax.set_ylabel("Resultado ($)")
    ax.text(K + 0.8, -27, "K", color=INK2, fontsize=15)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "fig_payoff")


def fig_vrp(fcs: dict):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True)
    for ax, (k, fc) in zip(axes, fcs.items()):
        ax.plot(fc.index, fc["IV_sig"] * 100, color=C_STAT, label="Volatilidad implícita (VIX / VXN)")
        ax.plot(fc.index, fc["RV"] * 100, color=C_BH, label="Volatilidad realizada del ciclo")
        ax.fill_between(fc.index, fc["RV"] * 100, fc["IV_sig"] * 100, where=fc["IV_sig"] > fc["RV"], color=C_STAT, alpha=0.18, lw=0)
        ax.set_title(f"{k}   (IV media {fc['IV_sig'].mean()*100:.1f}%  vs  realizada {fc['RV'].mean()*100:.1f}%)", loc="left")
        ax.set_ylabel("Vol. anualizada (%)")
        _years(ax)
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_vrp")


def fig_forecast(fcs: dict):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True)
    for ax, (k, fc) in zip(axes, fcs.items()):
        ax.plot(fc.index, fc["RV"] * 100, color=GREY, lw=1.8, label="Realizada")
        ax.plot(fc.index, fc["IV_sig"] * 100, color=C_STAT, lw=1.8, label="Implícita")
        ax.plot(fc.index, fc["Ensamble"] * 100, color=C_MOD, label="Ensamble (modelo propio)")
        ax.set_title(k, loc="left")
        ax.set_ylabel("Vol. anualizada (%)")
        _years(ax)
    axes[0].legend(frameon=False, ncol=3, loc="upper right", fontsize=14)
    fig.tight_layout()
    _save(fig, "fig_forecast")


def fig_errors(tables: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), sharey=True)
    order = pd.concat([t["QLIKE"] for t in tables.values()], axis=1).mean(axis=1).sort_values().index
    for i, (ax, (k, t)) in enumerate(zip(axes, tables.items())):
        t = t.loc[order]
        colors = [C_MOD if m == "Ensamble" else (C_STAT if m == "IV" else C_BH) for m in t.index]
        bars = ax.barh(t.index, t["QLIKE"], color=colors, height=0.62)
        if i == 0:
            ax.invert_yaxis()  # sharey: se invierte una sola vez
        ax.set_title(k, loc="left")
        ax.set_xlabel("QLIKE  (menor = mejor)")
        for b, v in zip(bars, t["QLIKE"]):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=14, color=INK2)
        ax.set_xlim(0, t["QLIKE"].max() * 1.18)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "fig_errors")


def fig_signal(fcs: dict, results: dict):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True)
    for ax, (k, fc) in zip(axes, fcs.items()):
        spread = (fc["IV_sig"] - fc["Ensamble"]) * 100
        sell = results[k]["CC modelo"]["sell"].values.astype(bool)
        ax.bar(fc.index[sell], spread[sell], width=22, color=C_MOD, label="Vende (σ̂ < IV)")
        ax.bar(fc.index[~sell], spread[~sell], width=22, color=C_STAT, label="No vende (σ̂ ≥ IV)")
        ax.axhline(0, color=INK2, lw=1)
        ax.set_title(f"{k}   (vende {sell.mean()*100:.0f}% de los meses)", loc="left")
        ax.set_ylabel("IV − σ̂  (pp)")
        _years(ax)
    axes[0].legend(frameon=False, loc="lower right", fontsize=14)
    fig.tight_layout()
    _save(fig, "fig_signal")


def fig_equity(results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=False)
    for ax, (k, R) in zip(axes, results.items()):
        for name in ["Buy & Hold", "CC estático ATM", "CC modelo"]:
            eq = (1 + R[name]["ret"]).cumprod()
            x = pd.concat([pd.Series([R[name].index[0]]), pd.Series(R[name]["texp"].values)]).values
            y = np.concatenate([[1.0], eq.values])
            ax.plot(x, y, color=STRAT_COLORS[name], label=name)
            ax.annotate(f"{y[-1]:.1f}×", xy=(x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=15, color=INK2)
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 5, 10, 20])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(k, loc="left")
        ax.set_ylabel("Valor de $1 invertido (escala log)")
        ax.xaxis.set_major_locator(mdates.YearLocator(6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.margins(x=0.1)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig_equity")


def fig_ablation(summary: dict):
    names = ["Buy & Hold", "CC estático ATM", "Siempre + strike modelo", "Timing modelo + ATM", "CC modelo"]
    labels = ["B&H", "CC\nestático", "Solo\nstrike", "Solo\ntiming", "CC modelo\n(ambos)"]
    colors = [C_BH, C_STAT, GREY, GREY, C_MOD]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    for ax, (k, s) in zip(axes, summary.items()):
        v = [s.loc[n, "sharpe"] for n in names]
        bars = ax.bar(range(5), v, color=colors, width=0.66)
        ax.set_xticks(range(5))
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_title(k, loc="left")
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x + 0.012, f"{x:.2f}", ha="center", fontsize=14, color=INK2)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(max(s.loc[n, "sharpe"] for n in names) * 1.15, 0.5))
    axes[0].set_ylabel("Sharpe")
    fig.tight_layout()
    _save(fig, "fig_ablation")
