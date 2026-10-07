"""Gráficos de diagnóstico de la Fase 1 (features) -> resultados/graficos/fase1_*.png"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fracdiff import find_optimal_d
from src.garch_model import fit_gjr_garch
from src.kalman_filter import kalman_beta_1d
from src.ou_process import calibrate_ou_ar1
from src.volatility_estimators import garman_klass_volatility

OUT = ROOT_DIR / "resultados" / "graficos"
BLUE, ORANGE, AQUA, GREY, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#9a9990", "#52514e"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "axes.edgecolor": "#e6e5e1", "lines.linewidth": 1.4})


def ou_sscore_figure(f: pd.DataFrame, path: Path, poster: bool = False):
    """Spread ln(VIX) − ln(VIX3M) y s-score OU con el régimen de estrés. poster=True: letra grande + PDF."""
    rc = {"font.size": 17, "axes.titlesize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15} if poster else {}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
        axes[0].plot(f["vix_log_spread"], color=BLUE, lw=0.9)
        axes[0].axhline(0, color=INK2, lw=1)
        axes[0].set_title("Spread ln(VIX) − ln(VIX3M)  (> 0 = backwardation)", loc="left")
        s = f["ou_s_score"]
        axes[1].plot(s, color=GREY, lw=0.8)
        axes[1].fill_between(s.index, 2, s, where=s >= 2, color=ORANGE, alpha=0.6, label="Estrés extremo (s ≥ 2)")
        axes[1].axhline(0, color=INK2, lw=1)
        axes[1].axhline(2, color=ORANGE, lw=1, ls="--")
        hl = f["ou_half_life"].median()
        axes[1].set_title(f"s-score OU (half-life mediano {hl:.0f} ruedas)", loc="left")
        axes[1].legend(frameon=False, loc="upper left")
        fig.tight_layout()
        path = Path(path)
        fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        if poster:
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(ROOT_DIR / "data" / "features_phase1.parquet")
    raw = pd.read_parquet(ROOT_DIR / "data" / "panel_ohlcv.parquet")

    # 1) GARCH: ajuste in-sample (mira el futuro) vs pronóstico walk-forward
    r = np.log(raw["SPY_Adj_Close"]).diff().dropna().rename("SPY")
    _, insample = fit_gjr_garch(r)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    s = slice("2019-06-01", "2021-06-30")
    ax.plot(insample.loc[s] * 100, color=GREY, label="In-sample (parámetros con toda la muestra)")
    ax.plot(f["SPY_gjr_forecast_vol"].loc[s] * 100, color=AQUA, label="Walk-forward (solo pasado)")
    ax.set_title("SPY: GJR-GARCH in-sample vs. walk-forward (detalle Covid)", loc="left")
    ax.set_ylabel("Vol. anualizada (%)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "fase1_garch_walkforward.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) VRP: con Garman-Klass (sesgado) vs Yang-Zhang
    gk = garman_klass_volatility(raw["SPY_Open"], raw["SPY_High"], raw["SPY_Low"], raw["SPY_Close"]).reindex(f.index)
    vix = f["vix_level"] / 100
    fig, ax = plt.subplots(figsize=(10, 4.2))
    a = (vix - gk).rolling(252).mean() * 100
    b = f["vrp_vix_minus_spy_yz"].rolling(252).mean() * 100
    ax.plot(a, color=ORANGE, label=f"VIX − Garman-Klass (media {100*(vix-gk).mean():.1f} pp)")
    ax.plot(b, color=BLUE, label=f"VIX − Yang-Zhang (media {100*f['vrp_vix_minus_spy_yz'].mean():.1f} pp)")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_title("Prima de varianza ex-ante de SPY (media móvil 1 año): GK la infla", loc="left")
    ax.set_ylabel("pp de vol.")
    ax.legend(frameon=False)
    fig.savefig(OUT / "fase1_vrp_gk_vs_yz.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) OU: s-score del spread ln(VIX) - ln(VIX3M) con regímenes
    ou_sscore_figure(f, OUT / "fase1_ou_sscore.png")
    # 4) Kalman: q, r fijados a mano (r con toda la muestra) vs MLE en el primer año
    lv, lv3 = np.log(raw["VIX"]), np.log(raw["VIX3M"])
    b_old, _ = kalman_beta_1d(lv, lv3, q=1e-5, r=float(lv.var() * 0.5))
    b_new, _ = kalman_beta_1d(lv, lv3)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(b_old, color=GREY, label="Antes: q = 1e-5 a mano, r = 0,5·var(toda la muestra)")
    ax.plot(b_new, color=AQUA, lw=1.0, label=f"Ahora: MLE en el primer año (q = {b_new.attrs['q']:.1e}, r = {b_new.attrs['r']:.1e})")
    ax.set_title("Kalman: ratio dinámico ln(VIX) / ln(VIX3M)", loc="left")
    ax.set_ylabel("beta_t")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.savefig(OUT / "fase1_kalman_mle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5) OU: half-life rodante, ventana 60 por MCO (antes) vs ventana 252 con corrección de sesgo (ahora)
    spread = (lv - lv3).dropna()
    rows = []
    for t in range(252, len(spread), 5):
        rows.append(dict(date=spread.index[t],
                         old=calibrate_ou_ar1(spread.iloc[t - 60:t], bias_correct=False)["half_life"],
                         new=calibrate_ou_ar1(spread.iloc[t - 252:t])["half_life"]))
    hl = pd.DataFrame(rows).set_index("date")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(hl["old"].clip(upper=40), color=GREY, lw=0.9, label=f"Antes: ventana 60, MCO (mediana {hl['old'].median():.1f} ruedas)")
    ax.plot(hl["new"].clip(upper=40), color=AQUA, label=f"Ahora: ventana 252, sesgo corregido (mediana {hl['new'].median():.1f} ruedas)")
    ax.set_title("Half-life del spread ln(VIX) − ln(VIX3M)", loc="left")
    ax.set_ylabel("Ruedas (recortado en 40)")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.savefig(OUT / "fase1_ou_halflife.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6) Fracdiff: p-valores ADF/KPSS y correlación con el nivel según d (entrenamiento 2007-2014)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, sym in zip(axes, ["SPY", "NVDA"]):
        d_opt, tab, _ = find_optimal_d(np.log(raw[f"{sym}_Adj_Close"]).rename(sym), d_values=np.linspace(0.05, 1, 20),
                                       train_end="2014-12-31")
        ax.plot(tab.d, tab.adf_p, color=BLUE, marker="o", ms=3, label="p-valor ADF (< 0,05 = estacionaria)")
        ax.plot(tab.d, tab.kpss_p, color=ORANGE, marker="o", ms=3, label="p-valor KPSS (> 0,05 = estacionaria)")
        ax.plot(tab.d, tab.correlation, color=AQUA, label="Correlación con el log-precio")
        ax.axhline(0.05, color=INK2, lw=1, ls=":")
        ax.axhline(0.9, color=AQUA, lw=1, ls=":")
        ax.axvline(d_opt, color=INK2, lw=1.2, ls="--")
        ax.text(d_opt + 0.015, 0.72, f"d* = {d_opt:.2f}", color=INK2)
        ax.set_title(f"{sym} (2007–2014)", loc="left")
        ax.set_xlabel("orden de diferenciación d")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(OUT / "fase1_fracdiff_d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Gráficos de fase 1 guardados en", OUT)


if __name__ == "__main__":
    main()
