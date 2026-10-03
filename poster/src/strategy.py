"""Backtest de Covered Calls por ciclos mensuales: B&H, CC estático ATM y CC con modelo de volatilidad.

Reglas (congeladas antes de mirar resultados):
  - Decisión con info hasta t0-1 (σ̂ del modelo vs IV_sig); ejecución al cierre de t0.
  - Timing:  vender si σ̂ < IV_sig * (1 - m)            (m = 0 por defecto)
  - Strike:  K = S0 * exp(z * σ̂ * sqrt(h/252))         (z = 0.5 ⇒ P_modelo(ejercicio) ≈ 31%)
  - Precio:  Black-Scholes con IV(K) = IV_exec * (a + b * ln(K/S0)), calibrado con calls SPY de IBKR.
  - Costo:   se cobra (1 - cost) * prima.
  - P&L:     (S_T + D - max(S_T-K,0)*1_venta + prima*(1-cost)*e^{rT}*1_venta)/S0 - 1
"""
from __future__ import annotations

import sys  # noqa
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import norm


def black_scholes_call_price(S, K, T, r, sigma, q=0.0):
    """Call europea Black-Scholes-Merton con dividendo continuo q (Hull, cap. 15)."""
    S, K, T, sigma = (np.maximum(np.asarray(x, dtype=float), 1e-6) for x in (S, K, T, sigma))
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    return np.maximum(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0.0)

# Calibración IBKR (SPY, calls 20-70 DTE, volumen>=10, jun-sep 2026): IV(K)/VIX ≈ A + B*ln(K/S)
SKEW_A, SKEW_B = 0.88, -4.33


@dataclass
class Spec:
    name: str
    sigma: str = "Ensamble"      # columna del pronóstico usada
    timing: bool = True          # False => vende siempre
    model_strike: bool = True    # False => ATM
    z: float = 0.5
    m: float = 0.0
    cost: float = 0.02
    skew: bool = True            # False => IV plana = IV_exec sin ajuste
    always: bool = False         # CC estático ATM


def iv_at_strike(iv_exec: float, mny: float, skew: bool) -> float:
    if not skew:
        return iv_exec
    return iv_exec * float(np.clip(SKEW_A + SKEW_B * mny, 0.5, 1.1))


def simulate(asset: dict, fc: pd.DataFrame, spec: Spec | None, cycles: pd.DataFrame) -> pd.DataFrame:
    """Retornos por ciclo. spec=None => Buy & Hold (retorno total)."""
    px, div, rf = asset["px"]["Close"], asset["div"], asset["rf"]
    cy = cycles.set_index("t0").loc[fc.index]
    rows = []
    for t0, c in cy.iterrows():
        S0, ST = px.iloc[c.pos0], px.iloc[c.posT]
        D = float(div[(div.index > t0) & (div.index <= c.texp)].sum()) if len(div) else 0.0
        h = int(c.h)
        Tyr = h / 252.0
        r = float(rf.iloc[c.pos0])
        base = (ST + D) / S0 - 1.0
        rf_cycle = np.exp(r * Tyr) - 1.0
        row = dict(t0=t0, texp=c.texp, h=h, ret_bh=base, rf=rf_cycle, sell=0, K=np.nan, prem=0.0)
        if spec is None:
            row["ret"] = base
        else:
            f = fc.loc[t0]
            sig = float(f[spec.sigma])
            sell = True if (spec.always or not spec.timing) else bool(sig < f["IV_sig"] * (1 - spec.m))
            if sell:
                K = S0 * np.exp(spec.z * sig * np.sqrt(Tyr)) if spec.model_strike else S0
                mny = float(np.log(K / S0))
                q = float(div[(div.index > t0 - pd.Timedelta(days=365)) & (div.index <= t0)].sum() / S0) if len(div) else 0.0
                iv = iv_at_strike(float(f["IV_exec"]), mny, spec.skew)
                C = float(black_scholes_call_price(S0, K, Tyr, r, iv, q))
                prem = C * (1 - spec.cost)
                row.update(sell=1, K=K, prem=prem / S0)
                row["ret"] = (ST + D - max(ST - K, 0.0) + prem * np.exp(r * Tyr)) / S0 - 1.0
            else:
                row["ret"] = base
        rows.append(row)
    return pd.DataFrame(rows).set_index("t0")


# --------------------------------------------------------------- métricas
def metrics(df: pd.DataFrame) -> dict:
    r = df["ret"]
    yrs = (df["texp"].iloc[-1] - df.index[0]).days / 365.25
    eq = (1 + r).cumprod()
    n_py = len(r) / yrs
    ex = r - df["rf"]
    dd = eq / eq.cummax() - 1
    return dict(
        retorno_total=eq.iloc[-1] - 1,
        CAGR=eq.iloc[-1] ** (1 / yrs) - 1,
        vol=r.std() * np.sqrt(n_py),
        sharpe=ex.mean() / r.std() * np.sqrt(n_py),
        maxdd=dd.min(),
        pct_vendido=df["sell"].mean(),
        n=len(r),
    )


def equity(df: pd.DataFrame) -> pd.Series:
    return (1 + df["ret"]).cumprod()


def stationary_bootstrap_diff(d: np.ndarray, n_py: float, block: int = 3, B: int = 5000, seed: int = 0) -> dict:
    """IC 95% de la media (anualizada) de la diferencia de retornos por ciclo (bootstrap estacionario)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    p = 1.0 / block
    means = np.empty(B)
    for b in range(B):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        for t in range(1, n):
            idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
        means[b] = d[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5]) * n_py
    return dict(media_anual=d.mean() * n_py, lo=lo, hi=hi, p_neg=float((means <= 0).mean()))


def psr(r: np.ndarray, sr_bench: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado) con SR por período."""
    from scipy.stats import norm, skew, kurtosis

    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    den = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(norm.cdf((sr - sr_bench) * np.sqrt(n - 1) / den))


# --------------------------------------------------------------- batería de estrategias
def standard_specs(sigma: str = "Ensamble", z: float = 0.5, m: float = 0.0, cost: float = 0.02, skew: bool = True) -> dict:
    kw = dict(z=z, m=m, cost=cost, skew=skew)
    return {
        "CC estático ATM": Spec("CC estático ATM", always=True, timing=False, model_strike=False, **kw),
        "Siempre + strike modelo": Spec("Siempre + strike modelo", sigma=sigma, timing=False, model_strike=True, **kw),
        "Timing modelo + ATM": Spec("Timing modelo + ATM", sigma=sigma, timing=True, model_strike=False, **kw),
        "CC modelo": Spec("CC modelo", sigma=sigma, timing=True, model_strike=True, **kw),
    }


def run_all(asset: dict, fc: pd.DataFrame, cycles: pd.DataFrame, **kw) -> dict[str, pd.DataFrame]:
    out = {"Buy & Hold": simulate(asset, fc, None, cycles)}
    for k, s in standard_specs(**kw).items():
        out[k] = simulate(asset, fc, s, cycles)
    return out


def summary_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({k: metrics(v) for k, v in results.items()}).T
