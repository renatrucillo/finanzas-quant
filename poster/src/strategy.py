"""Backtest de Covered Calls por ciclos mensuales: B&H, CC estático y CC condicionado por un modelo de volatilidad.

Reglas (convenciones explícitas):
  - Decisión con info hasta t0-1; ejecución al cierre de t0 (vencimiento anterior); liquidación al cierre de texp.
  - Tiempo: las vol. realizadas/pronosticadas se anualizan en base 252 ruedas (h ruedas del ciclo); la IV tipo
    VIX se anualiza en base 365 días corridos (cal días del ciclo). Toda comparación se hace en desvío POR CICLO:
        sd_modelo = σ̂ * sqrt(h/252)        sd_IV = IV * sqrt(cal/365)
  - Strike:  K = S0 * exp(z * sd), con sd del modelo (o de la IV para el benchmark sin modelo). z = 0 => ATM.
  - Precio:  Black-Scholes con T = cal/365, dividendo continuo q (últimos 12 meses) e
             IV(K) = IV_índice * skew(x),  x = ln(K/S) / (IV_índice * sqrt(T))  (moneyness estandarizada).
  - Señal:   vender si sd_modelo < sd de la IV DE LA CALL QUE SE VA A VENDER (IV(K) observada en t0-1).
             La versión original comparaba contra el VIX entero ("señal VIX"), que incluye el skew de los puts
             y sobreestima cuánto se cobra por la call; se conserva solo como comparación.
  - Costo:   se cobra (1 - cost) * prima.
  - Ejercicio anticipado (opciones americanas): la rueda previa a cada ex-dividendo, si la call está ITM y
    S - K > C_europea(S - D, ...), el comprador ejerce: se entregan las acciones a K y se pierde el dividendo.
  - P&L:     (S_T + D - max(S_T-K,0) + prima*(1-cost)*e^{rT}) / S0 - 1, o la versión con ejercicio anticipado.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import kurtosis, norm, skew as sample_skew

from poster.src.data import trailing_div_yield

TRADING, CALENDAR = 252.0, 365.0


def black_scholes_call_price(S, K, T, r, sigma, q=0.0):
    """Call europea Black-Scholes-Merton con dividendo continuo q (Hull, cap. 15)."""
    S, K, T, sigma = (np.maximum(np.asarray(x, dtype=float), 1e-12) for x in (S, K, T, sigma))
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    return np.maximum(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0.0)


def implied_vol(price, S, K, T, r, q=0.0) -> float:
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if not price > intrinsic + 1e-8:
        return np.nan
    f = lambda s: float(black_scholes_call_price(S, K, T, r, s, q)) - price
    try:
        return brentq(f, 1e-3, 3.0)
    except ValueError:
        return np.nan


# --------------------------------------------------------------- skew
@dataclass
class Skew:
    """IV(K) / IV_índice = a + b * x,  x = ln(K/S) / (IV_índice * sqrt(T)), con x acotado al rango calibrado.

    Medir la moneyness en desvíos (x) y no en ln(K/S) hace que la misma curva valga para distintos
    plazos y regímenes de volatilidad; fuera del rango calibrado se extrapola plano (no lineal).
    """
    a: float = 1.0
    b: float = 0.0
    x_lo: float = 0.0
    x_hi: float = 0.0
    n: int = 0
    source: str = "plana"

    def ratio(self, mny: float, iv_index: float, T: float) -> float:
        if self.b == 0.0:
            return self.a
        x = float(np.clip(mny / (iv_index * np.sqrt(T)), self.x_lo, self.x_hi))
        return self.a + self.b * x


FLAT = Skew()


def calibrate_skew(ib: pd.DataFrame, spy: dict, symbols=("SPX",), dte=(20, 45), x_fit=(-0.25, 1.25)) -> tuple[Skew, pd.DataFrame]:
    """Calibra la curva IV(K)/VIX con calls de IBKR (jun-sep 2026).

    - SPX por defecto: opciones europeas (BS exacto) y es el subyacente del VIX.
    - Solo trades de la sesión de la tarde (barra 16:00 UTC), volumen >= 10 y 20-45 días al vencimiento.
    - T en días corridos / 365 (misma convención que el backtest); q = rendimiento por dividendo de SPY 12m.
    """
    s = ib[ib.symbol.isin(symbols)].copy()
    s["date"] = pd.to_datetime(s["date"])
    s["expiration"] = pd.to_datetime(s["expiration"].astype(str))
    s = s[s.days_to_expiration.between(*dte) & (s.volume >= 10)]
    s = s[pd.to_datetime(s.last_bar_utc).dt.hour >= 16]
    s["T"] = (s.expiration - s.date).dt.days / CALENDAR
    s["r"] = spy["rf"].reindex(s.date).values
    s["vix"] = spy["iv"].reindex(s.date).values
    spy_close = spy["px"]["Close"]
    s["q"] = [trailing_div_yield(spy["div"], spy_close.asof(d), d) for d in s.date]
    s = s.dropna(subset=["r", "vix"])
    s["iv"] = [implied_vol(p, S, K, T, r, q) for p, S, K, T, r, q in
               zip(s.close, s.underlying_close, s.strike, s["T"], s.r, s.q)]
    s = s.dropna(subset=["iv"])
    s["m"] = np.log(s.strike / s.underlying_close)
    s["x"] = s.m / (s.vix * np.sqrt(s["T"]))
    s["ratio"] = s.iv / s.vix
    fit = s[s.x.between(*x_fit)]
    b, a = np.polyfit(fit.x, fit.ratio, 1)
    lo, hi = np.percentile(fit.x, [2, 98])
    return Skew(a=float(a), b=float(b), x_lo=float(lo), x_hi=float(hi), n=len(fit), source="+".join(symbols)), s


# --------------------------------------------------------------- estrategia
@dataclass
class Spec:
    name: str
    sigma: str = "Ensamble"      # columna del pronóstico usada para el strike y la señal
    timing: bool = True          # False => vende siempre
    model_strike: bool = True    # False => ATM
    z: float = 0.5
    m: float = 0.0               # margen exigido en la señal
    cost: float = 0.02
    skew: bool = True            # False => IV plana = IV del índice
    always: bool = False
    signal: str = "strike"       # "strike" (IV de la call) | "index" (VIX entero, versión original)
    early_exercise: bool = True  # False => opción europea (como el índice BXM sobre SPX)


def _cycle_sd(col: str, f: pd.Series, c) -> float:
    """Desvío del ciclo: base 252 para pronósticos, base 365 para columnas de IV."""
    if col.startswith("IV"):
        return float(f[col]) * np.sqrt(c.cal / CALENDAR)
    return float(f[col]) * np.sqrt(c.h / TRADING)


def _early_exercise(px, ivs, divs, c, K, r, skew, use_skew):
    """Rueda (posición) en la que el comprador ejerce antes de un ex-dividendo, o None."""
    idx = px.index
    for ex_date, D in divs.items():
        pe = idx.searchsorted(ex_date) - 1          # última rueda antes del ex-dividendo
        if pe <= c.pos0:
            continue
        S = float(px.iloc[pe])
        if S <= K:
            continue
        tau = (c.texp - idx[pe]).days / CALENDAR
        iv = float(ivs.iloc[pe])
        iv_k = iv * (skew.ratio(np.log(K / S), iv, tau) if use_skew else 1.0)
        hold = float(black_scholes_call_price(S - D, K, tau, r, iv_k, 0.0))
        if S - K > hold:
            return pe, ex_date
    return None


def simulate(asset: dict, fc: pd.DataFrame, spec: Spec | None, cycles: pd.DataFrame, skew: Skew = FLAT) -> pd.DataFrame:
    """Retornos por ciclo. spec=None => Buy & Hold (retorno total)."""
    px, div, rf, ivs = asset["px"]["Close"], asset["div"], asset["rf"], asset["iv"]
    cy = cycles.set_index("t0").loc[fc.index]
    empty = pd.Series(dtype=float)
    rows = []
    for t0, c in cy.iterrows():
        S0, ST = float(px.iloc[c.pos0]), float(px.iloc[c.posT])
        divs = div[(div.index > t0) & (div.index <= c.texp)].sort_index() if len(div) else empty
        D = float(divs.sum())
        T = c.cal / CALENDAR
        r = float(rf.iloc[c.pos0])
        base = (ST + D) / S0 - 1.0
        row = dict(t0=t0, texp=c.texp, pos0=int(c.pos0), posT=int(c.posT), h=int(c.h), cal=int(c.cal), r=r,
                   ret_bh=base, rf=np.exp(r * T) - 1.0, sell=0, K=np.nan, prem=0.0, prem_abs=0.0, q=0.0,
                   early=0, ex_pos=-1, D_before=0.0, ret=base)
        if spec is not None:
            f = fc.loc[t0]
            mny = spec.z * _cycle_sd(spec.sigma, f, c) if spec.model_strike else 0.0
            if spec.always or not spec.timing:
                sell = True
            else:
                sd_model = _cycle_sd(spec.sigma, f, c)
                iv_sig = float(f["IV_sig"])
                if spec.signal == "strike" and spec.skew:
                    iv_sig *= skew.ratio(mny, iv_sig, T)
                sell = sd_model < iv_sig * np.sqrt(T) * (1 - spec.m)
            if sell:
                K = S0 * np.exp(mny)
                iv_exec = float(f["IV_exec"])
                iv = iv_exec * (skew.ratio(mny, iv_exec, T) if spec.skew else 1.0)
                q = trailing_div_yield(div, S0, t0)
                prem = float(black_scholes_call_price(S0, K, T, r, iv, q)) * (1 - spec.cost)
                row.update(sell=1, K=K, prem=prem / S0, prem_abs=prem, q=q)
                ex = _early_exercise(px, ivs, divs, c, K, r, skew, spec.skew) if (spec.early_exercise and len(divs)) else None
                if ex is not None:
                    pe, ex_date = ex
                    D_before = float(divs[divs.index < ex_date].sum())
                    tau = (c.texp - px.index[pe]).days / CALENDAR
                    row.update(early=1, ex_pos=int(pe), D_before=D_before,
                               ret=(K * np.exp(r * tau) + D_before + prem * np.exp(r * T)) / S0 - 1.0)
                else:
                    row["ret"] = (ST + D - max(ST - K, 0.0) + prem * np.exp(r * T)) / S0 - 1.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("t0")


def daily_equity(asset: dict, sim: pd.DataFrame, skew: Skew = FLAT, use_skew: bool = True) -> pd.Series:
    """Valor diario de la estrategia (marcando la call a mercado con BS + skew) para medir el drawdown real."""
    px, div, ivs = asset["px"]["Close"], asset["div"], asset["iv"]
    idx = px.index
    eq, dates, vals = 1.0, [sim.index[0]], [1.0]
    for t0, row in sim.iterrows():
        S0, r = float(px.iloc[row.pos0]), row.r
        cyc_div = div[(div.index > t0) & (div.index <= row.texp)] if len(div) else pd.Series(dtype=float)
        for p in range(row.pos0 + 1, row.posT + 1):
            d, S = idx[p], float(px.iloc[p])
            Dcum = float(cyc_div[cyc_div.index <= d].sum())
            if not row.sell:
                V = S + Dcum
            elif row.early and p >= row.ex_pos:
                V = row.K * np.exp(r * (d - idx[row.ex_pos]).days / CALENDAR) + row.D_before \
                    + row.prem_abs * np.exp(r * (d - t0).days / CALENDAR)
            else:
                tau = (row.texp - d).days / CALENDAR
                if p == row.posT or tau <= 0:
                    C = max(S - row.K, 0.0)
                else:
                    iv = float(ivs.iloc[p])
                    iv_k = iv * (skew.ratio(np.log(row.K / S), iv, tau) if use_skew else 1.0)
                    C = float(black_scholes_call_price(S, row.K, tau, r, iv_k, row.q))
                V = S + Dcum - C + row.prem_abs * np.exp(r * (d - t0).days / CALENDAR)
            dates.append(d)
            vals.append(eq * V / S0)
        eq *= 1.0 + row.ret
    return pd.Series(vals, index=pd.DatetimeIndex(dates))


# --------------------------------------------------------------- métricas
def _per_year(df: pd.DataFrame) -> tuple[float, float]:
    yrs = (df["texp"].iloc[-1] - df.index[0]).days / 365.25
    return yrs, len(df) / yrs


def metrics(df: pd.DataFrame, daily: pd.Series | None = None) -> dict:
    r = df["ret"]
    yrs, n_py = _per_year(df)
    eq = (1 + r).cumprod()
    ex = r - df["rf"]
    downside = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    out = dict(
        retorno_total=eq.iloc[-1] - 1,
        CAGR=eq.iloc[-1] ** (1 / yrs) - 1,
        vol=r.std() * np.sqrt(n_py),
        sharpe=ex.mean() / ex.std() * np.sqrt(n_py),
        sortino=ex.mean() / downside * np.sqrt(n_py),
        maxdd_mensual=(eq / eq.cummax() - 1).min(),
        pct_vendido=df["sell"].mean(),
        aporte_opcion_anual=(r - df["ret_bh"]).mean() * n_py,
        ejercicios_anticipados=int(df["early"].sum()),
        n=len(r),
    )
    out["maxdd"] = (daily / daily.cummax() - 1).min() if daily is not None else out["maxdd_mensual"]
    return out


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


def psr(ex: np.ndarray, sr_bench: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012) sobre EXCESOS, SR por período."""
    n = len(ex)
    sr = ex.mean() / ex.std(ddof=1)
    g3, g4 = sample_skew(ex), kurtosis(ex, fisher=False)
    den = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(norm.cdf((sr - sr_bench) * np.sqrt(n - 1) / den))


def deflated_sharpe(ex: np.ndarray, trial_srs: np.ndarray) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014): PSR contra el máximo SR esperado entre N pruebas."""
    N = len(trial_srs)
    g = 0.5772156649
    sr0 = np.std(trial_srs, ddof=1) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e)))
    return dict(dsr=psr(ex, sr0), sr0_periodo=float(sr0), n_pruebas=N)


# --------------------------------------------------------------- batería de estrategias
def standard_specs(sigma: str = "Ensamble", z: float = 0.5, m: float = 0.0, cost: float = 0.02, use_skew: bool = True) -> dict:
    kw = dict(z=z, m=m, cost=cost, skew=use_skew)
    return {
        "CC estático ATM": Spec("CC estático ATM", always=True, timing=False, model_strike=False, **kw),
        "CC siempre OTM (IV)": Spec("CC siempre OTM (IV)", sigma="IV_sig", always=True, timing=False, model_strike=True, **kw),
        "Siempre + strike modelo": Spec("Siempre + strike modelo", sigma=sigma, timing=False, model_strike=True, **kw),
        "Timing modelo + ATM": Spec("Timing modelo + ATM", sigma=sigma, timing=True, model_strike=False, **kw),
        "CC modelo (señal VIX)": Spec("CC modelo (señal VIX)", sigma=sigma, signal="index", **kw),
        "CC modelo": Spec("CC modelo", sigma=sigma, **kw),
    }


def run_all(asset: dict, fc: pd.DataFrame, cycles: pd.DataFrame, skew: Skew = FLAT, **kw) -> dict[str, pd.DataFrame]:
    out = {"Buy & Hold": simulate(asset, fc, None, cycles)}
    for k, s in standard_specs(**kw).items():
        out[k] = simulate(asset, fc, s, cycles, skew)
    return out
