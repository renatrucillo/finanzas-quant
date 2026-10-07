"""Carga y alineación de datos del backtest Covered Call vs. volatilidad.

Fuentes:
  - SPY y VIX: data/raw/datos_finanzas.xlsx (dato del profe).
  - QQQ, ^VXN, ^IRX, XYLD/PBP (chequeo): yfinance, cacheado en poster/data/.
  - IBKR: data/raw/ibkr_calls_daily.parquet (calls SPY jun-sep 2026).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "poster" / "data"
CACHE.mkdir(parents=True, exist_ok=True)

START, END = "2007-01-01", "2026-10-01"


def _parse_sheet(path: Path, sheet: str) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (OHLC, dividendos). Las filas de dividendo vienen mezcladas con las de precio."""
    raw = pd.read_excel(path, sheet_name=sheet)
    raw.columns = [str(c).strip() for c in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"])
    is_div = raw["Open"].astype(str).str.contains("Dividend", case=False, na=False)
    div = pd.Series(dtype=float, name="div")
    if is_div.any():
        d = raw[is_div]
        div = pd.Series(
            d["Open"].astype(str).str.extract(r"([\d.]+)")[0].astype(float).values,
            index=d["Date"].values,
            name="div",
        )
        div = div.groupby(level=0).sum()
    px = raw[~is_div].copy()
    for c in ["Open", "High", "Low", "Close", "Adj Close"]:
        px[c] = pd.to_numeric(px[c], errors="coerce")
    px = px.dropna(subset=["Close"]).set_index("Date").sort_index()
    return px[["Open", "High", "Low", "Close", "Adj Close"]], div


def _yf(ticker: str, actions: bool = False) -> pd.DataFrame:
    f = CACHE / f"yf_{ticker.replace('^', '')}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    import yfinance as yf  # solo hace falta si no hay cache

    d = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False, actions=actions)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.index = pd.to_datetime(d.index).tz_localize(None)
    if len(d):
        d.to_parquet(f)
    return d


def load_panel() -> dict:
    """Panel por activo: precios OHLC, dividendos, IV (índice), tasa libre de riesgo."""
    xlsx = RAW / "datos_finanzas.xlsx"
    spy, spy_div = _parse_sheet(xlsx, "SPY ")
    vix_px, _ = _parse_sheet(xlsx, "VIX")

    q = _yf("QQQ", actions=True)
    qqq = q[["Open", "High", "Low", "Close", "Adj Close"]].dropna()
    qqq_div = q["Dividends"][q["Dividends"] > 0].rename("div") if "Dividends" in q else pd.Series(dtype=float)
    vxn = _yf("^VXN")["Close"].dropna()
    irx = discount_to_continuous(_yf("^IRX")["Close"].dropna() / 100.0)

    vix = vix_px["Close"]
    idx = spy.index.intersection(qqq.index).intersection(vix.index).intersection(vxn.index)
    idx = idx[(idx >= START) & (idx < END)]

    rf = irx.reindex(idx).ffill()  # sin bfill: no se rellena el pasado con datos futuros
    if rf.isna().any():
        raise ValueError("Falta la tasa libre de riesgo al inicio de la muestra")
    assets = {
        "SPY": dict(px=spy.reindex(idx), div=spy_div, iv=(vix.reindex(idx) / 100.0).rename("IV")),
        "QQQ": dict(px=qqq.reindex(idx), div=qqq_div, iv=(vxn.reindex(idx) / 100.0).rename("IV")),
    }
    for a in assets.values():
        a["rf"] = rf
    return assets


def discount_to_continuous(d: pd.Series, days: int = 91) -> pd.Series:
    """^IRX cotiza la letra a 13 semanas como tasa de DESCUENTO (base 360). Se convierte a tasa
    continua anual (base 365): precio = 1 - d * days/360,  r = -ln(precio) / (days/365)."""
    return -np.log(1.0 - d * days / 360.0) / (days / 365.0)


def trailing_div_yield(div: pd.Series, spot: float, date) -> float:
    """Rendimiento por dividendo de los últimos 12 meses (información disponible en `date`)."""
    if not len(div):
        return 0.0
    date = pd.Timestamp(date)
    return float(div[(div.index > date - pd.Timedelta(days=365)) & (div.index <= date)].sum() / spot)


def load_ibkr() -> pd.DataFrame:
    return pd.read_parquet(RAW / "ibkr_calls_daily.parquet")


def validate(assets: dict) -> pd.DataFrame:
    rows = []
    for k, a in assets.items():
        px, iv = a["px"], a["iv"]
        rows.append(
            dict(
                activo=k,
                desde=px.index.min().date(),
                hasta=px.index.max().date(),
                n=len(px),
                nan_px=int(px.isna().sum().sum()),
                nan_iv=int(iv.isna().sum()),
                n_div=len(a["div"]),
                div_ult12m=float(a["div"].loc[a["div"].index > px.index.max() - pd.Timedelta(days=365)].sum()),
                iv_medio=float(iv.mean()),
            )
        )
    return pd.DataFrame(rows)


def build_cycles(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Ciclos mensuales entre vencimientos estándar (3er viernes; si es feriado, la rueda anterior).

    t0 = vencimiento anterior (se vende al cierre), texp = vencimiento siguiente (liquidación).
    La señal usa información hasta la rueda previa a t0 (pos0 - 1).
    h = ruedas hábiles del ciclo (para volatilidades realizadas, base 252);
    cal = días corridos (para Black-Scholes con IV tipo VIX, que se anualiza en base 365).
    """
    months = pd.period_range(index[0], index[-1], freq="M")
    exps = []
    for m in months:
        fridays = pd.date_range(m.start_time, m.end_time, freq="W-FRI")
        if len(fridays) < 3:
            continue
        d = fridays[2]
        pos = index.searchsorted(d, side="right") - 1  # última rueda <= d
        if pos >= 0 and index[pos].to_period("M") == m:
            exps.append(index[pos])
    exps = pd.DatetimeIndex(exps)
    rows = []
    for a, b in zip(exps[:-1], exps[1:]):
        p0, p1 = index.get_loc(a), index.get_loc(b)
        rows.append(dict(t0=a, texp=b, pos0=p0, posT=p1, h=p1 - p0, cal=(b - a).days))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    A = load_panel()
    print(validate(A).to_string())
    print(load_ibkr().shape)
