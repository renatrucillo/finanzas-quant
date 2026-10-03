"""Chequeos de cordura del pipeline (se ejecutan también desde el notebook)."""
import numpy as np
import pandas as pd

from poster.src.data import build_cycles, load_panel, _yf
from poster.src.vol_models import forecast_cycles
from poster.src.strategy import Spec, simulate, metrics, equity


def check_no_lookahead(asset: dict, k: int = 100) -> dict:
    """Si se corrompen los precios posteriores a t0-1, los pronósticos en t0 no deben cambiar."""
    px, iv = asset["px"], asset["iv"]
    cyc = build_cycles(px.index)
    sub = cyc.iloc[: k + 1]
    f0 = forecast_cycles(px, iv, sub, verbose=False).iloc[-1]
    p0 = int(sub.iloc[-1].pos0)
    px2 = px.copy()
    rng = np.random.default_rng(1)
    junk = np.exp(rng.normal(0, 0.05, len(px2) - p0))
    for c in ["Open", "High", "Low", "Close"]:
        px2.iloc[p0:, px2.columns.get_loc(c)] = px2[c].iloc[p0:].values * junk
    f1 = forecast_cycles(px2, iv, sub, verbose=False).iloc[-1]
    cols = ["Rolling21", "EWMA", "GJR-GARCH", "HAR-RV", "Kalman", "Ensamble", "IV_sig"]
    return {c: float(abs(f0[c] - f1[c])) for c in cols}


def check_static_equals_z0(asset: dict, fc: pd.DataFrame) -> float:
    cyc = build_cycles(asset["px"].index)
    a = simulate(asset, fc, Spec("a", always=True, timing=False, model_strike=False), cyc)
    b = simulate(asset, fc, Spec("b", timing=False, model_strike=True, z=0.0), cyc)
    return float(np.abs(a.ret - b.ret).max())


def check_vs_buywrite(asset: dict, fc: pd.DataFrame) -> pd.DataFrame:
    """CC estático ATM simulado (SPY) vs ETF BuyWrite real PBP (Invesco S&P 500 BuyWrite, desde dic-2007)."""
    pbp = _yf("PBP")["Adj Close"].dropna()
    cyc = build_cycles(asset["px"].index)
    a = simulate(asset, fc, Spec("est", always=True, timing=False, model_strike=False), cyc)
    t = [fc.index[0]] + list(a["texp"])
    pb = pbp.reindex(t, method="ffill")
    r_pbp = pb.pct_change().dropna().values
    out = pd.DataFrame({"sim": a["ret"].values, "PBP": r_pbp}, index=a.index)
    out.attrs["corr"] = float(out.corr().iloc[0, 1])
    out.attrs["cagr_sim"] = float((1 + out.sim).prod() ** (1 / ((a["texp"].iloc[-1] - a.index[0]).days / 365.25)) - 1)
    out.attrs["cagr_pbp"] = float((1 + out.PBP).prod() ** (1 / ((a["texp"].iloc[-1] - a.index[0]).days / 365.25)) - 1)
    return out


if __name__ == "__main__":
    A = load_panel()
    a = A["SPY"]
    fc = pd.read_parquet("poster/data/forecasts_SPY.parquet")
    print("no look-ahead (|Δ| tras corromper el futuro):", check_no_lookahead(a))
    print("estático == siempre+z0 (max |Δret|):", check_static_equals_z0(a, fc))
    bw = check_vs_buywrite(a, fc)
    print("sim vs PBP: corr mensual %.3f | CAGR sim %.2f%% vs PBP %.2f%%" % (bw.attrs["corr"], 100 * bw.attrs["cagr_sim"], 100 * bw.attrs["cagr_pbp"]))
    print("RV media %.3f < IV media %.3f:" % (fc.RV.mean(), fc.IV_sig.mean()), fc.RV.mean() < fc.IV_sig.mean())
