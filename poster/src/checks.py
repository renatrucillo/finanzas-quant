"""Chequeos de cordura del pipeline (también se usan en tests/)."""
import numpy as np
import pandas as pd

from poster.src.data import build_cycles
from poster.src.strategy import FLAT, Skew, Spec, daily_equity, simulate
from poster.src.vol_models import forecast_cycles


def check_no_lookahead(asset: dict, k: int = 100) -> dict:
    """Si se corrompen los precios desde t0 en adelante, los pronósticos en t0 no deben cambiar."""
    px, iv = asset["px"], asset["iv"]
    sub = build_cycles(px.index).iloc[: k + 1]
    f0 = forecast_cycles(px, iv, sub, verbose=False).iloc[-1]
    p0 = int(sub.iloc[-1].pos0)
    px2 = px.copy()
    junk = np.exp(np.random.default_rng(1).normal(0, 0.05, len(px2) - p0))
    for c in ["Open", "High", "Low", "Close"]:
        px2.iloc[p0:, px2.columns.get_loc(c)] = px2[c].iloc[p0:].values * junk
    f1 = forecast_cycles(px2, iv, sub, verbose=False).iloc[-1]
    cols = ["Rolling21", "EWMA", "GJR-GARCH", "HAR-RV", "Kalman", "Ensamble", "IV_sig"]
    return {c: float(abs(f0[c] - f1[c])) for c in cols}


def check_static_equals_z0(asset: dict, fc: pd.DataFrame, skew: Skew = FLAT) -> float:
    """El CC estático ATM debe coincidir con 'vender siempre con z = 0'."""
    cyc = build_cycles(asset["px"].index)
    a = simulate(asset, fc, Spec("a", always=True, timing=False, model_strike=False), cyc, skew)
    b = simulate(asset, fc, Spec("b", timing=False, model_strike=True, z=0.0), cyc, skew)
    return float(np.abs(a.ret - b.ret).max())


def check_daily_matches_cycles(asset: dict, sim: pd.DataFrame, skew: Skew = FLAT) -> float:
    """El valor diario al vencimiento de cada ciclo debe coincidir con la equity por ciclos."""
    d = daily_equity(asset, sim, skew)
    eq = (1 + sim["ret"]).cumprod()
    return float(np.abs(d.reindex(sim["texp"]).values - eq.values).max())
