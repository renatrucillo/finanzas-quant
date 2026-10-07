"""Tests de fórmulas y de ausencia de lookahead.  Correr con:  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poster.src.data import discount_to_continuous  # noqa: E402
from poster.src.strategy import Skew, black_scholes_call_price, implied_vol  # noqa: E402
from src.fracdiff import fractional_diff  # noqa: E402
from src.garch_model import rolling_gjr_garch_forecast  # noqa: E402
from src.kalman_filter import kalman_beta_1d, kalman_capm_2d  # noqa: E402
from src.ou_process import calibrate_ou_ar1  # noqa: E402
from src.volatility_estimators import garman_klass_volatility, parkinson_volatility, yang_zhang_volatility  # noqa: E402

HAS_DATA = (ROOT / "data" / "raw" / "datos_finanzas.xlsx").exists() and (ROOT / "poster" / "data" / "yf_QQQ.parquet").exists()


# ------------------------------------------------------------------ pricing
def test_black_scholes_hull_example():
    # Hull, ejemplo 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5 -> c = 4.76
    assert float(black_scholes_call_price(42, 40, 0.5, 0.10, 0.20)) == pytest.approx(4.76, abs=0.005)


def test_implied_vol_roundtrip():
    c = float(black_scholes_call_price(100, 105, 30 / 365, 0.04, 0.18, 0.012))
    assert implied_vol(c, 100, 105, 30 / 365, 0.04, 0.012) == pytest.approx(0.18, abs=1e-6)


def test_skew_is_flat_outside_calibrated_range():
    sk = Skew(a=0.83, b=-0.18, x_lo=-0.2, x_hi=1.0)
    T, iv = 0.08, 0.2
    at = lambda x: sk.ratio(x * iv * np.sqrt(T), iv, T)
    assert at(0.0) == pytest.approx(0.83)
    assert at(3.0) == pytest.approx(at(1.0))
    assert at(-2.0) == pytest.approx(at(-0.2))


def test_discount_rate_conversion():
    # una letra a 91 días con descuento 5% rinde más que 5% en tasa continua
    r = float(discount_to_continuous(pd.Series([0.05])).iloc[0])
    assert r == pytest.approx(-np.log(1 - 0.05 * 91 / 360) / (91 / 365))
    assert 0.0505 < r < 0.052


# ------------------------------------------------------------------ estimadores
def _ohlc(n=300, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = c * np.exp(rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return [pd.Series(x, index=idx) for x in (o, h, l, c)]


def test_range_estimators_positive():
    o, h, l, c = _ohlc()
    for v in (parkinson_volatility(h, l), garman_klass_volatility(o, h, l, c), yang_zhang_volatility(o, h, l, c)):
        assert (v.dropna() > 0).all()


def test_ou_recovers_parameters():
    rng = np.random.default_rng(0)
    b, theta, n = 0.95, -0.1, 20000
    x = np.empty(n)
    x[0] = theta
    for t in range(1, n):
        x[t] = theta + b * (x[t - 1] - theta) + rng.normal(0, 0.02)
    p = calibrate_ou_ar1(pd.Series(x))
    assert p["kappa"] == pytest.approx(-np.log(b), rel=0.1)
    assert p["theta"] == pytest.approx(theta, abs=0.01)


def test_ou_bias_correction_reduces_bias():
    rng = np.random.default_rng(1)
    b, n = 0.95, 120
    raw, adj = [], []
    for _ in range(300):
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = b * x[t - 1] + rng.normal()
        raw.append(calibrate_ou_ar1(pd.Series(x), bias_correct=False)["b"])
        adj.append(calibrate_ou_ar1(pd.Series(x), bias_correct=True)["b"])
    assert abs(np.mean(adj) - b) < abs(np.mean(raw) - b)


def test_fracdiff_d1_is_first_difference():
    s = pd.Series(np.log(np.arange(1, 200, dtype=float)))
    fd = fractional_diff(s, 1.0).dropna()
    assert np.allclose(fd.values, s.diff().dropna().values)


# ------------------------------------------------------------------ causalidad
def test_garch_forecast_is_causal_and_updates_daily():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.standard_t(5, 700) * 0.01, index=pd.bdate_range("2015-01-01", periods=700))
    f0 = rolling_gjr_garch_forecast(r, min_obs=300, refit_step=50)
    r2 = r.copy()
    r2.iloc[500:] *= 5
    f1 = rolling_gjr_garch_forecast(r2, min_obs=300, refit_step=50)
    assert np.allclose(f0.iloc[:501].dropna(), f1.iloc[:501].dropna())   # el pronóstico de t usa hasta t-1
    window = f0.iloc[300:350]
    assert window.nunique() > 40                                          # se actualiza entre reajustes


def test_kalman_is_causal():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=600)
    x = pd.Series(rng.normal(0, 0.01, 600), index=idx)
    y = 1.2 * x + pd.Series(rng.normal(0, 0.005, 600), index=idx)
    b0, _ = kalman_beta_1d(y, x)
    y2 = y.copy()
    y2.iloc[400:] += 1.0
    b1, _ = kalman_beta_1d(y2, x)
    assert np.allclose(b0.iloc[:400], b1.iloc[:400])
    s0, _ = kalman_capm_2d(y, x)
    s1, _ = kalman_capm_2d(y2, x)
    assert np.allclose(s0.iloc[:400], s1.iloc[:400])


# ------------------------------------------------------------------ backtest (requiere datos)
@pytest.fixture(scope="module")
def spy():
    from poster.src.data import load_panel

    return load_panel()["SPY"]


@pytest.fixture(scope="module")
def spy_fc():
    return pd.read_parquet(ROOT / "poster" / "data" / "forecasts_SPY.parquet")


@pytest.mark.skipif(not HAS_DATA, reason="faltan datos")
def test_backtest_static_equals_z0(spy, spy_fc):
    from poster.src.checks import check_static_equals_z0

    assert check_static_equals_z0(spy, spy_fc, Skew(a=0.83, b=-0.18, x_lo=-0.2, x_hi=1.0)) < 1e-12


@pytest.mark.skipif(not HAS_DATA, reason="faltan datos")
def test_daily_equity_matches_cycles(spy, spy_fc):
    from poster.src.checks import check_daily_matches_cycles
    from poster.src.data import build_cycles
    from poster.src.strategy import Spec, simulate

    sk = Skew(a=0.83, b=-0.18, x_lo=-0.2, x_hi=1.0)
    sim = simulate(spy, spy_fc, Spec("m"), build_cycles(spy["px"].index), sk)
    assert check_daily_matches_cycles(spy, sim, sk) < 1e-9


@pytest.mark.skipif(not HAS_DATA, reason="faltan datos")
def test_forecasts_have_no_lookahead(spy):
    from poster.src.checks import check_no_lookahead

    diffs = check_no_lookahead(spy, k=20)
    assert max(diffs.values()) < 1e-10
