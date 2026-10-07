"""Corre el pipeline completo del backtest y deja resultados (json), tablas y figuras en resultados/.

Uso:  python -m poster.src.build            (usa los pronósticos cacheados si existen)
      python -m poster.src.build --refresh  (recalcula los pronósticos; tarda unos minutos)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from poster.src import figures as F
from poster.src.data import _yf, build_cycles, load_ibkr, load_panel
from poster.src.strategy import (FLAT, Spec, calibrate_skew, daily_equity, deflated_sharpe, metrics, psr, run_all,
                                 simulate, stationary_bootstrap_diff)
from poster.src.vol_models import MODELS, forecast_cycles, forecast_table

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "poster" / "data"
OUT = ROOT / "resultados"
FIGS = OUT / "graficos"

MAIN = ["Buy & Hold", "CC estático ATM", "CC siempre OTM (IV)", "CC modelo"]
COMPARE = ["Buy & Hold", "CC estático ATM", "CC siempre OTM (IV)", "Siempre + strike modelo", "CC modelo (señal VIX)"]


def get_forecasts(A, refresh=False):
    fcs = {}
    for k, a in A.items():
        f = CACHE / f"forecasts_{k}.parquet"
        if f.exists() and not refresh:
            fcs[k] = pd.read_parquet(f)
        else:
            print(f"Pronósticos walk-forward {k}...", flush=True)
            fcs[k] = forecast_cycles(a["px"], a["iv"], build_cycles(a["px"].index))
            fcs[k].to_parquet(f)
    return fcs


def n_per_year(df):
    return len(df) / ((df["texp"].iloc[-1] - df.index[0]).days / 365.25)


def trial_grid(a, fc, cyc, skew):
    """Todas las configuraciones probadas (modelo x z x señal): base del Deflated Sharpe y de la tabla de robustez."""
    rows = []
    for sigma in MODELS + ["Ensamble"]:
        for z in [0.0, 0.5, 1.0]:
            for signal in ["strike", "index"]:
                sim = simulate(a, fc, Spec("t", sigma=sigma, z=z, signal=signal), cyc, skew)
                ex = sim["ret"] - sim["rf"]
                rows.append(dict(sigma=sigma, z=z, signal=signal, sr_periodo=ex.mean() / ex.std(),
                                 sharpe=ex.mean() / ex.std() * np.sqrt(n_per_year(sim)), vende=sim["sell"].mean()))
    return pd.DataFrame(rows)


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [str(v) for v in r.values]) + " |")
    return "\n".join(lines)


def main(refresh=False):
    FIGS.mkdir(parents=True, exist_ok=True)
    A = load_panel()
    fcs = get_forecasts(A, refresh)
    skew, skew_sample = calibrate_skew(load_ibkr(), A["SPY"], symbols=("SPX",))
    skew_spy, _ = calibrate_skew(load_ibkr(), A["SPY"], symbols=("SPY",))
    print(f"Skew SPX: a={skew.a:.3f} b={skew.b:.3f} x∈[{skew.x_lo:.2f},{skew.x_hi:.2f}] n={skew.n} | "
          f"SPY: a={skew_spy.a:.3f} b={skew_spy.b:.3f}")

    res, daily, summ, errs, boot, extra, grids = {}, {}, {}, {}, {}, {}, {}
    for k, a in A.items():
        print(f"Backtest {k}...", flush=True)
        cyc = build_cycles(a["px"].index)
        fc = fcs[k]
        res[k] = run_all(a, fc, cyc, skew)
        daily[k] = {n: daily_equity(a, s, skew) for n, s in res[k].items() if n in MAIN + ["CC modelo (señal VIX)"]}
        summ[k] = pd.DataFrame({n: metrics(s, daily[k].get(n)) for n, s in res[k].items()}).T
        errs[k] = forecast_table(fc)

        n_py = n_per_year(res[k]["CC modelo"])
        mod = res[k]["CC modelo"]["ret"].values
        boot[k] = {f"vs {b}": stationary_bootstrap_diff(mod - res[k][b]["ret"].values, n_py) for b in COMPARE}

        grids[k] = trial_grid(a, fc, cyc, skew)
        ex = (res[k]["CC modelo"]["ret"] - res[k]["CC modelo"]["rf"]).values
        active = mod - res[k]["Buy & Hold"]["ret"].values
        sell = res[k]["CC modelo"]["sell"].astype(bool)
        # IV de la call vendida (con el strike del modelo) en cada ciclo
        T = fc["cal"] / 365.0
        mny = 0.5 * fc["Ensamble"] * np.sqrt(fc["h"] / 252.0)
        iv_call = fc["IV_exec"] * np.array([skew.ratio(m, i, t) for m, i, t in zip(mny, fc["IV_exec"], T)])

        rob = {}
        for label, kw, sk in [("costo 5%", dict(cost=0.05), skew), ("skew de SPY", {}, skew_spy),
                              ("sin skew (IV plana)", dict(use_skew=False), FLAT), ("σ̂ = Kalman", dict(sigma="Kalman"), skew),
                              ("z = 1", dict(z=1.0), skew)]:
            r2 = run_all(a, fc, cyc, skew=sk, **kw)
            rob[label] = {n: float(metrics(r2[n])["sharpe"]) for n in ["Buy & Hold", "CC estático ATM", "CC siempre OTM (IV)", "CC modelo"]}

        extra[k] = dict(
            psr_modelo=psr(ex),
            psr_activo_vs_bh=psr(active),
            dsr=deflated_sharpe(ex, grids[k]["sr_periodo"].values),
            iv_indice_media=float(fc["IV_sig"].mean()),
            iv_call_vendida_media=float(iv_call.mean()),
            rv_media=float(fc["RV"].mean()),
            ciclos_sin_venta_por_anio={int(y): int(n) for y, n in pd.Series(fc.index[~sell.values].year).value_counts().sort_index().items()},
            robustez_sharpe=rob,
        )
        fc["IV_call"] = iv_call

    # validación: CC estático ATM simulado vs ETF BuyWrite real (PBP, replica el BXM: calls europeas de SPX,
    # sin ejercicio anticipado), por eso se compara contra la simulación sin ejercicio anticipado.
    pbp = _yf("PBP")["Adj Close"].dropna()
    spy_cyc = build_cycles(A["SPY"]["px"].index)
    st = simulate(A["SPY"], fcs["SPY"], Spec("est-eu", always=True, timing=False, model_strike=False, early_exercise=False),
                  spy_cyc, skew)
    st_am = res["SPY"]["CC estático ATM"]
    t = [st.index[0]] + list(st["texp"])
    r_pbp = pbp.reindex(t, method="ffill").pct_change().dropna().values
    yrs = (st["texp"].iloc[-1] - st.index[0]).days / 365.25
    cagr = lambda r: float(np.prod(1 + np.asarray(r)) ** (1 / yrs) - 1)
    pbp_val = dict(corr=float(np.corrcoef(st["ret"].values, r_pbp)[0, 1]),
                   cagr_sim_europea=cagr(st["ret"]), cagr_sim_americana=cagr(st_am["ret"]),
                   cagr_pbp=cagr(r_pbp), pbp_expense_ratio=0.0075, cagr_pbp_bruto=cagr(r_pbp) + 0.0075)

    out = dict(
        skew=dict(spx=skew.__dict__, spy=skew_spy.__dict__),
        summary={k: v.to_dict("index") for k, v in summ.items()},
        errors={k: v.to_dict("index") for k, v in errs.items()},
        bootstrap=boot, extra=extra, pbp=pbp_val, n_cycles={k: len(v) for k, v in fcs.items()},
        grid={k: v.to_dict("records") for k, v in grids.items()},
    )
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False), encoding="utf-8")

    # ---------- tablas (markdown)
    T = []
    for k in A:
        s = summ[k].loc[["Buy & Hold", "CC estático ATM", "CC siempre OTM (IV)", "Siempre + strike modelo",
                         "Timing modelo + ATM", "CC modelo (señal VIX)", "CC modelo"]]
        tab = pd.DataFrame({
            "Total": s.retorno_total.map(lambda x: pct(x, 0)), "CAGR": s.CAGR.map(pct), "Vol.": s.vol.map(pct),
            "Sharpe": s.sharpe.map("{:.2f}".format), "Sortino": s.sortino.map("{:.2f}".format),
            "MaxDD diario": s.maxdd.map(lambda x: pct(x, 0)), "MaxDD mensual": s.maxdd_mensual.map(lambda x: pct(x, 0)),
            "Vende": s.pct_vendido.map(lambda x: pct(x, 0)), "Aporte call/año": s.aporte_opcion_anual.map(lambda x: f"{100*x:+.1f} pp"),
            "Ej. anticip.": s.ejercicios_anticipados.astype(int),
        })
        tab.index.name = k
        T.append(f"### Estrategias — {k}\n\n" + md_table(tab))
        b = pd.DataFrame({n: {"Diferencia anual": f"{100*v['media_anual']:+.1f} pp",
                              "IC 95%": f"[{100*v['lo']:+.1f}; {100*v['hi']:+.1f}]"} for n, v in boot[k].items()}).T
        b.index.name = f"CC modelo — {k}"
        T.append(f"### CC modelo contra cada alternativa — {k} (bootstrap estacionario)\n\n" + md_table(b))
        e = errs[k][["RMSE", "QLIKE", "sesgo", "R2_MZ"]].copy()
        e["RMSE"] = (100 * e.RMSE).map("{:.1f}".format)
        e["sesgo"] = (100 * e.sesgo).map("{:+.1f}".format)
        e["QLIKE"] = e.QLIKE.map("{:.3f}".format)
        e["R2_MZ"] = e.R2_MZ.map("{:.2f}".format)
        e.index.name = f"Pronóstico — {k}"
        T.append(f"### Error de pronóstico fuera de muestra — {k} (RMSE y sesgo en pp de vol.)\n\n" + md_table(e))
        rb = pd.DataFrame(extra[k]["robustez_sharpe"]).T.apply(lambda c: c.map("{:.2f}".format))
        rb.index.name = f"Sharpe — {k}"
        T.append(f"### Robustez (Sharpe) — {k}\n\n" + md_table(rb))
    (OUT / "tablas.md").write_text("# Tablas de resultados\n\n" + "\n\n".join(T) + "\n", encoding="utf-8")

    # ---------- figuras
    F.set_output(FIGS)
    F.fig_payoff()
    F.fig_skew(skew_sample, skew)
    F.fig_iv_call(fcs)
    F.fig_errors(errs)
    F.fig_signal(fcs, res)
    F.fig_equity(res, MAIN)
    F.fig_option_leg(res)
    F.fig_drawdown(daily)
    F.fig_sharpe(summ)
    F.fig_pbp(st, r_pbp)
    F.fig_early_exercise(st, st_am)
    F.fig_overfitting(grids, summ, extra, {k: n_per_year(res[k]["CC modelo"]) for k in res})
    F.fig_signal_by_year(res)

    # ---------- póster (A0): figuras con letra grande + PDF, y números/tablas del LaTeX
    F.set_output(ROOT / "poster" / "overleaf" / "figures", poster=True)
    F.fig_payoff()
    F.fig_iv_call(fcs)
    F.fig_errors(errs)
    F.fig_equity(res, MAIN)
    F.set_output(FIGS)
    from src.plot_phase1 import ou_sscore_figure
    ou_sscore_figure(pd.read_parquet(ROOT / "data" / "features_phase1.parquet"),
                     ROOT / "poster" / "overleaf" / "figures" / "fase1_ou_sscore.pdf", poster=True)
    from poster.src import tex_numbers
    tex_numbers.main()
    return out


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
