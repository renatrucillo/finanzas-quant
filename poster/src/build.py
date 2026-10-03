"""Corre el pipeline completo y deja resultados (json), tablas LaTeX y figuras para el póster."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from poster.src import figures as F
from poster.src.data import build_cycles, load_panel
from poster.src.strategy import metrics, psr, run_all, stationary_bootstrap_diff, summary_table
from poster.src.vol_models import MODELS, forecast_cycles, forecast_table

ROOT = Path(__file__).resolve().parents[1]
DATA, TEX = ROOT / "data", ROOT / "overleaf"


def get_forecasts(A, refresh=False):
    fcs = {}
    for k, a in A.items():
        f = DATA / f"forecasts_{k}.parquet"
        if f.exists() and not refresh:
            fcs[k] = pd.read_parquet(f)
        else:
            fcs[k] = forecast_cycles(a["px"], a["iv"], build_cycles(a["px"].index))
            fcs[k].to_parquet(f)
    return fcs


def pct(x, d=1):
    return f"{100 * x:.{d}f}" + r"\%"


def main(refresh=False):
    A = load_panel()
    fcs = get_forecasts(A, refresh)
    res, summ, errs, boot, extra = {}, {}, {}, {}, {}
    for k, a in A.items():
        cyc = build_cycles(a["px"].index)
        res[k] = run_all(a, fcs[k], cyc)
        summ[k] = summary_table(res[k])
        errs[k] = forecast_table(fcs[k])
        n_py = metrics(res[k]["CC modelo"])["n"] / ((res[k]["CC modelo"]["texp"].iloc[-1] - res[k]["CC modelo"].index[0]).days / 365.25)
        d_stat = (res[k]["CC modelo"]["ret"] - res[k]["CC estático ATM"]["ret"]).values
        d_bh = (res[k]["CC modelo"]["ret"] - res[k]["Buy & Hold"]["ret"]).values
        boot[k] = dict(vs_estatico=stationary_bootstrap_diff(d_stat, n_py), vs_bh=stationary_bootstrap_diff(d_bh, n_py))
        sold = res[k]["CC modelo"]["sell"].astype(bool)
        bh = res[k]["Buy & Hold"]["ret"]
        cc = res[k]["CC estático ATM"]["ret"]
        extra[k] = dict(
            psr_modelo=psr(res[k]["CC modelo"]["ret"].values),
            hit_sell=float((res[k]["CC modelo"]["ret"][sold] > bh[sold]).mean()),
            hit_nosell_estatico_pierde=float((cc[~sold] < bh[~sold]).mean()) if (~sold).any() else np.nan,
            gana_estatico_en_vendidos=float(((res[k]["CC modelo"]["ret"] - cc)[sold]).mean()),
            gana_estatico_en_no_vendidos=float(((bh - cc)[~sold]).mean()) if (~sold).any() else np.nan,
        )
    out = dict(summary={k: v.to_dict("index") for k, v in summ.items()}, errors={k: v.to_dict("index") for k, v in errs.items()},
               bootstrap=boot, extra=extra, n_cycles={k: len(v) for k, v in fcs.items()})
    (DATA / "results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    # ---------- figuras
    F.fig_payoff(); F.fig_vrp(fcs); F.fig_forecast(fcs); F.fig_errors(errs); F.fig_signal(fcs, res)
    F.fig_equity(res); F.fig_ablation(summ)

    # ---------- tablas LaTeX
    rows = []
    for k in A:
        for name in ["Buy & Hold", "CC estático ATM", "CC modelo"]:
            m = summ[k].loc[name]
            nm = name.replace("&", r"\&")
            sold = "--" if name == "Buy & Hold" else pct(m["pct_vendido"], 0)
            rows.append(f"{k} & {nm} & {pct(m['retorno_total'],0)} & {pct(m['CAGR'])} & {pct(m['vol'])} & {m['sharpe']:.2f} & {pct(m['maxdd'],0)} & {sold} \\\\")
        if k != list(A)[-1]:
            rows.append(r"\midrule")
    head = (r"\begin{tabular}{@{}llrrrrrr@{}}" "\n" r"\toprule" "\n"
            r"& & \textbf{Total} & \textbf{CAGR} & \textbf{Vol.} & \textbf{Sharpe} & \textbf{MaxDD} & \textbf{Vende} \\ \midrule")
    (TEX / "tabla_estrategias.tex").write_text(head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" "\n" r"\end{tabular}" "\n", encoding="utf-8")

    erows = []
    for m in MODELS + ["Ensamble", "IV"]:
        cells = []
        for k in A:
            e = errs[k].loc[m]
            cells += [f"{100*e['RMSE']:.1f}", f"{e['QLIKE']:.2f}", f"{e['R2_MZ']:.2f}"]
        nm = {"IV": "Implícita (VIX/VXN)"}.get(m, m)
        erows.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    (TEX / "tabla_errores.tex").write_text("\n".join(erows), encoding="utf-8")
    return fcs, res, summ, errs, boot, extra


if __name__ == "__main__":
    main()
