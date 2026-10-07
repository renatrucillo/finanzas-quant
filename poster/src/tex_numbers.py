"""Genera overleaf/numeros.tex (macros) y overleaf/tabla_estrategias.tex a partir de resultados/results.json
y de las features de la fase 1: ningún número del póster se escribe a mano."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TEX = ROOT / "poster" / "overleaf"
BS = chr(92)  # backslash

STRATS = [("Buy & Hold", "BH"), ("CC estático ATM", "Stat"), ("CC siempre OTM (IV)", "OTM"), ("CC modelo", "Mod")]


def num(x: float, d: int = 1) -> str:
    """Número con coma decimal y signo menos tipográfico."""
    s = f"{x:.{d}f}".replace(".", "{,}")
    return s.replace("-", "$-$")


def signed(x: float, d: int = 1) -> str:
    s = f"{x:+.{d}f}".replace(".", "{,}")
    return s.replace("-", "$-$")


def main():
    r = json.loads((ROOT / "resultados" / "results.json").read_text(encoding="utf-8"))
    L = []

    def mac(name, val):
        L.append(BS + "newcommand{" + BS + name + "}{" + val + "}")

    pc = lambda x, d=1: num(100 * x, d) + BS + "%"
    for k in ["SPY", "QQQ"]:
        s, e, b = r["summary"][k], r["extra"][k], r["bootstrap"][k]
        mac(f"{k}IV", pc(e["iv_indice_media"]))
        mac(f"{k}IVcall", pc(e["iv_call_vendida_media"]))
        mac(f"{k}RV", pc(e["rv_media"]))
        for key, nm in STRATS:
            mac(f"{k}{nm}CAGR", pc(s[key]["CAGR"]))
            mac(f"{k}{nm}Sharpe", num(s[key]["sharpe"], 2))
            mac(f"{k}{nm}Sortino", num(s[key]["sortino"], 2))
            mac(f"{k}{nm}DD", pc(s[key]["maxdd"], 0))
            mac(f"{k}{nm}Vol", pc(s[key]["vol"]))
            mac(f"{k}{nm}Sold", pc(s[key]["pct_vendido"], 0))
            mac(f"{k}{nm}Aporte", signed(100 * s[key]["aporte_opcion_anual"]) + " pp")
            mac(f"{k}{nm}Early", str(int(s[key]["ejercicios_anticipados"])))
        for key, nm in [("vs Buy & Hold", "VsBH"), ("vs CC siempre OTM (IV)", "VsOTM"), ("vs CC estático ATM", "VsStat")]:
            mac(f"{k}{nm}", signed(100 * b[key]["media_anual"]) + " pp")
            mac(f"{k}{nm}CI", "[" + signed(100 * b[key]["lo"]) + "; " + signed(100 * b[key]["hi"]) + "]")
        mac(f"{k}PSRactivo", num(e["psr_activo_vs_bh"], 2))
        mac(f"{k}DSR", num(e["dsr"]["dsr"], 2))
        mac(f"{k}QlikeKalman", num(r["errors"][k]["Kalman"]["QLIKE"], 2))
        mac(f"{k}QlikeIV", num(r["errors"][k]["IV"]["QLIKE"], 2))
        mac(f"{k}QlikeEns", num(r["errors"][k]["Ensamble"]["QLIKE"], 2))
    mac("NCycles", str(r["n_cycles"]["SPY"]))
    mac("NTrials", str(r["extra"]["SPY"]["dsr"]["n_pruebas"]))
    mac("SkewA", num(r["skew"]["spx"]["a"], 2))
    mac("SkewB", num(abs(r["skew"]["spx"]["b"]), 2))
    mac("SkewN", str(r["skew"]["spx"]["n"]))
    mac("PBPcorr", num(r["pbp"]["corr"], 2))
    mac("PBPsim", pc(r["pbp"]["cagr_sim_europea"]))
    mac("PBPsimAm", pc(r["pbp"]["cagr_sim_americana"]))
    mac("PBPbruto", pc(r["pbp"]["cagr_pbp_bruto"]))
    mac("EarlyCost", num(100 * (r["pbp"]["cagr_sim_europea"] - r["pbp"]["cagr_sim_americana"]), 1) + " pp")

    # fase 1
    f = pd.read_parquet(ROOT / "data" / "features_phase1.parquet")
    raw = pd.read_parquet(ROOT / "data" / "panel_ohlcv.parquet")
    from src.volatility_estimators import garman_klass_volatility

    gk = garman_klass_volatility(raw["SPY_Open"], raw["SPY_High"], raw["SPY_Low"], raw["SPY_Close"]).reindex(f.index)
    mac("VrpGK", num(100 * (f["vix_level"] / 100 - gk).mean()) + " pp")
    mac("VrpYZ", num(100 * f["vrp_vix_minus_spy_yz"].mean()) + " pp")
    mac("OUhalflife", num(f["ou_half_life"].median(), 0))
    mac("OUstress", pc((f["ou_regime"] == "estres_extremo").mean()))
    for sym in ["SPY", "QQQ", "NVDA"]:
        mac(f"Dopt{sym}", num(f[f"{sym}_d_optimal"].iloc[0], 2))
    (TEX / "numeros.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # tabla de estrategias
    rows = []
    for k in ["SPY", "QQQ"]:
        s = r["summary"][k]
        for i, (key, nm) in enumerate(STRATS):
            label = {"Buy & Hold": r"Buy \& Hold", "CC estático ATM": "CC estático ATM",
                     "CC siempre OTM (IV)": "CC siempre OTM", "CC modelo": r"\textbf{CC modelo}"}[key]
            first = rf"\multirow{{4}}{{*}}{{\textbf{{{k}}}}}" if i == 0 else ""
            sold = "--" if key == "Buy & Hold" else pc(s[key]["pct_vendido"], 0)
            rows.append(f"{first} & {label} & {pc(s[key]['CAGR'])} & {pc(s[key]['vol'])} & {num(s[key]['sharpe'], 2)} & "
                        f"{num(s[key]['sortino'], 2)} & {pc(s[key]['maxdd'], 0)} & {sold} \\\\")
        if k == "SPY":
            rows.append(r"\midrule")
    head = (r"\begin{tabular}{@{}llrrrrrr@{}}" "\n" r"\toprule" "\n"
            r"& & \textbf{CAGR} & \textbf{Vol.} & \textbf{Sharpe} & \textbf{Sortino} & \textbf{MaxDD} & \textbf{Vende} \\ \midrule")
    (TEX / "tabla_estrategias.tex").write_text(head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" "\n" r"\end{tabular}" "\n",
                                               encoding="utf-8")
    print(f"{len(L)} macros -> {TEX / 'numeros.tex'}")


if __name__ == "__main__":
    main()
