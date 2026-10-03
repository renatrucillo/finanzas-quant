"""Genera overleaf/numeros.tex (macros LaTeX) a partir de data/results.json: texto y tablas siempre en sintonía."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BS = chr(92)  # backslash


def tex_minus(x: str) -> str:
    return x.replace("-", "$-$")


def main():
    r = json.loads((ROOT / "data" / "results.json").read_text(encoding="utf-8"))
    fc = {k: pd.read_parquet(ROOT / "data" / f"forecasts_{k}.parquet") for k in ["SPY", "QQQ"]}
    L = []

    def mac(name, val):
        L.append(BS + "newcommand{" + BS + name + "}{" + val + "}")

    pc = lambda x, d=1: f"{100 * x:.{d}f}" + BS + "%"
    pp = lambda x, d=1: tex_minus(f"{100 * x:+.{d}f}") + " pp"
    for k in ["SPY", "QQQ"]:
        s = r["summary"][k]
        mac(f"{k}IV", pc(fc[k]["IV_sig"].mean()))
        mac(f"{k}RV", pc(fc[k]["RV"].mean()))
        mac(f"{k}VRP", f"{100 * (fc[k]['IV_sig'].mean() - fc[k]['RV'].mean()):.1f} pp")
        for key, nm in [("Buy & Hold", "BH"), ("CC estático ATM", "Stat"), ("CC modelo", "Mod")]:
            mac(f"{k}{nm}CAGR", pc(s[key]["CAGR"]))
            mac(f"{k}{nm}Sharpe", f"{s[key]['sharpe']:.2f}")
            mac(f"{k}{nm}DD", tex_minus(pc(s[key]["maxdd"], 0)))
            mac(f"{k}{nm}Vol", pc(s[key]["vol"]))
        mac(f"{k}Sold", pc(s["CC modelo"]["pct_vendido"], 0))
        for key, nm in [("vs_estatico", "VsStat"), ("vs_bh", "VsBH")]:
            b = r["bootstrap"][k][key]
            mac(f"{k}{nm}", pp(b["media_anual"]))
            mac(f"{k}{nm}CI", tex_minus(f"[{100 * b['lo']:+.1f}; {100 * b['hi']:+.1f}]"))
        mac(f"{k}Hit", pc(r["extra"][k]["hit_sell"], 0))
        for key, nm in [("Siempre + strike modelo", "Strike"), ("Timing modelo + ATM", "Timing")]:
            mac(f"{k}{nm}Sharpe", f"{s[key]['sharpe']:.2f}")
    (ROOT / "overleaf" / "numeros.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(L)} macros -> overleaf/numeros.tex")


if __name__ == "__main__":
    main()
