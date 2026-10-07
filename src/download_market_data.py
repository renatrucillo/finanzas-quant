"""
Script para descargar datos diarios de activos y variables macroeconómicas con yfinance.
Limpia NaNs y exporta el panel consolidado a formato .parquet.
"""

from datetime import datetime
from pathlib import Path
import pandas as pd
import yfinance as yf

# ==============================================================================
# CONFIGURACIÓN DE PARÁMETROS (Modificar aquí)
# ==============================================================================

# 1. Panel de activos de interés
TICKERS = ["SPY", "NVDA", "AAPL", "MSFT", "QQQ"]

# 2. Variables macroeconómicas / de mercado:
#    ^VIX: CBOE Volatility Index
#    ^VIX3M: CBOE 3-Month Volatility Index
#    ^IRX: 13-Week Treasury Bill Rate (Tasa libre de riesgo anualizada en %)
MACRO_TICKERS = {
    "^VIX": "VIX",
    "^VIX3M": "VIX3M",
    "^IRX": "IRX",  # Tasa 13w T-Bill anualizada (%)
}

# 3. Rango de fechas (formato 'YYYY-MM-DD'). Si END_DATE es None, usa la fecha de hoy.
START_DATE = "2007-01-01"
END_DATE = None  # Ej: '2025-12-31' o None para fecha actual

# 4. Ruta del archivo de salida
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = str(ROOT_DIR / "data" / "panel_activos_macro.parquet")

# ==============================================================================
# FUNCIÓN PRINCIPAL DE DESCARGA Y PROCESAMIENTO
# ==============================================================================

def download_and_process_data(
    tickers: list[str],
    macro_mapping: dict[str, str],
    start_date: str,
    end_date: str | None = None,
    output_path: str = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Descarga precios ajustados de activos y variables macroeconómicas,
    alinea fechas, limpia NaNs y exporta a .parquet.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    all_symbols = list(tickers) + list(macro_mapping.keys())
    print(f"Descargando datos desde {start_date} hasta {end_date}...")
    print(f"Símbolos solicitados ({len(all_symbols)}): {all_symbols}")

    # Descarga unificada
    raw_data = yf.download(
        tickers=all_symbols,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=True,
    )

    if raw_data.empty:
        raise ValueError("No se descargaron datos. Verificá los tickers y el rango de fechas.")

    # Extraer precios ajustados ('Adj Close')
    if "Adj Close" in raw_data.columns:
        df_prices = raw_data["Adj Close"].copy()
    else:
        df_prices = raw_data["Close"].copy()

    # Si se descargó un solo activo, pandas lo devuelve como Series
    if isinstance(df_prices, pd.Series):
        df_prices = df_prices.to_frame(name=all_symbols[0])

    # Renombrar columnas de variables macroeconómicas
    df_prices.rename(columns=macro_mapping, inplace=True)

    # Asegurar orden cronológico e índice datetime sin zona horaria
    df_prices.sort_index(inplace=True)
    if df_prices.index.tz is not None:
        df_prices.index = df_prices.index.tz_localize(None)

    # -------------------------------------------------------------------------
    # TRATAMIENTO Y LIMPIEZA DE NaNs
    # -------------------------------------------------------------------------
    print("\nResumen inicial de NaNs por columna:")
    print(df_prices.isna().sum())

    # 1. Forward-fill: completa días no operativos en bonos/macro con el último valor disponible.
    #    No se usa backward-fill: rellenaría el pasado con valores futuros (lookahead).
    df_cleaned = df_prices.ffill()

    # 2. Eliminar filas residuales con NaNs (inicio de series que arrancan más tarde)
    df_cleaned = df_cleaned.dropna()

    print(f"\nDimensiones finales del dataset limpio: {df_cleaned.shape[0]} filas x {df_cleaned.shape[1]} columnas.")
    print(f"Período cubierto: {df_cleaned.index.min().date()} a {df_cleaned.index.max().date()}")

    # -------------------------------------------------------------------------
    # GUARDADO EN PARQUET (Precios ajustados estándar)
    # -------------------------------------------------------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_parquet(output_path, engine="pyarrow")
    print(f"Dataset de precios ajustados guardado exitosamente en: {output_path}")

    # -------------------------------------------------------------------------
    # GUARDADO EN PARQUET DE PANEL OHLCV COMPLETO (Para Parkinson, GK, YZ)
    # -------------------------------------------------------------------------
    try:
        ohlcv_dict = {}
        # Procesar activos con OHLCV
        for sym in tickers:
            for field in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                if (field in raw_data.columns) and (sym in raw_data[field].columns):
                    ohlcv_dict[f"{sym}_{field.replace(' ', '_')}"] = raw_data[field][sym]

        # Agregar variables macro (Close o Adj Close)
        for sym, name in macro_mapping.items():
            if ("Adj Close" in raw_data.columns) and (sym in raw_data["Adj Close"].columns):
                ohlcv_dict[name] = raw_data["Adj Close"][sym]
            elif ("Close" in raw_data.columns) and (sym in raw_data["Close"].columns):
                ohlcv_dict[name] = raw_data["Close"][sym]

        df_ohlcv = pd.DataFrame(ohlcv_dict, index=raw_data.index)
        df_ohlcv.sort_index(inplace=True)
        if df_ohlcv.index.tz is not None:
            df_ohlcv.index = df_ohlcv.index.tz_localize(None)

        df_ohlcv = df_ohlcv.ffill().dropna()
        ohlcv_path = Path(output_path).parent / "panel_ohlcv.parquet"
        df_ohlcv.to_parquet(ohlcv_path, engine="pyarrow")
        print(f"Dataset OHLCV completo guardado exitosamente en: {ohlcv_path}")
    except Exception as e:
        print(f"Aviso: no se pudo guardar panel_ohlcv.parquet: {e}")

    return df_cleaned


if __name__ == "__main__":
    df = download_and_process_data(
        tickers=TICKERS,
        macro_mapping=MACRO_TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        output_path=OUTPUT_FILE,
    )
    print("\nPrimeras 5 filas:")
    print(df.head())
