# Financial Market & Macro Panel Data Pipeline

Pipeline en Python para la descarga automática, limpieza, alineación de series temporales y almacenamiento en formato columnar `.parquet` de un panel de activos financieros y variables macroeconómicas mediante `yfinance`.

---

## 📁 Estructura del Repositorio

```text
├── data/
│   └── panel_activos_macro.parquet  # Dataset consolidado y limpio en formato columnar
├── docs/
│   ├── esquema_tp.md                # 📄 Especificación metodológica y arquitectura del TP
│   ├── clase*.pdf                   # Diapositivas teóricas y prácticas del curso
│   └── clase*.html                  # Versiones exportadas en HTML
├── notebooks/
│   ├── clase2_practica_F.ipynb      # Práctica: No-arbitraje, smile de vol, Delta hedging
│   ├── clase3_practica_alumnos.ipynb
│   └── clase7_practica_A.ipynb
├── src/
│   ├── __init__.py
│   └── download_market_data.py      # Script modular de descarga, limpieza y exportación
├── .gitignore                       # Reglas de exclusión para Git y Python
├── README.md                        # Documentación general del repositorio
└── requirements.txt                 # Dependencias del proyecto
```

> 📖 Para consultar la especificación académica completa, hipótesis, modelos y fases de desarrollo, ver [**`docs/esquema_tp.md`**](docs/esquema_tp.md).

---

## ⚙️ Configuración y Variables

Todas las variables configurables se encuentran al inicio de [`src/download_market_data.py`](src/download_market_data.py):

```python
# 1. Panel de activos de interés
TICKERS = ["SPY", "NVDA", "AAPL", "MSFT", "QQQ"]

# 2. Variables macroeconómicas / de mercado:
MACRO_TICKERS = {
    "^VIX": "VIX",       # CBOE Volatility Index (volatilidad implícita 30d)
    "^VIX3M": "VIX3M",   # CBOE 3-Month Volatility Index
    "^IRX": "IRX",       # 13-Week Treasury Bill Rate (% anualizado)
}

# 3. Rango de fechas (formato 'YYYY-MM-DD')
START_DATE = "2020-01-01"
END_DATE = None  # None toma la fecha actual

# 4. Archivo de salida (por defecto en data/panel_activos_macro.parquet)
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = str(ROOT_DIR / "data" / "panel_activos_macro.parquet")
```

---

## 🛠️ Instalación y Requisitos

Se recomienda utilizar **Python 3.10+**.

Instalar las librerías necesarias con `pip`:

```bash
pip install -r requirements.txt
```

O con el lanzador de Python en Windows:

```powershell
py -3.13 -m pip install -r requirements.txt
```

---

## 🚀 Ejecución

Para ejecutar la descarga y actualizar el archivo `.parquet`:

```powershell
py -3.13 src/download_market_data.py
```

### Salida esperada por consola

```text
Descargando datos desde 2020-01-01 hasta 2026-09-30...
Símbolos solicitados (8): ['SPY', 'NVDA', 'AAPL', 'MSFT', 'QQQ', '^VIX', '^VIX3M', '^IRX']
[*********************100%***********************]  8 of 8 completed

Dimensiones finales del dataset limpio: 1696 filas x 8 columnas.
Período cubierto: 2020-01-02 a 2026-09-29
Dataset guardado exitosamente en: .../data/panel_activos_macro.parquet
```

---

## 📊 Cómo Cargar y Usar los Datos en tus Notebooks / Scripts

El archivo `.parquet` preserva los tipos de datos nativos de fecha y punto flotante con compresión eficiente:

```python
import pandas as pd

# Cargar dataset
df = pd.read_parquet("data/panel_activos_macro.parquet")

# Precios y variables
print(df.head())

# Cálculo de retornos diarios de activos
returns = df[["SPY", "NVDA", "AAPL", "MSFT", "QQQ"]].pct_change().dropna()

# Comparación con niveles de volatilidad (VIX)
analysis = returns.join(df[["VIX", "VIX3M", "IRX"]])
print(analysis.describe())
```

---

## 🧹 Tratamiento de Datos y Limpieza

1. **Precios Ajustados (`Adj Close`)**: Considera splits y dividendos en acciones y ETFs.
2. **Desfasaje de Calendarios (Feriados Bursátiles vs. Renta Fija)**:
   - Los bonos de corto plazo (`^IRX`) y las acciones (`SPY`, `AAPL`, etc.) operan bajo distintos calendarios de feriados (SIFMA vs. NYSE).
   - Se utiliza **Forward Fill (`ffill`)** para mantener la tasa vigente sin generar sesgos de anticipación (*lookahead bias*).
   - Se aplica `dropna()` para remover los períodos en que algún activo nuevo aún no había comenzado su cotización.
