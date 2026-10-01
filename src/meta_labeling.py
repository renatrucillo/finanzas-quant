"""
Módulo de Meta-Labeling, Triple Barrera y Purged Cross-Validation (Fase 3)
==========================================================================
Marco teórico: Marcos López de Prado (Advances in Financial Machine Learning),
enseñado en la Clase 6 de Finanzas Cuantitativas (FCEN-UBA).

Componentes:
1. Triple Barrier Method: Etiquetado de oportunidades de venta de Covered Calls.
2. Purged K-Fold Cross-Validation con Embargo: Validación sin fuga de información temporal.
3. Meta-Modelos Supervisados:
   - Logistic Regression (interpretable, baseline).
   - Random Forest Classifier (árboles no lineales, importancias Gini).
   - HistGradientBoostingClassifier (potencia no lineal rápida y calibrada).
4. Feature Selection & Bet Sizing: Generación de probabilidades P(y=1|X_t) para filtrar regímenes.
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

from src.pricing import (
    black_scholes_call_price,
    strike_from_delta,
    estimate_asset_implied_vol,
    covered_call_pnl_at_expiry,
)


# ==============================================================================
# 1. PURGED K-FOLD CON EMBARGO (Clase 6 - Diapositivas 26-28)
# ==============================================================================

class PurgedKFold:
    """
    K-Fold Cross Validation con Purga y Embargo temporal para observaciones con horizonte temporal.
    
    Evita la fuga de información (leakage) entre particiones contiguas debida a la superposición
    de la ventana de retención (ej. opciones a 21 ruedas de expiración).
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.02):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        t1: pd.Series,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Genera los índices de entrenamiento y prueba respetando la purga y embargo.

        Parámetros:
        -----------
        X : pd.DataFrame
            Matriz de features con DateTimeIndex.
        t1 : pd.Series
            Serie con la fecha de fin (vencimiento) de cada trade / observación.
            El índice de t1 debe coincidir con el de X.
        """
        indices = np.arange(len(X))
        step = int(np.ceil(len(X) / float(self.n_splits)))
        splits = []

        # Tamaño del embargo en número de observaciones
        embargo = int(np.ceil(len(X) * self.embargo_pct))

        for i in range(self.n_splits):
            test_start_idx = i * step
            test_end_idx = min((i + 1) * step, len(X))
            test_indices = indices[test_start_idx:test_end_idx]

            if len(test_indices) == 0:
                continue

            test_t0 = X.index[test_indices[0]]
            test_t1 = t1.iloc[test_indices[-1]]

            # 1. Purga: remover del conjunto de entrenamiento trades que se solapen con test
            train_mask = np.ones(len(X), dtype=bool)
            train_mask[test_indices] = False

            # Para cada observación candidata a train, verificar si su intervalo [t_start, t_end]
            # se solapa con el intervalo de test [test_t0, test_t1]
            train_t0 = X.index
            train_t1 = t1.values

            # Solapamiento: (train_t0 <= test_t1) & (train_t1 >= test_t0)
            overlap = (train_t0 <= test_t1) & (train_t1 >= test_t0)
            train_mask[overlap] = False

            # 2. Embargo: eliminar del conjunto de entrenamiento los días inmediatamente
            # posteriores al final del test set
            embargo_end_idx = min(test_indices[-1] + embargo, len(X))
            train_mask[test_indices[-1] : embargo_end_idx] = False

            train_indices = indices[train_mask]
            splits.append((train_indices, test_indices))

        return splits


# ==============================================================================
# 2. MÉTODO DE TRIPLE BARRERA PARA COVERED CALLS (Clase 6)
# ==============================================================================

def generate_covered_call_triple_barrier_labels(
    df_features: pd.DataFrame,
    asset: str = "SPY",
    roll_frequency: int = 21,  # Cada 21 ruedas (~1 mes hábil)
    delta_target: float = 0.30,
    take_profit_premium_pct: float = 0.75,  # Captura 75% de la prima
    stop_loss_vol_multiplier: float = 2.0,  # Caída del subyacente > 2 * vol_diaria * sqrt(t)
    div_yield: float = 0.015,
) -> pd.DataFrame:
    """
    Aplica el algoritmo de Triple Barrera para etiquetar cada oportunidad de venta de Call.

    Lógica financiera del etiquetado (Mundo Real vs Mundo Q):
    - Modelo Primario: Proponer Covered Call mensual a Delta = 0.30.
    - Barrera Superior (Take-Profit): Si el precio de la call cae a <= 25% de la prima original
      (captura de >= 75% del valor temporal sin esperar al vencimiento), trade exitoso (y = 1).
    - Barrera Inferior (Stop-Loss): Si el subyacente cae por debajo de S_0 - stop_loss_buffer
      superando con creces la prima amortiguadora, el régimen es bajista/crash destructivo (y = 0).
    - Barrera Vertical (Expiración a 21 ruedas):
      Si no tocó barreras intermedias:
      Si el P&L del Covered Call es positivo y no sufre capping severo en rally explosivo
      (es decir, CC return > 0 y stock_return - cc_return < 0.05), el trade es provechoso (y = 1).
      Si el activo cayó o el capping fue severo frente a un rally parabólico, se etiqueta como (y = 0).
    """
    adj_close_col = f"{asset}_adj_close"
    garch_vol_col = f"{asset}_gjr_cond_vol"
    spy_garch_col = "SPY_gjr_cond_vol"

    prices = df_features[adj_close_col]
    rf_annual = df_features["rf_annual"]
    vix = df_features["vix_level"]
    garch_vols = df_features[garch_vol_col]
    spy_vols = df_features[spy_garch_col]

    n_bars = len(df_features)
    trades = []

    # Iterar cada roll_frequency ruedas
    for i in range(0, n_bars - roll_frequency, roll_frequency):
        t0 = df_features.index[i]
        t1_idx = i + roll_frequency
        t1 = df_features.index[t1_idx]

        S0 = prices.iloc[i]
        r = rf_annual.iloc[i]
        sigma_g = garch_vols.iloc[i]
        sigma_spy_g = spy_vols.iloc[i]
        vix_lvl = vix.iloc[i]

        # Estimar volatilidad implícita específica del activo
        sigma_iv = estimate_asset_implied_vol(vix_lvl, sigma_g, sigma_spy_g)

        # Invertir Strike K para Delta objetivo
        T_entry = roll_frequency / 252.0
        K = strike_from_delta(S0, delta_target, T_entry, r, sigma_iv, q=div_yield)

        # Prima inicial cobrada
        C0 = black_scholes_call_price(S0, K, T_entry, r, sigma_iv, q=div_yield)

        # Umbrales de barrera
        take_profit_threshold = (1.0 - take_profit_premium_pct) * C0
        daily_vol = (sigma_g / np.sqrt(252.0))
        
        # Trayectoria intrames (días 1 a roll_frequency)
        touched_barrier = None
        exit_day_offset = roll_frequency
        exit_price = prices.iloc[t1_idx]
        exit_date = t1
        label = 0

        for day_step in range(1, roll_frequency + 1):
            curr_idx = i + day_step
            curr_date = df_features.index[curr_idx]
            S_curr = prices.iloc[curr_idx]
            t_rem = max((roll_frequency - day_step) / 252.0, 1e-4)

            # Valuación intermedia de la opción vendida
            C_curr = black_scholes_call_price(S_curr, K, t_rem, r, sigma_iv, q=div_yield)
            net_cc_ret = (S_curr - S0 + C0 - max(S_curr - K, 0.0)) / S0

            # 1. Barrera Inferior: Stop-loss / régimen bajista extremo (pérdida neta > 4%)
            loss_buffer_pct = max(stop_loss_vol_multiplier * daily_vol * np.sqrt(day_step), 0.04)
            if net_cc_ret < -loss_buffer_pct:
                touched_barrier = "lower_sl"
                exit_day_offset = day_step
                exit_price = S_curr
                exit_date = curr_date
                label = 0
                break

            # 2. Barrera Superior: Take-Profit en prima (captura >= 75% Y subyacente no en pérdida)
            if C_curr <= take_profit_threshold and S_curr >= S0:
                touched_barrier = "upper_tp"
                exit_day_offset = day_step
                exit_price = S_curr
                exit_date = curr_date
                label = 1
                break

        # 3. Si no tocó barreras intermedias, evalúa a vencimiento (Barrera Vertical)
        if touched_barrier is None:
            touched_barrier = "vertical_expiry"
            pnl_dict = covered_call_pnl_at_expiry(S0, exit_price, K, C0)
            cc_ret = pnl_dict["covered_call_return"]
            stock_ret = pnl_dict["stock_return"]

            # Favorable si CC es rentable Y no fue amputado severamente en un rally explosivo (<2.5% capping)
            if cc_ret >= 0.0 and (stock_ret - cc_ret) < 0.025:
                label = 1
            else:
                label = 0

        pnl_final = covered_call_pnl_at_expiry(S0, exit_price, K, C0)

        trades.append({
            "t0": t0,
            "t1": exit_date,
            "expiry_t1": t1,
            "asset": asset,
            "S0": S0,
            "K": K,
            "moneyness": K / S0 - 1.0,
            "C0": C0,
            "premium_pct": C0 / S0,
            "sigma_garch": sigma_g,
            "sigma_iv": sigma_iv,
            "exit_price": exit_price,
            "barrier_touched": touched_barrier,
            "holding_days": exit_day_offset,
            "stock_return": pnl_final["stock_return"],
            "cc_return": pnl_final["covered_call_return"],
            "is_assigned": pnl_final["is_assigned"],
            "label": label,
        })

    df_trades = pd.DataFrame(trades).set_index("t0")
    return df_trades


# ==============================================================================
# 3. CONSTRUCCIÓN DE LA MATRIZ DE FEATURES PARA EL META-MODELO
# ==============================================================================

def prepare_meta_features(
    df_features: pd.DataFrame,
    df_trades: pd.DataFrame,
    asset: str = "SPY",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Alinea las features de Fase 1 con las fechas t0 de los trades etiquetados.

    Features seleccionadas basadas en la teoría de la materia:
    - Volatilidad condicional GJR-GARCH (Clase 4)
    - Term-structure Ornstein-Uhlenbeck: s-score, half-life, kappa (Clase 7)
    - Variance Risk Premium: VIX - GARCH, ratio VIX/GARCH (Clase 2 & 4)
    - Diferenciación fraccionaria con memoria d* (Clase 6)
    - Kalman Beta y Volatilidad Idiosincrática (Clase 4 y 7)
    - Distancia al máximo histórico (Drawdown del subyacente)
    - Estimadores de volatilidad Yang-Zhang y Garman-Klass (Clase 4)
    """
    idx = df_trades.index.intersection(df_features.index)
    trades_aligned = df_trades.loc[idx]
    feats_aligned = df_features.loc[idx]

    # Cálculo de drawdown histórico respecto a ATH rodante
    price_full = df_features[f"{asset}_adj_close"]
    ath_rolling = price_full.expanding(min_periods=60).max()
    drawdown_series = (price_full - ath_rolling) / ath_rolling

    X = pd.DataFrame(index=idx)

    # 1. Regímenes de Volatilidad y VRP
    X["gjr_cond_vol"] = feats_aligned[f"{asset}_gjr_cond_vol"]
    X["vol_yz"] = feats_aligned[f"{asset}_vol_yz_21d"]
    X["vol_gk"] = feats_aligned[f"{asset}_vol_gk_21d"]
    X["vix_level"] = feats_aligned["vix_level"]
    X["vrp_spread"] = feats_aligned["vrp_vix_minus_spy_garch"]
    X["vrp_ratio"] = feats_aligned["vrp_ratio_vix_garch"]

    # 2. Term Structure OU del VIX
    X["ou_s_score"] = feats_aligned["ou_s_score"]
    X["ou_half_life"] = feats_aligned["ou_half_life"]
    X["ou_kappa"] = feats_aligned["ou_kappa"]
    X["vix_kalman_spread"] = feats_aligned["vix_kalman_spread"]

    # 3. Memoria y CAPM
    X["fracdiff_ret"] = feats_aligned[f"{asset}_fracdiff_dopt"]
    X["kalman_beta"] = feats_aligned[f"{asset}_kalman_beta"]
    X["idiosyncratic_vol"] = feats_aligned[f"{asset}_idiosyncratic_vol_21d"]
    X["drawdown_ath"] = drawdown_series.loc[idx]

    # 4. Ratios de Moneyness de la opción propuesta
    X["premium_yield"] = trades_aligned["premium_pct"]
    X["moneyness"] = trades_aligned["moneyness"]

    # Limpieza de NaNs residuales con forward-fill
    X = X.ffill().bfill()
    y = trades_aligned["label"].astype(int)
    t1 = trades_aligned["t1"]

    return X, y, t1


# ==============================================================================
# 4. ENTRENAMIENTO Y EVALUACIÓN CON PURGED CROSS-VALIDATION
# ==============================================================================

def train_and_evaluate_meta_models(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    n_splits: int = 5,
    embargo_pct: float = 0.02,
) -> Dict[str, any]:
    """
    Entrena y compara los tres meta-modelos bajo Purged K-Fold con Embargo:
    1. Regresión Logística (L2, estandarizada).
    2. Random Forest Classifier (con árboles restringidos para evitar sobreajuste).
    3. HistGradientBoostingClassifier (boosting no lineal regularizado).

    Retorna métricas honestas OOF (Out-Of-Fold) y probabilidades calibradas.
    """
    purged_cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    splits = purged_cv.split(X, t1)

    models = {
        "LogisticRegression": LogisticRegression(C=0.5, solver="lbfgs", max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, max_depth=3, min_samples_leaf=10, random_state=42),
    }

    results = {}
    oof_predictions = {name: pd.Series(np.nan, index=X.index) for name in models}
    feature_importances = {name: pd.Series(0.0, index=X.columns) for name in ["LogisticRegression", "RandomForest", "HistGradientBoosting"]}

    for name, model in models.items():
        y_true_all = []
        y_pred_probs_all = []
        importances_list = []

        for train_idx, test_idx in splits:
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            # Si alguna partición solo tiene una clase por purga, saltear
            if len(np.unique(y_train)) < 2:
                continue

            # Escalamiento específico por fold para evitar data leakage
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            if name == "LogisticRegression":
                model.fit(X_train_sc, y_train)
                probs = model.predict_proba(X_test_sc)[:, 1]
                importances_list.append(np.abs(model.coef_[0]))
            elif name == "RandomForest":
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]
                importances_list.append(model.feature_importances_)
            else:  # HistGradientBoosting
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]

            oof_predictions[name].iloc[test_idx] = probs
            y_true_all.extend(y_test.values)
            y_pred_probs_all.extend(probs)

        # Métricas Out-Of-Fold consolidadas
        valid_mask = ~oof_predictions[name].isna()
        y_true_valid = y.loc[valid_mask]
        y_prob_valid = oof_predictions[name].loc[valid_mask]
        y_pred_binary = (y_prob_valid >= 0.50).astype(int)

        auc = roc_auc_score(y_true_valid, y_prob_valid)
        acc = accuracy_score(y_true_valid, y_pred_binary)
        prec = precision_score(y_true_valid, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_valid, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_valid, y_pred_binary, zero_division=0)
        brier = brier_score_loss(y_true_valid, y_prob_valid)

        avg_imp = pd.Series(0.0, index=X.columns)
        if len(importances_list) > 0:
            avg_imp = pd.Series(np.mean(importances_list, axis=0), index=X.columns)

        results[name] = {
            "roc_auc": auc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "brier_score": brier,
            "feature_importances": avg_imp,
        }

    # Modelo entrenado sobre todo el histórico para inferencia / producción
    fitted_models = {}
    scaler_full = StandardScaler().fit(X)
    for name, model in models.items():
        if name == "LogisticRegression":
            model.fit(scaler_full.transform(X), y)
        else:
            model.fit(X, y)
        fitted_models[name] = model

    return {
        "metrics": results,
        "oof_probabilities": pd.DataFrame(oof_predictions),
        "fitted_models": fitted_models,
        "scaler": scaler_full,
        "splits_count": len(splits),
    }


def run_phase3_pipeline(
    features_path: str = "data/features_phase1.parquet",
    assets: List[str] = ["SPY", "QQQ", "NVDA"],
) -> Dict[str, any]:
    """
    Ejecuta el pipeline completo de Fase 3 para todos los activos del panel.
    """
    print("=================================================================")
    print("       INICIANDO PIPELINE DE FASE 3: META-LABELING (ML)          ")
    print("=================================================================")
    df_features = pd.read_parquet(features_path)
    all_results = {}

    for asset in assets:
        print(f"\n>>> [Fase 3] Procesando {asset} con Triple Barrera y Purged CV...")
        df_trades = generate_covered_call_triple_barrier_labels(
            df_features,
            asset=asset,
            roll_frequency=21,
            delta_target=0.30,
        )
        print(f"  Total trades generados: {len(df_trades)}")
        print(f"  Tasa de éxito (label=1): {df_trades['label'].mean():.2%}")
        print(f"  Desglose de barreras: {dict(df_trades['barrier_touched'].value_counts())}")

        X, y, t1 = prepare_meta_features(df_features, df_trades, asset=asset)
        print(f"  Matriz de Features X: {X.shape[0]} muestras x {X.shape[1]} variables")

        cv_results = train_and_evaluate_meta_models(X, y, t1, n_splits=5, embargo_pct=0.02)
        print(f"  Resultados Purged K-Fold CV:")
        for m_name, mets in cv_results["metrics"].items():
            print(f"    - {m_name:20s}: ROC-AUC={mets['roc_auc']:.3f}, Acc={mets['accuracy']:.2%}, Prec={mets['precision']:.2%}, Rec={mets['recall']:.2%}, Brier={mets['brier_score']:.4f}")

        # Top 3 features más influyentes
        rf_imp = cv_results["metrics"]["RandomForest"]["feature_importances"].sort_values(ascending=False)
        print(f"  Top 3 Features (Random Forest): {', '.join([f'{k} ({v:.3f})' for k, v in rf_imp.head(3).items()])}")

        all_results[asset] = {
            "df_trades": df_trades,
            "X": X,
            "y": y,
            "t1": t1,
            "cv_results": cv_results,
        }

    print("\n[OK] Pipeline de Fase 3 completado exitosamente para todo el panel!")
    return all_results


if __name__ == "__main__":
    run_phase3_pipeline()
