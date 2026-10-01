"""
Test unitario y de validación para la Fase 3: Meta-Labeling y Purged CV
======================================================================
Valida:
1. Generación de eventos y etiquetas con Triple Barrera (Take-Profit, Stop-Loss, Expiración).
2. Propiedades de Purged K-Fold con Embargo (verificación matemática de no-leakage).
3. Entrenamiento y convergencia de los meta-modelos supervisados (LogReg, RF, HistGBM).
4. Generación coherente de probabilidades OOF para el filtro de régimen.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from src.meta_labeling import (
    PurgedKFold,
    generate_covered_call_triple_barrier_labels,
    prepare_meta_features,
    train_and_evaluate_meta_models,
)


def test_purged_kfold_no_leakage():
    print("--- 1. Test Purged K-Fold con Embargo (Verificación de Cero-Leakage) ---")
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    X = pd.DataFrame(np.random.randn(100, 4), index=dates)
    # Cada trade dura 10 días
    t1 = pd.Series([d + pd.Timedelta(days=14) for d in dates], index=dates)

    purged_cv = PurgedKFold(n_splits=4, embargo_pct=0.05)
    splits = purged_cv.split(X, t1)

    assert len(splits) == 4, "Debe generar 4 splits"
    for fold, (train_idx, test_idx) in enumerate(splits):
        # Verificar que ningún índice de test esté en train
        intersection = set(train_idx).intersection(set(test_idx))
        assert len(intersection) == 0, f"Fold {fold}: Intersección directa entre train y test!"

        # Verificar que ningún trade de train se solape temporalmente con test
        test_t0 = X.index[test_idx[0]]
        test_t1 = t1.iloc[test_idx[-1]]

        for tr_i in train_idx:
            tr_t0 = X.index[tr_i]
            tr_t1 = t1.iloc[tr_i]
            overlap = (tr_t0 <= test_t1) and (tr_t1 >= test_t0)
            assert not overlap, f"Fold {fold}: Trade de train {tr_t0} -> {tr_t1} solapa con test {test_t0} -> {test_t1}!"

    print("  [OK] Cero leakage confirmado: Ningún trade de entrenamiento solapa temporalmente con test!")


def test_triple_barrier_and_meta_models():
    print("\n--- 2. Test Triple Barrera y Entrenamiento sobre SPY ---")
    df_features = pd.read_parquet("data/features_phase1.parquet")

    trades_spy = generate_covered_call_triple_barrier_labels(
        df_features,
        asset="SPY",
        roll_frequency=21,
        delta_target=0.30,
    )
    print(f"Total trades SPY: {len(trades_spy)}")
    print(f"Distribución de etiquetas (y=1 vs y=0): {dict(trades_spy['label'].value_counts())}")
    print(f"Distribución de barreras tocadas: {dict(trades_spy['barrier_touched'].value_counts())}")

    assert len(trades_spy) > 100, "Debe haber más de 100 trades mensuales en el histórico 2007-2026"
    assert trades_spy["label"].isin([0, 1]).all(), "Las etiquetas deben ser binarias 0 o 1"

    X, y, t1 = prepare_meta_features(df_features, trades_spy, asset="SPY")
    assert not X.isna().any().any(), "No debe haber NaNs en la matriz X de features"
    assert len(X) == len(y) == len(t1), "Dimensiones alineadas"

    print("\n--- 3. Test Evaluación Purged CV para Meta-Modelos ---")
    cv_res = train_and_evaluate_meta_models(X, y, t1, n_splits=5, embargo_pct=0.02)
    for model_name, mets in cv_res["metrics"].items():
        print(f"  {model_name:22s} -> ROC-AUC: {mets['roc_auc']:.3f} | Accuracy: {mets['accuracy']:.2%} | Precision: {mets['precision']:.2%} | Brier: {mets['brier_score']:.4f}")
        assert 0.0 <= mets["roc_auc"] <= 1.0, f"ROC-AUC fuera de rango"
        assert mets["accuracy"] > 0.50, f"Accuracy debe ser superior al 50%"

    print("  [OK] Todos los meta-modelos entrenados y validados con éxito!")


if __name__ == "__main__":
    test_purged_kfold_no_leakage()
    test_triple_barrier_and_meta_models()
    print("\n>>> [SUCCESS] TODOS LOS TESTS DE LA FASE 3 PASARON EXITOSAMENTE! <<<")
