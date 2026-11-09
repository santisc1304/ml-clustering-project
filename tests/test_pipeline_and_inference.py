"""
Pruebas automáticas unitarias y de integración para el pipeline MLOps de Clustering.
Verifica determinismo matemático, integridad de datos y contrato de inferencia.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from ml_clustering_project.src.config import (
    FEATURES,
    RANDOM_SEED,
    N_CLUSTERS,
    DATA_PATH_PRIMARY,
    MODEL_PATH_DEFAULT
)
from ml_clustering_project.src.clustering_pipeline import (
    build_clustering_pipeline,
    evaluate_clustering
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_dataset_integrity():
    """Verifica que el dataset exista, tenga dimensiones correctas y las columnas requeridas."""
    data_path = PROJECT_ROOT / DATA_PATH_PRIMARY
    assert data_path.exists(), f"El archivo {data_path} no existe."

    df = pd.read_csv(data_path)
    assert len(df) == 3150, f"Se esperaban 3150 registros, se obtuvieron {len(df)}."
    for col in FEATURES:
        assert col in df.columns, f"Columna requerida '{col}' no encontrada en el dataset."
        assert pd.api.types.is_numeric_dtype(df[col]), f"La columna '{col}' debe ser numérica."


def test_pipeline_determinism():
    """Verifica que el pipeline produzca exactamente los mismos clusters bajo la misma semilla."""
    data_path = PROJECT_ROOT / DATA_PATH_PRIMARY
    df = pd.read_csv(data_path)
    X = df[FEATURES]

    pipe1 = build_clustering_pipeline(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
    labels1 = pipe1.fit_predict(X)

    pipe2 = build_clustering_pipeline(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
    labels2 = pipe2.fit_predict(X)

    np.testing.assert_array_equal(labels1, labels2, err_msg="El pipeline no es determinista.")


def test_clustering_metrics_threshold():
    """Verifica que las métricas de calidad de agrupamiento superen los umbrales mínimos industriales."""
    data_path = PROJECT_ROOT / DATA_PATH_PRIMARY
    df = pd.read_csv(data_path)
    X = df[FEATURES]

    pipe = build_clustering_pipeline(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
    pipe.fit(X)

    metrics = evaluate_clustering(pipe, X)
    assert metrics["silhouette_score"] > 0.30, f"Silhouette demasiado bajo: {metrics['silhouette_score']}"
    assert metrics["calinski_harabasz_score"] > 1000.0, f"Calinski-Harabasz deficiente: {metrics['calinski_harabasz_score']}"
    assert metrics["inertia"] > 0, "La inercia debe ser un valor positivo."


def test_model_artifact_contract():
    """Verifica que el archivo model.joblib contenga todos los metadatos necesarios para Streamlit."""
    model_path = PROJECT_ROOT / MODEL_PATH_DEFAULT
    assert model_path.exists(), f"El artefacto {model_path} no existe. Ejecute train.py primero."

    artifact = joblib.load(model_path)
    required_keys = [
        "model", "features", "numeric_features", "task",
        "n_clusters", "cluster_descriptions", "profiles", "metrics",
        "pca_model", "version"
    ]
    for key in required_keys:
        assert key in artifact, f"Clave crítica '{key}' ausente en el artefacto."

    assert artifact["task"] == "clustering"
    assert artifact["n_clusters"] == 3


def test_audit_cases_inference():
    """Verifica que los tres casos de prueba prototípicos del README se asignen al cluster esperado."""
    model_path = PROJECT_ROOT / MODEL_PATH_DEFAULT
    artifact = joblib.load(model_path)
    model = artifact["model"]

    # Caso 1: SMS Champion -> Esperado: Cluster 1
    case_1 = pd.DataFrame([[32.0, 1.0, 4500.0, 65.0, 290.0, 23.0, 28.0, 1350.0]], columns=FEATURES)
    pred_1 = int(model.predict(case_1)[0])
    assert pred_1 == 1, f"Caso 1 esperado Cluster 1, obtenido: {pred_1}"

    # Caso 2: Heavy Caller -> Esperado: Cluster 2
    case_2 = pd.DataFrame([[34.0, 3.0, 11500.0, 160.0, 60.0, 45.0, 35.0, 680.0]], columns=FEATURES)
    pred_2 = int(model.predict(case_2)[0])
    assert pred_2 == 2, f"Caso 2 esperado Cluster 2, obtenido: {pred_2}"

    # Caso 3: Base Estándar -> Esperado: Cluster 0
    case_3 = pd.DataFrame([[30.0, 0.0, 2000.0, 38.0, 15.0, 16.0, 30.0, 150.0]], columns=FEATURES)
    pred_3 = int(model.predict(case_3)[0])
    assert pred_3 == 0, f"Caso 3 esperado Cluster 0, obtenido: {pred_3}"

