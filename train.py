"""
Entrenamiento y serialización reproducible del modelo de Clustering.
Cumple con la Sección 11 de la Guía Oficial y estándares industriales MLOps.
Ejecución: python train.py [--data datasets/datos.csv] [--output model.joblib]
"""
import argparse
from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np

# Añadir directorio actual al path para importar src
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.config import (
    RANDOM_SEED,
    N_CLUSTERS,
    DATA_PATH_PRIMARY,
    DATA_PATH_FALLBACK,
    MODEL_PATH_DEFAULT,
    FEATURES,
    FEATURE_DESCRIPTIONS,
    CLUSTER_DESCRIPTIONS,
    GRAPHS_DIR
)
from src.clustering_pipeline import (
    build_clustering_pipeline,
    evaluate_clustering,
    compute_cluster_profiles,
    fit_pca_projection
)
from src.evaluate_and_plot import generate_evaluation_plots


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Carga el dataset verificando rutas primarias y de respaldo.
    """
    target_path = data_path
    if not target_path.exists():
        if DATA_PATH_PRIMARY.exists():
            target_path = DATA_PATH_PRIMARY
        elif DATA_PATH_FALLBACK.exists():
            target_path = DATA_PATH_FALLBACK
        else:
            raise FileNotFoundError(f"No se encontró el dataset en {data_path}, {DATA_PATH_PRIMARY} ni {DATA_PATH_FALLBACK}")

    print(f"[1/5] Cargando dataset desde: {target_path}")
    df = pd.read_csv(target_path)
    # Limpieza de espacios en blanco en cabeceras
    df.columns = [c.strip().replace("  ", " ") for c in df.columns]

    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing_cols}")

    print(f"      Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


def train_clustering(df: pd.DataFrame, n_clusters: int, random_state: int) -> dict:
    """
    Ajusta el pipeline, evalúa estructura, perfila clusters y empaqueta el artefacto.
    """
    X = df[FEATURES].copy()

    print(f"[2/5] Construyendo pipeline hermético (Imputer + Scaler + KMeans k={n_clusters})...")
    pipeline = build_clustering_pipeline(n_clusters=n_clusters, random_state=random_state)

    print("[3/5] Entrenando modelo de agrupamiento no supervisado...")
    pipeline.fit(X)

    kmeans = pipeline.named_steps["cluster"]
    labels = kmeans.labels_

    print("[4/5] Evaluando métricas matemáticas de clustering...")
    metrics = evaluate_clustering(pipeline, X)
    print(f"      - Silhouette Score:          {metrics['silhouette_score']:.4f}")
    print(f"      - Calinski-Harabasz Score:   {metrics['calinski_harabasz_score']:.2f}")
    print(f"      - Davies-Bouldin Score:      {metrics['davies_bouldin_score']:.4f}")
    print(f"      - Inercia (Inertia):         {metrics['inertia']:.2f}")

    # Calcular perfiles
    profiles = compute_cluster_profiles(df, labels, FEATURES)
    print("\n--- Perfiles Promedio Descubiertos por Cluster ---")
    print(profiles.to_string())

    # Proyección PCA 2D
    pca_model, X_pca = fit_pca_projection(pipeline, X)

    # Generar gráficos diagnósticos
    X_scaled = pipeline[:-1].transform(X)
    saved_plots = generate_evaluation_plots(X_scaled, labels, FEATURES)
    print(f"[Gráficos generados] {list(saved_plots.keys())} en carpeta '{GRAPHS_DIR}'")

    # Rangos de características para validación e inferencia en Streamlit
    feature_ranges = {}
    for col in FEATURES:
        feature_ranges[col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
            "median": float(X[col].median()),
            "std": float(X[col].std())
        }

    # Empaquetado completo del artefacto
    artifact = {
        "model": pipeline,
        "features": FEATURES,
        "numeric_features": FEATURES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "task": "clustering",
        "n_clusters": n_clusters,
        "cluster_descriptions": CLUSTER_DESCRIPTIONS,
        "profiles": profiles.to_dict(),
        "profiles_df": profiles,
        "metrics": metrics,
        "feature_ranges": feature_ranges,
        "pca_model": pca_model,
        "pca_explained_variance": pca_model.explained_variance_ratio_.tolist(),
        "version": "1.0.0"
    }

    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Script CLI de entrenamiento reproducible para Clustering MLOps.")
    parser.add_argument("--data", type=Path, default=DATA_PATH_PRIMARY, help="Ruta al archivo CSV con los datos.")
    parser.add_argument("--output", type=Path, default=MODEL_PATH_DEFAULT, help="Ruta de destino para model.joblib.")
    parser.add_argument("--clusters", type=int, default=N_CLUSTERS, help="Número de clusters k.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Semilla de aleatoriedad determinista.")
    args = parser.parse_args()

    print("================================================================================")
    print("🚀 INICIANDO ENTRENAMIENTO DETERMINISTA DE CLUSTERING (MLOps RUTA B)")
    print(f"Semilla: {args.seed} | k: {args.clusters}")
    print("================================================================================")

    df = load_data(args.data)
    artifact = train_clustering(df, n_clusters=args.clusters, random_state=args.seed)

    print(f"\n[5/5] Serializando artefacto de inferencia en: {args.output}")
    joblib.dump(artifact, args.output)

    # Verificar existencia y peso del archivo generado
    if args.output.exists():
        size_kb = args.output.stat().st_size / 1024
        print(f"✔ ÉXITO: {args.output} generado correctamente ({size_kb:.1f} KB).")
        print("El modelo está listo para ser servido en Streamlit o empaquetado en Docker.")
    else:
        print("❌ ERROR: No se pudo verificar la creación de model.joblib", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

