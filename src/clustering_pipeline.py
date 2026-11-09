"""
Pipeline reproducible de Clustering utilizando Scikit-Learn.
Encapsula imputación, escalamiento y K-Means en un solo objeto invocable.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from .config import RANDOM_SEED, N_CLUSTERS, N_INIT, MAX_ITER


def build_clustering_pipeline(
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_SEED
) -> Pipeline:
    """
    Construye el pipeline completo de scikit-learn.
    Asegura que el escalado y el estimador no sufran fugas de datos.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("cluster", KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=N_INIT,
            max_iter=MAX_ITER
        ))
    ])
    return pipeline


def evaluate_clustering(
    pipeline: Pipeline,
    X: pd.DataFrame
) -> Dict[str, float]:
    """
    Calcula métricas matemáticas formales de calidad de agrupamiento.
    """
    # Extraer datos preprocesados antes del estimador final
    X_processed = pipeline[:-1].transform(X)
    kmeans: KMeans = pipeline.named_steps["cluster"]
    labels = kmeans.labels_

    sil = float(silhouette_score(X_processed, labels))
    ch = float(calinski_harabasz_score(X_processed, labels))
    db = float(davies_bouldin_score(X_processed, labels))
    inertia = float(kmeans.inertia_)

    return {
        "silhouette_score": sil,
        "calinski_harabasz_score": ch,
        "davies_bouldin_score": db,
        "inertia": inertia
    }


def compute_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: list[str]
) -> pd.DataFrame:
    """
    Calcula medias, medianas y conteos de los clusters descubiertos.
    """
    df_copy = df[features].copy()
    df_copy["cluster"] = labels
    profiles = df_copy.groupby("cluster")[features].mean().round(2)
    counts = df_copy["cluster"].value_counts().sort_index()
    percentages = (counts / len(df_copy) * 100).round(1)
    
    profiles.insert(0, "n_samples", counts)
    profiles.insert(1, "percentage", percentages)
    return profiles


def fit_pca_projection(
    pipeline: Pipeline,
    X: pd.DataFrame
) -> Tuple[PCA, np.ndarray]:
    """
    Ajusta una proyección PCA bidimensional sobre los datos escalados
    para graficación e inspección visual en Streamlit y reportes.
    """
    X_scaled = pipeline[:-1].transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca

