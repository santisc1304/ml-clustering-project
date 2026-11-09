"""
Generación de gráficos diagnósticos y perfiles de clustering.
Guarda artefactos visuales en la carpeta graphs/ para documentación y Streamlit.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from .config import RANDOM_SEED, GRAPHS_DIR, N_INIT


def generate_evaluation_plots(X_scaled: np.ndarray, labels: np.ndarray, features: list[str]) -> dict[str, str]:
    """
    Genera y guarda gráficos de codo (inercia), silhouette, proyección PCA y radar.
    """
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    saved_plots = {}

    # 1. Curvas de Inercia y Silhouette para K = 2 hasta 7
    k_range = list(range(2, 8))
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=N_INIT)
        l = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, l))

    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
    color = "#800020"  # Rojo oscuro
    ax1.set_xlabel("Número de Clusters (k)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Inercia (Suma Cuadrados)", color=color, fontsize=11, fontweight="bold")
    ax1.plot(k_range, inertias, marker="o", color=color, linewidth=2.5, markersize=8, label="Inercia")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    color2 = "#6A0DAD"  # Violeta
    ax2.set_ylabel("Coeficiente Silhouette Promedio", color=color2, fontsize=11, fontweight="bold")
    ax2.plot(k_range, silhouettes, marker="s", color=color2, linewidth=2.5, markersize=8, label="Silhouette")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Resaltar k=3
    ax1.axvline(x=3, color="#FFD700", linestyle=":", linewidth=2, label="k Óptimo = 3")

    plt.title("Método del Codo e Índice Silhouette para Selección de k", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    elbow_path = GRAPHS_DIR / "elbow_and_silhouette.png"
    plt.savefig(elbow_path)
    plt.close()
    saved_plots["elbow_and_silhouette"] = str(elbow_path)

    # 2. Proyección PCA Bidimensional con Centroides
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    # Entrenar k=3 para centroides en espacio PCA
    km3 = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=N_INIT)
    km3.fit(X_scaled)
    centroids_pca = pca.transform(km3.cluster_centers_)

    colors = ["#800020", "#6A0DAD", "#C71585"]
    cluster_names = ["Cluster 0 (Estándar)", "Cluster 1 (SMS & High-Value)", "Cluster 2 (Heavy Callers)"]

    plt.figure(figsize=(9, 6), dpi=300)
    for c in range(3):
        mask = (labels == c)
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors[c],
            label=cluster_names[c],
            alpha=0.45,
            edgecolors="none",
            s=35
        )

    plt.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        c="#FFD700",
        s=220,
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        label="Centroides",
        zorder=10
    )

    plt.title(f"Proyección PCA 2D de Clusters (Varianza explicada: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
              fontsize=13, fontweight="bold")
    plt.xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10, fontweight="bold")
    plt.ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10, fontweight="bold")
    plt.legend(frameon=True, facecolor="white", edgecolor="#ddd", loc="best")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    pca_path = GRAPHS_DIR / "pca_clusters_2d.png"
    plt.savefig(pca_path)
    plt.close()
    saved_plots["pca_clusters_2d"] = str(pca_path)

    return saved_plots

