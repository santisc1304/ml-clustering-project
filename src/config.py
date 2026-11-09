"""
Configuraciones y constantes globales del proyecto MLOps de Clustering.
Cumple con reproducibilidad matemática y estandarización industrial.
"""
from pathlib import Path

# Semilla de aleatoriedad determinista
RANDOM_SEED: int = 42

# Hiperparámetros óptimos de K-Means seleccionados por Silhouette e Inercia
N_CLUSTERS: int = 3
N_INIT: int = 10
MAX_ITER: int = 300

# Rutas de datos y artefactos
DATA_PATH_PRIMARY: Path = Path("datasets/datos.csv")
MODEL_PATH_DEFAULT: Path = Path("model.joblib")
GRAPHS_DIR: Path = Path("graphs")
DOCS_DIR: Path = Path("docs")

# Definición de variables del dominio de clientes
FEATURES: list[str] = [
    "Subscription Length",
    "Charge Amount",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
    "Age",
    "Customer Value"
]

# Descripciones comprensibles de cada variable
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "Subscription Length": "Antigüedad del cliente en meses",
    "Charge Amount": "Nivel o categoría del cargo mensual facturado",
    "Seconds of Use": "Segundos acumulados de llamadas en el período",
    "Frequency of use": "Número total de llamadas salientes realizadas",
    "Frequency of SMS": "Cantidad total de mensajes de texto SMS enviados",
    "Distinct Called Numbers": "Cantidad de contactos telefónicos únicos llamados",
    "Age": "Edad del titular de la línea en años",
    "Customer Value": "Valor monetario acumulado y margen del cliente"
}

# Perfiles e interpretaciones ejecutivas de negocio validadas
CLUSTER_DESCRIPTIONS: dict[int, dict[str, str]] = {
    0: {
        "title": "Cluster 0: Base Masiva Tradicional (Bajo Consumo / Estándar)",
        "summary": "Clientes con uso básico de servicios móviles. Llamadas moderadas (~38 min), bajo volumen de SMS (~17) y valor monetario contenido (~158 USD). Representan la mayoría del parque de clientes.",
        "action": "Estrategia de retención pasiva y ofertas de upselling orientadas a paquetes de datos o beneficios por permanencia.",
        "badge_color": "#800020"  # Rojo oscuro elegante
    },
    1: {
        "title": "Cluster 1: Digital Champions & High-Value SMS Lovers",
        "summary": "Segmento de alto valor económico (~1,354 USD de Customer Value). Intensivos en mensajería digital y SMS (~288 SMS), consumo de voz medio y comportamiento proactivo.",
        "action": "Atención prioritaria VIP, programas de lealtad exclusivos y preventas de servicios digitales premium.",
        "badge_color": "#6A0DAD"  # Violeta profundo
    },
    2: {
        "title": "Cluster 2: Heavy Callers / Ejecutivos y Negocios de Voz",
        "summary": "Usuarios de voz extremadamente intensiva (~184 minutos de llamadas, 155 llamadas a más de 43 contactos distintos). Alto cargo mensual periódico y valor estratégico crítico.",
        "action": "Planes ilimitados multidestino, roaming corporativo y soporte técnico preferencial de baja latencia.",
        "badge_color": "#C71585"  # Fucsia/Violeta vibrante complementario
    }
}

