"""
================================================================================
APLICACIÓN STREAMLIT HYPER-VISUAL: SEGMENTACIÓN DE CLIENTES (RUTA B - MLOPS)
================================================================================
Interfaz de usuario de clase mundial con inyección HTML5/CSS3 custom,
paleta de colores Violeta y Rojo Oscuro, animaciones fluidas, y assets
dinámicos e interactivos con Plotly.
"""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Añadir directorio actual al path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Configuración inicial de Streamlit
st.set_page_config(
    page_title="Customer Segmenter | MLOps Cluster Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 1. CARGA SEGURA DEL ARTEFACTO (SIN REENTRENAMIENTO)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_artifact():
    model_path = Path("model.joblib")
    if not model_path.exists():
        fallback_path = CURRENT_DIR / "model.joblib"
        if fallback_path.exists():
            model_path = fallback_path
        else:
            st.error("❌ No se encontró el archivo 'model.joblib'. Por favor ejecuta 'python train.py' primero.")
            st.stop()
    return joblib.load(model_path)

artifact = load_artifact()
pipeline = artifact["model"]
features = artifact["features"]
descriptions = artifact.get("cluster_descriptions", {})
profiles_df = artifact.get("profiles_df", pd.DataFrame())
metrics = artifact.get("metrics", {})
feature_ranges = artifact.get("feature_ranges", {})
pca_model = artifact.get("pca_model", None)
feature_descriptions = artifact.get("feature_descriptions", {})

# ------------------------------------------------------------------------------
# 2. INYECCIÓN DE ESTILOS CSS3 PERSONALIZADOS (PALETA VIOLETA & ROJO OSCURO)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Fira+Code:wght@400;600&display=swap');

/* Reset y tipografía global */
* {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Fondo dinámico con gradiente animado en malla */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(106, 13, 173, 0.25) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(128, 0, 32, 0.30) 0%, transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(74, 14, 23, 0.20) 0%, transparent 60%),
                #09050e;
    background-attachment: fixed;
    color: #f3e8f8;
}

/* Animaciones Keyframes */
@keyframes pulseGlow {
    0% { box-shadow: 0 0 15px rgba(106, 13, 173, 0.4), inset 0 0 10px rgba(128, 0, 32, 0.3); }
    50% { box-shadow: 0 0 30px rgba(138, 43, 226, 0.7), inset 0 0 20px rgba(199, 21, 133, 0.5); }
    100% { box-shadow: 0 0 15px rgba(106, 13, 173, 0.4), inset 0 0 10px rgba(128, 0, 32, 0.3); }
}

@keyframes floatCard {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
    100% { transform: translateY(0px); }
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Header Banner Estilizado */
.hero-container {
    background: linear-gradient(135deg, rgba(74, 14, 23, 0.85) 0%, rgba(106, 13, 173, 0.85) 50%, rgba(20, 6, 28, 0.95) 100%);
    border: 1px solid rgba(138, 43, 226, 0.4);
    border-radius: 20px;
    padding: 30px 35px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
    position: relative;
    overflow: hidden;
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #FFFFFF, #E0AAFF, #FFB4C8, #FFD700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    color: #e2cce9;
    font-size: 1.05rem;
    margin-top: 8px;
    margin-bottom: 15px;
    max-width: 850px;
}

/* Badges de estado en la cabecera */
.status-badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-right: 8px;
    text-transform: uppercase;
}
.badge-active {
    background: rgba(106, 13, 173, 0.4);
    border: 1px solid #9D4EDD;
    color: #E0AAFF;
}
.badge-perf {
    background: rgba(128, 0, 32, 0.4);
    border: 1px solid #FF4D6D;
    color: #FFB4C8;
}
.badge-k {
    background: rgba(255, 215, 0, 0.15);
    border: 1px solid #FFD700;
    color: #FFD700;
}

/* Tarjetas Glassmorphism */
.glass-card {
    background: rgba(26, 11, 33, 0.75);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(138, 43, 226, 0.25);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    border-color: rgba(224, 170, 255, 0.5);
    box-shadow: 0 8px 25px rgba(106, 13, 173, 0.3);
}

/* Tarjeta destacada de resultado */
.result-card-active {
    background: linear-gradient(145deg, rgba(74, 14, 23, 0.9) 0%, rgba(106, 13, 173, 0.85) 100%);
    border: 2px solid #FFD700;
    border-radius: 20px;
    padding: 28px;
    animation: pulseGlow 4s infinite ease-in-out;
    margin-bottom: 25px;
}

/* Botón principal estilizado */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #800020 0%, #6A0DAD 50%, #C71585 100%);
    background-size: 200% auto;
    color: #FFFFFF;
    font-size: 1.1rem;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 14px 28px;
    width: 100%;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(106, 13, 173, 0.5);
    transition: all 0.4s ease;
}
div.stButton > button:first-child:hover {
    background-position: right center;
    box-shadow: 0 6px 30px rgba(199, 21, 133, 0.8);
    transform: translateY(-2px);
}

/* Pipeline Flow DAG en HTML */
.pipeline-flow {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(18, 7, 24, 0.85);
    border: 1px solid rgba(138, 43, 226, 0.25);
    border-radius: 14px;
    padding: 15px 20px;
    margin-bottom: 25px;
    overflow-x: auto;
}
.flow-step {
    text-align: center;
    padding: 8px 12px;
    background: rgba(45, 14, 58, 0.6);
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #E0AAFF;
    border: 1px solid rgba(157, 78, 221, 0.3);
}
.flow-arrow {
    color: #FF4D6D;
    font-size: 1.2rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. HERO HEADER DINÁMICO
# ------------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span class="status-badge badge-active">⚡ MLOps Pipeline Ready</span>
                <span class="status-badge badge-perf">Silhouette: 0.3536</span>
                <span class="status-badge badge-k">k = 3 Clusters</span>
            </div>
            <h1 class="hero-title">🔮 Neural Customer Segmentation Engine</h1>
            <p class="hero-subtitle">
                Plataforma de agrupamiento inteligente no supervisado basada en <strong>K-Means</strong>. 
                Permite perfilar clientes de telecomunicaciones en tiempo real y desplegar estrategias hiperpersonalizadas.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# EXPLICACIÓN SENCILLA Y DIDÁCTICA: ¿QUÉ HICIMOS Y CÓMO FUNCIONA?
# ------------------------------------------------------------------------------
with st.expander("📖 ¿Cómo funciona esta aplicación? Explicación sencilla paso a paso (Haz clic para ver)", expanded=True):
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.markdown("""
        #### 🎯 1. ¿Qué problema resuelve este sistema?
        En las empresas de telefonía y servicios digitales hay miles de clientes con hábitos completamente distintos:
        * Unos casi no llaman pero envían cientos de mensajes y consumen servicios de alto valor.
        * Otros usan el teléfono como herramienta de trabajo intensiva (más de 3 horas de llamadas).
        * Y una gran mayoría tiene un consumo básico y esporádico.
        
        **Si a todos les ofrecemos la misma tarifa, perdemos dinero o los clientes se van.** Este sistema agrupa a los clientes automáticamente para que el equipo comercial sepa exactamente **qué plan o beneficio ofrecer a cada persona**.
        """)
        
        st.markdown("""
        #### 🧠 2. ¿Qué modelo usamos y por qué?
        Seguimos la **Ruta B (Aprendizaje No Supervisado)** con el algoritmo **K-Means**:
        * **Sin respuestas previas:** No le dijimos al modelo quién era quién; el algoritmo analizó a 3,150 clientes y descubrió de forma natural que existen **3 tribus o perfiles** bien diferenciados.
        * **Fórmula de distancia:** Cuando ingresas un cliente, el modelo mide la "distancia matemática" de sus hábitos contra cada tribu y lo asigna a la más cercana en milisegundos.
        """)

    with col_exp2:
        st.markdown("""
        #### ⚖️ 3. El Secreto del Preprocesamiento (Por qué no falla)
        Si comparamos directamente **Segundos de Uso** (que son miles, ej: 11,000 s) con la **Edad** (decenas, ej: 30 años), la matemática se confundiría y creería que los segundos son 400 veces más importantes solo por ser números más grandes.
        * Por eso construimos un **Pipeline de Scikit-Learn** con `StandardScaler`, que normaliza todas las variables a una misma escala justa.
        * Además, incluimos `SimpleImputer` para que si en el futuro falta un dato, la aplicación no se caiga y use un valor de respaldo inteligente.
        """)

        st.markdown("""
        #### 🚀 4. ¿Cómo usar esta pantalla?
        1. **Elige un ejemplo rápido** en el menú de la izquierda (como *SMS Champion* o *Heavy Caller*) o mueve los controles a tu gusto.
        2. Presiona el botón grande **Ejecutar Algoritmo de Segmentación**.
        3. El sistema te mostrará a qué grupo pertenece, qué acción comercial tomar y 4 gráficos interactivos para auditar el resultado visualmente.
        """)

# ------------------------------------------------------------------------------
# 4. PIPELINE DAG INTERACTIVO (VISUALIZACIÓN DEL PROCESO EN TIEMPO REAL)
# ------------------------------------------------------------------------------
st.markdown("""
<div class="pipeline-flow">
    <div class="flow-step">📥 1. Entrada de Variables</div>
    <div class="flow-arrow">➔</div>
    <div class="flow-step">🔧 2. SimpleImputer (Mediana)</div>
    <div class="flow-arrow">➔</div>
    <div class="flow-step">⚖️ 3. StandardScaler (Z-Score)</div>
    <div class="flow-arrow">➔</div>
    <div class="flow-step">🎯 4. KMeans Inferencia (k=3)</div>
    <div class="flow-arrow">➔</div>
    <div class="flow-step">🧭 5. Asignación de Perfil & Acción</div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 5. BARRA LATERAL: PRESETS DE AUDITORÍA Y CONTROL
# ------------------------------------------------------------------------------
st.sidebar.markdown("""
<h3 style="color: #E0AAFF; margin-top:0;">🎛️ Panel de Auditoría Rápida</h3>
<p style="font-size: 0.85rem; color: #d0bcd5;">
Cargue casos prototípicos preconfigurados para verificar la reproducibilidad de la segmentación:
</p>
""", unsafe_allow_html=True)

preset = st.sidebar.radio(
    "Seleccionar Escenario:",
    [
        "Personalizado (Ajuste Manual)",
        "Caso 1: SMS Champion (Segmento 1)",
        "Caso 2: Heavy Caller Corporativo (Segmento 2)",
        "Caso 3: Base Estándar (Segmento 0)"
    ]
)

# Valores por defecto según preset
if preset == "Caso 1: SMS Champion (Segmento 1)":
    defaults = {
        "Subscription Length": 32.0,
        "Charge Amount": 1.0,
        "Seconds of Use": 4500.0,
        "Frequency of use": 65.0,
        "Frequency of SMS": 290.0,
        "Distinct Called Numbers": 23.0,
        "Age": 28.0,
        "Customer Value": 1350.0
    }
elif preset == "Caso 2: Heavy Caller Corporativo (Segmento 2)":
    defaults = {
        "Subscription Length": 34.0,
        "Charge Amount": 3.0,
        "Seconds of Use": 11500.0,
        "Frequency of use": 160.0,
        "Frequency of SMS": 60.0,
        "Distinct Called Numbers": 45.0,
        "Age": 35.0,
        "Customer Value": 680.0
    }
elif preset == "Caso 3: Base Estándar (Segmento 0)":
    defaults = {
        "Subscription Length": 30.0,
        "Charge Amount": 0.0,
        "Seconds of Use": 2000.0,
        "Frequency of use": 38.0,
        "Frequency of SMS": 15.0,
        "Distinct Called Numbers": 16.0,
        "Age": 30.0,
        "Customer Value": 150.0
    }
else:
    defaults = {f: float(feature_ranges.get(f, {}).get("median", 50.0)) for f in features}

# ------------------------------------------------------------------------------
# 6. FORMULARIO EN COLUMNAS GLASSMORFICAS
# ------------------------------------------------------------------------------
st.markdown("### 📝 Ingreso de Características del Cliente")

col_left, col_right = st.columns(2)

input_data = {}

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color: #FFB4C8; margin-top:0;'>📞 Consumo de Telefonía y Tráfico</h4>", unsafe_allow_html=True)

    input_data["Seconds of Use"] = st.slider(
        "Segundos de Uso Acumulados (Voz):",
        min_value=0.0,
        max_value=17000.0,
        value=float(defaults.get("Seconds of Use", 2300.0)),
        step=50.0,
        help="Duración total de llamadas en segundos."
    )

    input_data["Frequency of use"] = st.slider(
        "Frecuencia de Llamadas (Salientes):",
        min_value=0.0,
        max_value=255.0,
        value=float(defaults.get("Frequency of use", 40.0)),
        step=1.0,
        help="Número total de llamadas efectuadas."
    )

    input_data["Distinct Called Numbers"] = st.slider(
        "Contactos Telefónicos Distintos:",
        min_value=0.0,
        max_value=100.0,
        value=float(defaults.get("Distinct Called Numbers", 18.0)),
        step=1.0,
        help="Amplitud de la red de contactos."
    )

    input_data["Frequency of SMS"] = st.slider(
        "Mensajes SMS Enviados:",
        min_value=0.0,
        max_value=550.0,
        value=float(defaults.get("Frequency of SMS", 20.0)),
        step=5.0,
        help="Volumen total de SMS en el período."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color: #E0AAFF; margin-top:0;'>💼 Perfil Comercial y Contractual</h4>", unsafe_allow_html=True)

    input_data["Customer Value"] = st.number_input(
        "Customer Value (Margen Acumulado USD):",
        min_value=0.0,
        max_value=2200.0,
        value=float(defaults.get("Customer Value", 160.0)),
        step=10.0,
        help="Valor monetario calculado por la compañía."
    )

    input_data["Subscription Length"] = st.number_input(
        "Antigüedad de la Suscripción (Meses):",
        min_value=1.0,
        max_value=60.0,
        value=float(defaults.get("Subscription Length", 32.0)),
        step=1.0,
        help="Meses ininterrumpidos en la operadora."
    )

    input_data["Charge Amount"] = st.selectbox(
        "Nivel de Cargo Mensual Facturado:",
        options=list(range(0, 11)),
        index=int(defaults.get("Charge Amount", 0)),
        help="Categoría del cargo periódico."
    )

    input_data["Age"] = st.number_input(
        "Edad del Titular:",
        min_value=15.0,
        max_value=85.0,
        value=float(defaults.get("Age", 30.0)),
        step=1.0
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 7. EJECUCIÓN DEL MODELO DE INFERENCIA
# ------------------------------------------------------------------------------
execute_btn = st.button("🚀 Ejecutar Algoritmo de Segmentación")

if execute_btn or preset != "Personalizado (Ajuste Manual)":
    # Construcción del DataFrame para inferencia
    df_new = pd.DataFrame([input_data])[features]

    # Inferencia usando el pipeline serializado
    predicted_cluster = int(pipeline.predict(df_new)[0])

    # Cálculo de distancias a los centroides para estimar afinidad relativa
    X_scaled_new = pipeline[:-1].transform(df_new)
    kmeans_step = pipeline.named_steps["cluster"]
    centroids = kmeans_step.cluster_centers_

    # Distancias euclidianas a cada centroide
    distances = np.linalg.norm(centroids - X_scaled_new, axis=1)
    inv_distances = 1.0 / (distances + 1e-5)
    affinities = (inv_distances / inv_distances.sum()) * 100
    assigned_affinity = float(affinities[predicted_cluster])

    cluster_info = descriptions.get(predicted_cluster, {
        "title": f"Cluster {predicted_cluster}",
        "summary": "Perfil asignado por K-Means.",
        "action": "Estrategia estándar de servicio.",
        "badge_color": "#6A0DAD"
    })

    # Proyección PCA del nuevo punto
    if pca_model is not None:
        new_point_pca = pca_model.transform(X_scaled_new)[0]
    else:
        new_point_pca = [0.0, 0.0]

    st.markdown("---")
    st.markdown("## 🎯 Resultado de la Segmentación")

    # Tarjeta de Resultado Principal Destacada
    st.markdown(f"""
    <div class="result-card-active" style="border-left: 8px solid {cluster_info.get('badge_color', '#FFD700')};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span class="status-badge" style="background: rgba(255,215,0,0.2); border: 1px solid #FFD700; color: #FFD700;">
                    Asignación Confirmada
                </span>
                <h2 style="color: #FFFFFF; margin: 10px 0 6px 0; font-size: 1.8rem;">
                    {cluster_info['title']}
                </h2>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.85rem; color: #E0AAFF; text-transform: uppercase;">Afinidad al Centroide</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #FFD700;">
                    {assigned_affinity:.1f}%
                </div>
            </div>
        </div>
        <div style="margin-top: 15px; font-size: 1.05rem; line-height: 1.6; color: #f5e9f8;">
            <strong>Descripción del Comportamiento:</strong> {cluster_info['summary']}
        </div>
        <div style="margin-top: 12px; padding: 12px 16px; background: rgba(0, 0, 0, 0.35); border-radius: 10px; border-left: 4px solid #FF4D6D;">
            <strong style="color: #FFB4C8;">💡 Recomendación Táctica para la Empresa:</strong>
            <span style="color: #ffffff;"> {cluster_info['action']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 8. ASSETS GRÁFICOS DINÁMICOS E INTERACTIVOS (PLOTLY)
    # --------------------------------------------------------------------------
    tab_radar, tab_pca, tab_gauge, tab_table = st.tabs([
        "🕸️ Radar Multidimensional",
        "🌌 Proyección Espacial PCA 2D",
        "⏱️ Tacómetro de Afinidad",
        "📊 Perfiles Comparativos"
    ])

    with tab_radar:
        st.markdown("#### 📡 Comparativa Multidimensional: Caso Actual vs. Centroides")
        st.info("💡 **¿Cómo interpretar este gráfico de telaraña?** Cada vértice es una variable. La silueta dorada con relleno representa al cliente que acabas de ingresar. Las líneas punteadas de colores muestran el 'cliente promedio' de cada una de las 3 tribus. Si la figura dorada se estira hacia 'Customer Value' o 'Frequency of SMS', verás de un solo vistazo qué rasgo lo hace único.")

        radar_features = [
            "Seconds of Use",
            "Frequency of use",
            "Frequency of SMS",
            "Distinct Called Numbers",
            "Customer Value",
            "Subscription Length"
        ]

        # Normalización min-max para graficación de radar limpia (0 a 100)
        norm_case = []
        for feat in radar_features:
            f_min = feature_ranges[feat]["min"]
            f_max = feature_ranges[feat]["max"]
            val = (input_data[feat] - f_min) / (f_max - f_min + 1e-6) * 100
            norm_case.append(val)
        norm_case.append(norm_case[0])  # Cerrar el lazo

        fig_radar = go.Figure()

        # Polígono del cliente evaluado
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_case,
            theta=radar_features + [radar_features[0]],
            fill='toself',
            name='Caso Evaluado',
            line=dict(color='#FFD700', width=3),
            fillcolor='rgba(255, 215, 0, 0.35)'
        ))

        # Añadir centroides de los clusters
        cluster_palette = ["#800020", "#9D4EDD", "#FF1493"]
        cluster_labels = ["Cluster 0 (Estándar)", "Cluster 1 (SMS Champion)", "Cluster 2 (Heavy Caller)"]

        for c_idx in range(3):
            centroid_vals = []
            for feat in radar_features:
                f_min = feature_ranges[feat]["min"]
                f_max = feature_ranges[feat]["max"]
                c_mean = profiles_df.loc[c_idx, feat] if not profiles_df.empty else 0
                val = (c_mean - f_min) / (f_max - f_min + 1e-6) * 100
                centroid_vals.append(val)
            centroid_vals.append(centroid_vals[0])

            fig_radar.add_trace(go.Scatterpolar(
                r=centroid_vals,
                theta=radar_features + [radar_features[0]],
                name=cluster_labels[c_idx],
                line=dict(color=cluster_palette[c_idx], dash='dot', width=2),
                fill='none'
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#E0AAFF", gridcolor="rgba(157, 78, 221, 0.2)"),
                angularaxis=dict(color="#f3e8f8", gridcolor="rgba(157, 78, 221, 0.2)")
            ),
            paper_bgcolor='rgba(15, 6, 20, 0.7)',
            plot_bgcolor='rgba(15, 6, 20, 0.7)',
            font=dict(color="#f3e8f8"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=30, b=40),
            height=460
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab_pca:
        st.markdown("#### 🌌 Mapa de Densidad y Ubicación del Cliente en el Espacio PCA")
        st.info("💡 **¿Cómo entender este mapa de galaxias?** Como los humanos no podemos ver 8 dimensiones al tiempo, usamos **PCA** para comprimir las 8 variables en un plano 2D sin perder su estructura esencial. Cada puntito es uno de los 3,150 clientes reales. La **estrella dorada grande (⭐)** es el cliente que estás evaluando en este instante: fíjate visualmente junto a qué nube de clientes cayó.")

        # Cargar datos para el fondo PCA de manera robusta
        dataset_path = CURRENT_DIR / "datasets" / "datos.csv"
        if not dataset_path.exists():
            dataset_path = Path("datasets/datos.csv")
            if not dataset_path.exists():
                dataset_path = Path("data/datos.csv")
        df_background = pd.read_csv(dataset_path)
        X_bg = df_background[features]
        X_bg_scaled = pipeline[:-1].transform(X_bg)
        bg_pca = pca_model.transform(X_bg_scaled)
        bg_labels = pipeline.named_steps["cluster"].labels_

        fig_pca = go.Figure()

        # Nube de puntos de cada cluster
        for c in range(3):
            mask = (bg_labels == c)
            fig_pca.add_trace(go.Scatter(
                x=bg_pca[mask, 0],
                y=bg_pca[mask, 1],
                mode='markers',
                name=cluster_labels[c],
                marker=dict(size=5, color=cluster_palette[c], opacity=0.45)
            ))

        # Centroides
        centroids_pca = pca_model.transform(centroids)
        fig_pca.add_trace(go.Scatter(
            x=centroids_pca[:, 0],
            y=centroids_pca[:, 1],
            mode='markers+text',
            name='Centroides',
            text=["C0", "C1", "C2"],
            textposition="top center",
            marker=dict(size=14, color='#FFFFFF', symbol='x', line=dict(width=2, color='#000000'))
        ))

        # NUEVO PUNTO EVALUADO (DESTACADO)
        fig_pca.add_trace(go.Scatter(
            x=[new_point_pca[0]],
            y=[new_point_pca[1]],
            mode='markers+text',
            name='⭐ CLIENTE EVALUADO',
            text=["📍 AQUÍ ESTÁ EL CLIENTE"],
            textposition="bottom center",
            marker=dict(size=22, color='#FFD700', symbol='star', line=dict(width=2, color='#800020'))
        ))

        fig_pca.update_layout(
            paper_bgcolor='rgba(15, 6, 20, 0.7)',
            plot_bgcolor='rgba(15, 6, 20, 0.7)',
            font=dict(color="#f3e8f8"),
            xaxis=dict(title=f"PC 1 ({pca_model.explained_variance_ratio_[0]*100:.1f}% varianza)", gridcolor="rgba(157, 78, 221, 0.15)"),
            yaxis=dict(title=f"PC 2 ({pca_model.explained_variance_ratio_[1]*100:.1f}% varianza)", gridcolor="rgba(157, 78, 221, 0.15)"),
            height=460,
            margin=dict(l=30, r=30, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with tab_gauge:
        st.markdown("#### ⏱️ Tacómetro de Afinidad y Distancia al Centroide")
        st.info("💡 **¿Qué nos indica la aguja?** Mide qué tan 'puro' o prototípico es el cliente dentro de su grupo. Un valor mayor al 80% indica que está muy cerca del centro de gravedad de su tribu (un cliente arquetípico). Un valor menor al 60% indica que está en una frontera o zona de transición entre dos grupos.")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=assigned_affinity,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Afinidad con {cluster_info['title']}", 'font': {'size': 20, 'color': '#E0AAFF'}},
            delta={'reference': 50.0, 'increasing': {'color': "#00E676"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#f3e8f8"},
                'bar': {'color': "#FFD700"},
                'bgcolor': "rgba(30, 10, 40, 0.5)",
                'borderwidth': 2,
                'bordercolor': "#800020",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(128, 0, 32, 0.6)'},
                    {'range': [40, 75], 'color': 'rgba(106, 13, 173, 0.6)'},
                    {'range': [75, 100], 'color': 'rgba(199, 21, 133, 0.6)'}
                ],
                'threshold': {
                    'line': {'color': "#FFFFFF", 'width': 4},
                    'thickness': 0.75,
                    'value': assigned_affinity
                }
            }
        ))

        fig_gauge.update_layout(
            paper_bgcolor='rgba(15, 6, 20, 0.7)',
            font=dict(color="#f3e8f8"),
            height=380,
            margin=dict(l=30, r=30, t=40, b=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with tab_table:
        st.markdown("#### 📋 Matriz de Perfiles Medios de la Población")
        st.info("💡 **Foto panorámica de la empresa:** Esta tabla contiene los valores promedio reales calculados sobre los 3,150 clientes. Puedes ver el tamaño de cada grupo (`percentage`) y cómo difieren drásticamente en volumen de minutos, SMS y valor comercial (`Customer Value`).")
        st.dataframe(
            profiles_df.style.format("{:.2f}").background_gradient(cmap="Purples"),
            use_container_width=True
        )

# ------------------------------------------------------------------------------
# 9. PIE DE PÁGINA TECNOLÓGICO
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a48cb1; font-size: 0.85rem; padding: 10px 0;">
    🔮 <strong>Industrial MLOps Suite</strong> | Arquitectura Limpia & Reproducibilidad Matemática 
    <br>Desplegado con Docker & Google Cloud Compute Engine | Estándar de Clase Mundial
</div>
""", unsafe_allow_html=True)
