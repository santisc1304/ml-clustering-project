# 🔮 Neural Customer Segmentation Engine | MLOps Suite
## Segmentación de Clientes y Serving Productivo con Streamlit, Docker y Google Cloud Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E.svg)](https://scikit-learn.org/)
[![GCP](https://img.shields.io/badge/GCP-Compute%20Engine-4285F4.svg)](https://cloud.google.com/compute)

---

## 📌 Mapa de Entregables del Proyecto

Para facilitar la revisión por parte del docente y la realización de la **Prueba Cruzada** con otra pareja/evaluador, a continuación se detalla la ubicación exacta de cada entregable exigido en la guía oficial:

| Entregable Oficial | Ubicación en el Repositorio | Descripción / Estado |
|---|---|---|
| **Dataset Oficial** | [`datasets/datos.csv`](datasets/datos.csv) | 3,150 registros x 8 variables de uso y valor. Única fuente de datos del proyecto. |
| **Pipeline & Entrenamiento CLI** | [`train.py`](train.py) | Script autónomo y determinista (semilla 42, k=3, métricas Silhouette/Inercia). |
| **Artefacto Serializado** | [`model.joblib`](model.joblib) | Objeto binario inmutable: Pipeline hermético + PCA + Perfiles + Metadatos. |
| **Aplicación Web Interactiva** | [`app.py`](app.py) | Streamlit Hyper-Visual con HTML5/CSS (Violeta/Rojo Oscuro) y gráficos Plotly. |
| **Dependencias Pinned** | [`requirements.txt`](requirements.txt) | Librerías con versiones compatibles y verificadas. |
| **Empaquetado Docker** | [`Dockerfile`](Dockerfile) y [`.dockerignore`](.dockerignore) | Imagen de producción segura (usuario no-root `appuser` y healthcheck). |
| **Documentación Oficial** | [`README.md`](README.md) | Guía técnica completa con instrucciones de ejecución y reproducción. |
| **Notebook para Google Colab** | [`notebook_colab_reto_ml.ipynb`](notebook_colab_reto_ml.ipynb) | Flujo paso a paso ejecutable en Google Colab con un solo clic. |
| **Gráficos Diagnósticos** | [`graphs/`](graphs/) | Figuras en alta resolución: Método del Codo, Silhouette y Proyección PCA 2D. |
| **Interpretación de Negocio** | [`docs/cluster_business_interpretation.md`](docs/cluster_business_interpretation.md) | Análisis detallado de los 3 segmentos descubiertos y acciones tácticas. |
| **Protocolo de Auditoría Ciega** | [`docs/reproducibility_peer_audit.md`](docs/reproducibility_peer_audit.md) | Matriz de verificación y registro para la prueba cruzada. |
| **Evidencia de Auditoría Cruzada (Pareja)** | [`resena_trabajo_santiago_sandoval.md`](resena_trabajo_santiago_sandoval.md) | Dictamen formal emitido por la pareja evaluadora (Miguel Moreno & Julián Duarte) con nota 5.0/5.0. |
| **Preguntas de Sustentación** | [`docs/preguntas_discusion_final.md`](docs/preguntas_discusion_final.md) | Respuestas técnicas y conceptuales a las 12 preguntas de la Sección 35. |
| **Suite de Pruebas Unitarias** | [`tests/test_pipeline_and_inference.py`](tests/test_pipeline_and_inference.py) | 5 pruebas automatizadas con `pytest` (**5/5 pasadas en verde**). |

---

## 1. Título y Objetivo del Proyecto
**Plataforma Integral de Segmentación No Supervisada y Despliegue MLOps**.
El objetivo de este proyecto es construir, validar, empaquetar y desplegar una solución completa de Machine Learning para segmentar automáticamente clientes de telecomunicaciones según sus patrones de uso y valor comercial. La solución permite al equipo comercial y de marketing clasificar en tiempo real a nuevos usuarios mediante una interfaz analítica de vanguardia, facilitando la toma de decisiones basada en datos.

---

## 2. Descripción del Dataset y Unidad de Observación
- **Dataset:** Registro de Comportamiento y Tráfico de Clientes Móviles ([`datasets/datos.csv`](datasets/datos.csv)).
- **Población:** 3,150 clientes individuales observados durante un período de facturación completo.
- **Unidad de Observación:** Un cliente individual o línea telefónica activa.
- **Procedencia:** Repositorio público de comportamiento y consumo de clientes de telecomunicaciones.

---

## 3. Ruta Elegida: Ruta B (Aprendizaje No Supervisado)
- **Pregunta de Negocio:** *¿Podemos descubrir perfiles y comportamientos latentes de clientes sin una etiqueta previa, para diseñar ofertas y planes a la medida?*
- **Paradigma:** Agrupamiento no supervisado con K-Means y proyecciones espaciales en PCA.

---

## 4. Variables Utilizadas
El modelo opera sobre 8 características cuantitativas rigurosamente seleccionadas para evitar correlaciones espurias o sesgos:

| Variable | Tipo | Unidad | Descripción Funcional |
|---|:---:|:---:|---|
| `Subscription Length` | Numérica | Meses | Antigüedad ininterrumpida del cliente en la empresa. |
| `Charge Amount` | Ordinal | Nivel (0-10) | Nivel o categoría del cargo mensual facturado. |
| `Seconds of Use` | Numérica | Segundos | Consumo total acumulado de tráfico de voz. |
| `Frequency of use` | Numérica | Entero | Número total de llamadas salientes realizadas. |
| `Frequency of SMS` | Numérica | Entero | Cantidad total de mensajes de texto SMS emitidos. |
| `Distinct Called Numbers`| Numérica | Entero | Cantidad de contactos telefónicos únicos alcanzados. |
| `Age` | Numérica | Años | Edad cronológica del titular de la cuenta. |
| `Customer Value` | Numérica | USD | Margen y valor económico acumulado por el cliente. |

---

## 5. Preparación de Datos y Arquitectura de Pipeline
Siguiendo los principios de **Clean Architecture** y prevención de fuga de información (*zero data-leakage*), todo el preprocesamiento está encapsulado en un único objeto `Pipeline` de scikit-learn:

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('cluster', KMeans(n_clusters=3, random_state=42, n_init=10))
])
```
- **Imputación:** Se utiliza la mediana para garantizar robustez ante posibles valores atípicos.
- **Estandarización:** Escalado Z-Score ($\mu=0, \sigma=1$) indispensable en algoritmos basados en distancia euclidiana como K-Means.

---

## 6. Modelo y Justificación Técnica
Se seleccionó **K-Means** con $k=3$ debido a:
1. **Capacidad de Inferencia Productiva:** Permite clasificar nuevos casos en tiempo real mediante el método `predict()`, calculando la distancia mínima al centroide más cercano en tiempo $\mathcal{O}(k \cdot d)$.
2. **Determinismo y Estabilidad:** Semilla fijada (`random_state=42`) y $n\_init=10$ ejecuciones con centroides iniciales mediante `k-means++`.
3. **Consistencia Teórica:** Validado mediante el análisis conjunto de la curva de inercia (método del codo) y el coeficiente Silhouette.

---

## 7. Métricas y Criterios Matemáticos de Evaluación
El agrupamiento fue auditado mediante métricas de estructura interna:

- **Silhouette Score:** `0.3536` (Pico óptimo en $k=3$; sin observaciones con coeficientes negativos severos).
- **Calinski-Harabasz Index:** `1203.58` (Alta separación entre grupos y compacidad interna).
- **Davies-Bouldin Index:** `1.2517` (Baja similitud media entre clusters).
- **Inercia Intra-Cluster:** `14278.35`.

Gráficos diagnósticos generados automáticamente en `graphs/`:
- `graphs/elbow_and_silhouette.png`
- `graphs/pca_clusters_2d.png`

---

## 8. Interpretación de la Salida y Perfiles Descubiertos

```
Perfiles Promedio Descubiertos:
cluster   n_samples  % Base   Seconds of Use  Freq Use  Freq SMS  Distinct Numbers  Age    Customer Value  Cargo
0 (Base)      1938   61.5%           2,298 s      41.4      16.8              16.9  30.6       159.30 USD   0.45
1 (SMS VIP)    562   17.8%           4,277 s      65.8     281.8              22.7  28.6     1,334.92 USD   1.09
2 (Heavy Call) 650   20.6%          11,126 s     156.2      60.8              43.8  34.1       653.25 USD   2.28
```

- **Cluster 0: Base Masiva Tradicional (Bajo Consumo / Estándar)**
  - *Perfil:* Clientes con uso funcional y esporádico. Bajo volumen de minutos (~38 min) y SMS (~17).
  - *Acción Comercial:* Campañas de retención automatizadas y cross-selling de paquetes de datos.
- **Cluster 1: Digital Champions & High-Value SMS Lovers**
  - *Perfil:* Clientes jóvenes, altamente digitales y con el mayor Customer Value de toda la empresa (~1,335 USD).
  - *Acción Comercial:* Fidelización VIP exclusiva, planes ilimitados de mensajería y recompensas digitales.
- **Cluster 2: Heavy Callers / Ejecutivos y Negocios de Voz**
  - *Perfil:* Consumo masivo de voz (> 3 horas mensuales) y red amplia de contactos (> 43 números distintos).
  - *Acción Comercial:* Planes corporativos multidestino, roaming preferencial y soporte técnico Gold Tier.

---

## 9. Cómo Ejecutar el Reentrenamiento (`train.py`)
El script es 100% reproducible y autónomo desde la línea de comandos:

```bash
# Ejecución estándar leyendo directamente de datasets/datos.csv
python train.py

# O con parámetros personalizados
python train.py --data datasets/datos.csv --output model.joblib --clusters 3 --seed 42
```
*Salida:* Imprime métricas, genera gráficos en `graphs/` y serializa el artefacto inmutable `model.joblib`.

---

## 10. Cómo Ejecutar la Aplicación Streamlit Localmente
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Iniciar el servidor
streamlit run app.py
```
Abra en su navegador: `http://localhost:8501`.

---

## 11. Cómo Construir la Imagen Docker
El Dockerfile implementa un contenedor seguro con usuario sin privilegios (*non-root*) y healthcheck integrado:

```bash
docker build -t ml-clustering-app:v1.0 .
```

Verifique la imagen construida:
```bash
docker images
```

---

## 12. Cómo Ejecutar el Contenedor Docker
Ejecute el contenedor en segundo plano con reinicio automático ante caídas del sistema:

```bash
docker run -d \
  --name ml-clustering \
  --restart unless-stopped \
  -p 8501:8501 \
  ml-clustering-app:v1.0
```

Verifique el estado y los logs:
```bash
docker ps
docker logs -f ml-clustering
```

---

## 13. Protocolo de Validación Cruzada entre Parejas (Sección 31 de la Guía)

Para cumplir de forma literal con la **Sección 31 de la Guía Oficial**, cualquier compañero, pareja evaluadora o auditor externo debe reproducir este proyecto siguiendo **estrictamente este flujo de 7 pasos**, sin instrucciones verbales adicionales:

```mermaid
flowchart TD
    A["1. Recibir únicamente la URL del repositorio"] --> B["2. Clonar el repositorio"]
    B --> C["3. Leer README sin instrucciones verbales"]
    C --> D["4. Construir la imagen Docker"]
    D --> E["5. Ejecutar el contenedor"]
    E --> F["6. Probar al menos un caso de prueba"]
    F --> G["7. Registrar resultado y observaciones en el formato"]
```

### Guía de Ejecución de los 7 Pasos del Protocolo:

#### Paso 1: Recibir únicamente la URL del repositorio
* URL oficial del proyecto: `https://github.com/santisc1304/ml-clustering-project.git`

#### Paso 2: Clonar el repositorio
Abra una terminal en un entorno limpio (Linux, macOS, WSL o VM) y ejecute:
```bash
git clone https://github.com/santisc1304/ml-clustering-project.git
cd ml-clustering-project
```

#### Paso 3: Leer el README sin instrucciones verbales adicionales
Todo el procedimiento, variables y casos de prueba están auto-contenidos en este documento. *(Opcionalmente, puede verificar la reproducibilidad del pipeline corriendo `python train.py` o ejecutando las pruebas con `python -m pytest tests/ -v`).*

#### Paso 4: Construir la imagen Docker
Ejecute la construcción de la imagen en el directorio raíz del proyecto:
```bash
docker build -t ml-clustering-app:v1.0 .
```

#### Paso 5: Ejecutar el contenedor
Ponga a rodar el contenedor en segundo plano con el puerto mapeado:
```bash
docker run -d --name ml-clustering -p 8501:8501 --restart unless-stopped ml-clustering-app:v1.0
```
Verifique que esté activo:
```bash
docker ps
```

#### Paso 6: Probar al menos un caso de prueba
Abra en su navegador `http://localhost:8501` (o `http://<IP_DE_LA_VM>:8501` si está en la nube). Ingrese cualquiera de los siguientes casos auditados (o use los botones del *Panel de Auditoría Rápida* en la barra lateral izquierda) y verifique que la salida coincida exactamente:

| Parámetro de Entrada | Caso 1 (SMS Champion) | Caso 2 (Heavy Caller) | Caso 3 (Base Estándar) |
|---|:---:|:---:|:---:|
| `Subscription Length` | 32.0 | 34.0 | 30.0 |
| `Charge Amount` | 1 | 3 | 0 |
| `Seconds of Use` | 4500.0 | 11500.0 | 2000.0 |
| `Frequency of use` | 65.0 | 160.0 | 38.0 |
| `Frequency of SMS` | **290.0** | 60.0 | 15.0 |
| `Distinct Called Numbers` | 23.0 | 45.0 | 16.0 |
| `Age` | 28.0 | 35.0 | 30.0 |
| `Customer Value` | **1350.0** | 680.0 | 150.0 |
| **Cluster Predicho Esperado** | **Cluster 1: SMS Champions** | **Cluster 2: Heavy Callers** | **Cluster 0: Base Estándar** |
| **Afinidad al Centroide** | **> 85%** | **> 90%** | **> 80%** |
| **Criterio de Validación** | ✔ Debe coincidir 100% | ✔ Debe coincidir 100% | ✔ Debe coincidir 100% |

#### Paso 7: Registrar cualquier instrucción faltante o error de reproducibilidad y capturar evidencias

Para que el evaluador / pareja externa pueda entregar el soporte formal de la auditoría en un solo archivo listo para adjuntar, debe seguir estas instrucciones de captura de evidencias:

1. **Capturas de Pantalla Requeridas:**
   * **Evidencia A (Terminal):** Captura de pantalla de la terminal mostrando la ejecución exitosa de `docker run` o `python train.py` con las métricas en pantalla.
   * **Evidencia B (Navegador):** Captura de pantalla de la aplicación web en `http://localhost:8501` (o la IP pública) tras ingresar uno de los casos de prueba (ej: *Caso 1: SMS Champion*), donde se aprecie claramente el **Cluster Asignado**, la **Afinidad (%)**, y el gráfico **Radar** o **PCA 2D**.
2. **Diligenciar la Matriz de Validación:**
   * Abra la plantilla oficial en [`docs/reproducibility_peer_audit.md`](docs/reproducibility_peer_audit.md).
   * Llene los nombres de los evaluadores, la fecha, marque los 8 criterios de la lista de chequeo y consigne si hubo alguna observación o instrucción faltante.
3. **Exportar y Enviar como Archivo de Evidencia:**
   * El evaluador puede guardar ese archivo completado como PDF o Markdown (`Auditoria_Cruzada_EquipoEvaluador.pdf` o `.md`) o enviarlo por correo/Teams junto con las capturas de pantalla para ser anexado directamente a la entrega final.

> 📄 **Evidencia Recibida de la Pareja Evaluadora:**
> El dictamen de auditoría emitido por los compañeros Miguel Moreno y Julián Duarte ha sido incorporado al repositorio en:
> [`resena_trabajo_santiago_sandoval.md`](resena_trabajo_santiago_sandoval.md) con una calificación final de **5.0 / 5.0 (Excelente)**.

---

## 14. Despliegue en Google Cloud Platform (Compute Engine)

### Paso a Paso en GCP:
1. **Crear Instancia VM:**
   - Sistema Operativo: Ubuntu 24.04 LTS.
   - Tipo de máquina: `e2-micro` o `e2-small`.
   - Red: Habilitar IP Externa Efímera o Estática.
   - Etiqueta de red (*Network Tag*): `ml-streamlit-server`.
2. **Configurar Regla de Firewall:**
   ```bash
   gcloud compute firewall-rules create allow-streamlit-8501 \
       --direction=INGRESS \
       --priority=1000 \
       --network=default \
       --action=ALLOW \
       --rules=tcp:8501 \
       --source-ranges=0.0.0.0/0 \
       --target-tags=ml-streamlit-server
   ```
3. **Instalación y Despliegue en la VM (vía SSH):**
   ```bash
   # Actualizar paquetes e instalar docker y git
   sudo apt update && sudo apt install -y docker.io git
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER
   
   # Clonar el proyecto
   git clone https://github.com/santisc1304/ml-clustering-project.git
   cd ml-clustering-project
   
   # Construir y lanzar contenedor
   docker build -t ml-clustering-app .
   docker run -d --name ml-clustering --restart unless-stopped -p 8501:8501 ml-clustering-app
   ```
4. **Acceso Web:**
   Navegar a: `http://<IP_EXTERNA_DE_LA_VM>:8501`.

---

## 15. Autor y Créditos
- **Autor:** Santiago Sandoval Cortés
- **Curso:** Machine Learning II (2026-2)
- **Licencia:** MIT License
