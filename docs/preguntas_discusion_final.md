# Respuestas Técnicas a las Preguntas de Discusión Final (Sección 35)
## Argumentación de Nivel Industrial para la Sustentación del Proyecto MLOps

---

### 1. ¿Qué diferencia existe entre el notebook de experimentación y `train.py`?
- **Notebook de Experimentación:** Diseñado para la exploración interactiva, prototipado rápido, visualizaciones ad-hoc y depuración paso a paso. Mantiene el estado en la memoria del kernel de forma no secuencial y oculta, lo que induce errores de reproducibilidad (celdas ejecutadas fuera de orden).
- **`train.py`:** Es un script de software industrial determinista, autónomo y ejecutable vía CLI. Sigue el principio de responsabilidad única: toma datos limpios como entrada, ajusta un pipeline inmutable bajo una semilla fija (`random_state=42`), evalúa el desempeño formalmente y serializa el artefacto `model.joblib`. No depende de estados volátiles de memoria.

---

### 2. ¿Por qué conviene guardar preprocesamiento y modelo dentro del mismo `Pipeline`?
Encapsular transformaciones (`SimpleImputer`, `StandardScaler`) y el estimador (`KMeans`) dentro de un `Pipeline` de scikit-learn garantiza la **hermeticidad del flujo de inferencia** y elimina la fuga de datos (*data leakage*):
1. **Consistencia matemática:** Las transformaciones aplicadas durante el entrenamiento (ej. medias y desviaciones de estandarización Z-score calculadas en `fit`) se reutilizan idénticas al llamar `predict()` en producción.
2. **Atomicidad del artefacto:** La aplicación cliente (`app.py` o microservicio) solo interactúa con un objeto que recibe el dato crudo y entrega la predicción, evitando duplicar código de preprocesamiento en el frontend.

---

### 3. ¿Qué ocurre si Streamlit transforma los datos de una forma distinta al entrenamiento?
Ocurre un fenómeno catastrófico conocido como **desalineación de entrenamiento-servicio (*Training-Serving Skew*)**:
- Si el orden de las columnas se altera, si una variable no se escala o si se escala con estadísticas calculadas sobre el dato entrante (y no sobre la base de entrenamiento), las distancias euclidianas en el espacio multidimensional se corrompen por completo.
- El modelo asignará centroides y clusters erróneos de forma silenciosa, sin arrojar una excepción visible, entregando recomendaciones comerciales espurias.

---

### 4. ¿Qué diferencia existe entre `model.joblib` y la imagen Docker?
- **`model.joblib`:** Es un artefacto binario serializado que contiene los pesos, parámetros aprendidos (centroides de K-Means, medias y varianzas de `StandardScaler`), hiperparámetros y metadatos del pipeline de Machine Learning. No es ejecutable por sí mismo; requiere un intérprete de Python compatible y librerías instaladas.
- **Imagen Docker:** Es un paquete inmutable y autocontenido que encapsula todo el entorno de ejecución: el sistema operativo base (Debian/Linux), el runtime de Python 3.12, las librerías con sus versiones exactas (`requirements.txt`), el código de la aplicación (`app.py`), el servidor web y el artefacto `model.joblib`. Garantiza reproducibilidad en cualquier infraestructura (local, Colab o GCP).

---

### 5. ¿Qué contiene realmente una imagen Docker?
Una imagen Docker es un conjunto de capas de solo lectura (*read-only layers*) compuestas por:
1. **Capa base del SO:** Una distribución mínima de Linux (ej. `python:3.12-slim`).
2. **Capa de librerías del sistema:** Binarios, utilidades del kernel y certificados CA.
3. **Capa de dependencias Python:** Paquetes instalados en el entorno (Streamlit, scikit-learn, etc.).
4. **Capa de código de usuario y datos:** Código fuente (`app.py`, `train.py`), artefacto `model.joblib` y datasets.
5. **Metadatos e instrucciones de inicio:** Puertos expuestos (`EXPOSE 8501`), variables de entorno, usuario de ejecución (`appuser`) y el comando de arranque (`CMD`).

---

### 6. ¿Qué diferencia existe entre una imagen y un contenedor?
- **Imagen (Clase/Molde):** Estructura pasiva, inmutable y empaquetada en disco. Representa la definición estática del software y su entorno.
- **Contenedor (Instancia/Objeto):** Es un proceso vivo en ejecución aislado en el host mediante *namespaces* y *cgroups* del kernel de Linux, instanciado a partir de una imagen. Posee una capa de lectura/escritura efímera superpuesta sobre las capas de la imagen.

---

### 7. ¿Por qué escribimos el Dockerfile en Colab pero lo construimos en la VM?
- **Limitación de Colab:** Google Colab opera dentro de un contenedor gestionado donde el demonio de Docker (*Docker Daemon*) no está disponible ni se tienen privilegios para crear sub-contenedores (*Docker-in-Docker*). Por tanto, en Colab únicamente se redactan los archivos de configuración (`Dockerfile`, `requirements.txt`).
- **VM en GCP:** La máquina virtual de Compute Engine proporciona un sistema operativo completo con acceso `root`/`sudo`, permitiendo instalar Docker, construir la imagen con `docker build` y exponer puertos públicos con `docker run`.

---

### 8. ¿Por qué necesitamos mapear puertos (`-p 8501:8501`)?
El contenedor opera en su propia red interna aislada (*Bridge Network*) con una dirección IP privada invisible fuera del host.
- El parámetro `-p 8501:8501` crea una regla de reenvío de puertos (*Port Forwarding / NAT*): todo el tráfico entrante al puerto 8501 del host (la VM) se canaliza directamente hacia el puerto 8501 del contenedor donde Streamlit está escuchando en `0.0.0.0`.

---

### 9. ¿Por qué existe además una regla de firewall en GCP?
El mapeo de puertos de Docker opera únicamente a nivel local dentro de la máquina virtual (capa del host).
- Sin embargo, la red virtual de Google Cloud (**VPC**) posee un firewall perimetral por defecto que bloquea todas las conexiones entrantes desde Internet por motivos de ciberseguridad.
- La regla de firewall en GCP (`gcloud compute firewall-rules`) autoriza explícitamente a los enrutadores perimetrales de Google a dejar pasar paquetes TCP destinados al puerto 8501 de las instancias que posean la etiqueta correspondiente (`ml-streamlit-server`).

---

### 10. ¿Qué parte del flujo cambia entre clasificación y clustering?
- **En la Preparación de Datos:** En clasificación se requiere la variable objetivo $y$ (etiquetas conocidas) y se realiza una partición estratificada (`train_test_split`). En clustering únicamente se procesa la matriz de características $X$, sin etiquetas supervisadas.
- **En el Estimador:** Se reemplaza el clasificador (ej. `LogisticRegression`, `RandomForestClassifier`) por un agrupador basado en distancias (ej. `KMeans`).
- **En la Evaluación:** Se sustituyen la matriz de confusión, ROC-AUC y F1-score por métricas de cohesión y separación interna: coeficiente Silhouette, inercia (codo), índice Calinski-Harabasz y perfilamiento descriptivo de centroides.
- **En el Serving:** La clasificación entrega una probabilidad o clase discreta; el clustering entrega el cluster asignado, afinidad al centroide y la descripción cualitativa del perfil poblacional.

---

### 11. ¿Qué parte del flujo sería prácticamente idéntica si el modelo fuera una red neuronal?
El **90% de la arquitectura MLOps se mantiene intacta**:
1. El contrato del pipeline: entrada de datos crudos, transformaciones y salida de inferencia.
2. La serialización del artefacto (ej. guardar pesos con PyTorch/TensorFlow o ONNX).
3. La interfaz de usuario en Streamlit (`app.py`), que seguirá recibiendo los mismos datos y consumiendo `predict()`.
4. El empaquetado en Docker, el mapeo de puertos y la infraestructura en GCP Compute Engine.
Lo único que cambiaría sería la librería de cálculo en `requirements.txt` (ej. `torch` o `tensorflow`), la posible adición de soporte para GPU en Docker (`nvidia-container-toolkit`) y la función interna de entrenamiento en `train.py`.

---

### 12. ¿Qué mejoraríamos para convertir este laboratorio en un despliegue productivo?
1. **Desacoplamiento Frontend / Backend:** Separar Streamlit en una capa cliente estricta y crear una API REST de inferencia de baja latencia con **FastAPI** detrás de un balanceador de carga.
2. **Seguridad y Cifrado:** Implementar un Reverse Proxy con **Nginx**, certificados SSL/TLS con **HTTPS**, autenticación JWT y limitación de tasa (*rate limiting*).
3. **Serverless Hosting:** Migrar de una VM persistente a **Google Cloud Run**, permitiendo autoescalado a cero instancias para reducir costos cuando no haya tráfico.
4. **CI/CD Automatizado:** Construir pipelines con **GitHub Actions** para ejecutar pruebas automáticas con `pytest`, construir la imagen y desplegarla en GCP automáticamente ante cada `git push`.
5. **Monitoreo de Data Drift:** Implementar herramientas como Evidently AI o Prometheus/Grafana para detectar si la distribución de los clientes cambia en el tiempo (*Covariate Shift*).

