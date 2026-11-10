# 📋 Reseña Técnica y Dictamen de Auditoría Cruzada
## Evaluación del Proyecto: *Neural Customer Segmentation Engine | MLOps Suite*
### Autor Auditado: Santiago Sandoval Cortés
**Repositorio Oficial:** [`https://github.com/santisc1304/ml-clustering-project`](https://github.com/santisc1304/ml-clustering-project)

---

### 1. Resumen Ejecutivo de la Evaluación

En cumplimiento con la **Sección 31 (Prueba cruzada entre parejas)** y **Sección 32 (Entregables)** de la guía oficial del reto de Machine Learning 2, se llevó a cabo una auditoría independiente y ciega (*blind peer review*) sobre el proyecto de Santiago Sandoval Cortés.

* **Resultado Global:** **APROBADO CON EXCELENCIA (100 / 100)**.
* **Nivel de Reproducibilidad:** **Totalmente determinista y reproducible**.
* **Observaciones de Ejecución:** Cero incidencias o bloqueos en clonación, construcción de imagen Docker y serving web.

---

### 2. Verificación de los 7 Pasos de la Prueba Cruzada (Sección 31)

| Paso | Requerimiento de la Guía | Cumplimiento | Detalle de la Observación |
| :---: | :--- | :---: | :--- |
| **1** | Recibir únicamente la URL del repositorio | ✔ **100%** | Se suministró únicamente el enlace a GitHub sin archivos adicionales ni asistencia manual. |
| **2** | Clonar el repositorio | ✔ **100%** | Descarga limpia de ramas y archivos: dataset (3,150 registros), `train.py`, `app.py`, `Dockerfile`, `.dockerignore`, `requirements.txt` y documentación. |
| **3** | Leer README sin instrucciones verbales | ✔ **100%** | El `README.md` es exhaustivo, describe el mapa de entregables, las 8 variables, el pipeline, métricas matemáticas y los comandos exactos de CLI y Docker. |
| **4** | Construir la imagen Docker | ✔ **100%** | Se ejecutó `docker build -t peer-clustering-app .`. Compilación multi-stage con usuario no-root (`appuser`), finalizada con código de salida 0. |
| **5** | Ejecutar el contenedor | ✔ **100%** | Se desplegó con `docker run -d -p 8502:8501 peer-clustering-app`. Estado `Up (healthy)` inmediato y respuesta `HTTP 200 OK`. |
| **6** | Probar al menos un caso de inferencia | ✔ **100%** | Se evaluaron tres perfiles distintos (SMS Champions, Heavy Callers y Base Estándar); todos arrojaron la salida y afinidad esperadas. |
| **7** | Registrar incidencias o errores | ✔ **100%** | No se registraron instrucciones faltantes ni fallos de dependencias. |

---

### 3. Resultados de los Casos de Prueba Auditados

Se validaron las entradas del modelo frente a los centroides de K-Means calculados en `model.joblib`:

1. **Caso 1 — Digital Champions & High-Value SMS Lovers (Clúster 1):**
   * *Entradas:* SMS = 300, Customer Value = $1,400 USD, Segundos de voz = 4,500s, Edad = 28.
   * *Resultado obtenido:* **Clúster 1** con afinidad > 88%.
   * *Diagnóstico:* Correctamente clasificado en el segmento de clientes jóvenes, de alto tráfico de mensajería y máximo valor económico.
2. **Caso 2 — Heavy Callers / Negocios de Voz (Clúster 2):**
   * *Entradas:* Segundos de voz = 12,000s (~3.3 horas), Llamadas = 160, Números distintos = 48.
   * *Resultado obtenido:* **Clúster 2** con afinidad > 90%.
   * *Diagnóstico:* Clasificado con exactitud en el segmento ejecutivo de alta demanda de tráfico de voz.
3. **Caso 3 — Base Masiva Tradicional (Clúster 0):**
   * *Entradas:* Segundos de voz = 1,800s, Llamadas = 30, SMS = 10, Customer Value = $120 USD.
   * *Resultado obtenido:* **Clúster 0** con afinidad > 80%.
   * *Diagnóstico:* Identificado como cliente convencional de uso esporádico (61.5% del total de clientes).

---

### 4. Fortalezas Destacadas del Proyecto

1. **Arquitectura y Calidad de Código:**
   * Preprocesamiento hermético dentro de un `Pipeline` de scikit-learn con `SimpleImputer` y `StandardScaler`.
   * Modelo totalmente desacoplado mediante artefacto `model.joblib`, evitando reentrenamientos en inferencia.
2. **Experiencia de Usuario (UI/UX):**
   * Interfaz personalizada en Streamlit con diseño visual violeta/rojo oscuro.
   * Incorporación de gráficos interactivos de proyección espacial (PCA 2D) y radar comparativo de centroides mediante Plotly.
3. **Buenas Prácticas de Ingeniería y Seguridad en Docker:**
   * Uso de imagen base ligera (`python:3.12-slim`).
   * Creación de usuario sin privilegios `appuser` para no correr como `root`.
   * Mecanismo de comprobación de salud (*Healthcheck*) integrado en el puerto 8501.
4. **Documentación Complementaria:**
   * Inclusión de pruebas automatizadas con `pytest` (`tests/test_pipeline_and_inference.py`), análisis de negocio (`docs/cluster_business_interpretation.md`) y respuestas conceptuales a las preguntas del docente (`docs/preguntas_discusion_final.md`).

---

### 5. Dictamen Final

| Criterio Evaluado | Ponderación | Calificación |
| :--- | :---: | :---: |
| Autonomía de Instrucciones en README | 25% | **5.0 / 5.0** |
| Reproducibilidad de Construcción Docker | 25% | **5.0 / 5.0** |
| Estabilidad del Contenedor en Ejecución | 25% | **5.0 / 5.0** |
| Coherencia de Predicciones e Interpretación | 25% | **5.0 / 5.0** |
| **CALIFICACIÓN FINAL DE AUDITORÍA** | **100%** | **5.0 / 5.0 (Excelente)** |

**Auditor:** Miguel Moreno & Julian Duarte  
**Fecha:** 10 de Noviembre de 2026
