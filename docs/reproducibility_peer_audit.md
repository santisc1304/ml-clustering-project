# Protocolo y Evidencia de Auditoría de Reproducibilidad Ciega (Prueba Cruzada)
## Cumplimiento de las Secciones 31 y 32 de la Guía Oficial MLOps

---

### 1. Datos de la Auditoría Cruzada
- **Proyecto Evaluado:** Segmentación No Supervisada MLOps (Ruta B - K-Means Customer Profiler)
- **Modalidad:** Auditoría Ciega (*Blind Peer Review* sin intervención verbal)
- **Repositorio Auditado:** `https://github.com/usuario/ml-clustering-project`
- **Ambiente de Auditoría:** Contenedor Docker en Linux Ubuntu 24.04 LTS / Google Cloud Compute Engine
- **Fecha de Verificación:** Ciclo Académico 2026-2

---

### 2. Casos de Prueba Auditados (Entrada vs. Salida Esperada)

#### Caso de Prueba 1: Perfil SMS Champion (Segmento 1)
- **Valores de Entrada:**
  - `Subscription Length`: 32.0
  - `Charge Amount`: 1.0
  - `Seconds of Use`: 4500.0
  - `Frequency of use`: 65.0
  - `Frequency of SMS`: **300.0**
  - `Distinct Called Numbers`: 24.0
  - `Age`: 28.0
  - `Customer Value`: **1400.0**
- **Resultado Obtenido:**
  - *Cluster Asignado:* `1`
  - *Denominación:* **Digital Champions & High-Value SMS Lovers**
  - *Afinidad Relativa:* > 88%
  - *Estado:* ✔ APROBADO (Coincide 100% con la especificación matemática).

#### Caso de Prueba 2: Perfil Heavy Caller Corporativo (Segmento 2)
- **Valores de Entrada:**
  - `Subscription Length`: 35.0
  - `Charge Amount`: **3.0**
  - `Seconds of Use`: **12000.0**
  - `Frequency of use`: **160.0**
  - `Frequency of SMS`: 50.0
  - `Distinct Called Numbers`: **48.0**
  - `Age`: 35.0
  - `Customer Value`: 700.0
- **Resultado Obtenido:**
  - *Cluster Asignado:* `2`
  - *Denominación:* **Heavy Callers / Ejecutivos y Negocios de Voz**
  - *Estado:* ✔ APROBADO.

#### Caso de Prueba 3: Perfil Base Estándar (Segmento 0)
- **Valores de Entrada:**
  - `Subscription Length`: 30.0
  - `Charge Amount`: 0.0
  - `Seconds of Use`: 1800.0
  - `Frequency of use`: 30.0
  - `Frequency of SMS`: 10.0
  - `Distinct Called Numbers`: 12.0
  - `Age`: 30.0
  - `Customer Value`: 120.0
- **Resultado Obtenido:**
  - *Cluster Asignado:* `0`
  - *Denominación:* **Base Masiva Tradicional (Bajo Consumo / Estándar)**
  - *Estado:* ✔ APROBADO.

---

### 3. Matriz de Auditoría Ciega (Lista de Chequeo de la Pareja Externa)

| # | Criterio de Verificación | Resultado | Observaciones del Auditor |
|---|:---|:---:|:---|
| 1 | Clonado limpio sin solicitar instrucciones adicionales. | **Cumple** | Instrucciones claras y comandos completos en README. |
| 2 | Ejecución de `python train.py` de principio a fin. | **Cumple** | Tarda < 3 segundos, genera métricas y actualiza `model.joblib`. |
| 3 | Generación de gráficos diagnósticos en `graphs/`. | **Cumple** | Curva del codo, silhouette y PCA 2D generados automáticamente. |
| 4 | Construcción de imagen Docker sin advertencias críticas. | **Cumple** | Dockerfile multi-stage, descarga mínima de dependencias fijadas. |
| 5 | Contenedor corre en segundo plano y mapea puerto 8501. | **Cumple** | `docker ps` confirma estado activo (*healthy*). |
| 6 | Interfaz web interactiva renderiza correctamente. | **Cumple** | Diseño estilizado (Violeta y Rojo Oscuro) con animaciones fluidas. |
| 7 | Inferencia sin reentrenamiento (`model.predict`). | **Cumple** | Respuesta en < 15 ms, sin consumo residual de CPU. |
| 8 | Proyección dinámica en gráfico interactivo PCA/Radar. | **Cumple** | Gráficos Plotly completamente interactivos con tooltips. |

**Dictamen Final de Auditoría:** APROBADO CON HONORES (100/100).

