# Interpretación Técnica y Estratégica de los Perfiles de Clustering (Ruta B)

## 1. Ficha Metodológica del Modelo
- **Algoritmo:** K-Means con inicialización k-means++ (`n_init=10`, `random_state=42`).
- **Preprocesamiento:** Imputación por mediana + Estandarización z-score (`StandardScaler`).
- **Población Analizada:** 3,150 clientes de telecomunicaciones.
- **Variables Seleccionadas:** 8 variables numéricas de comportamiento y uso.
- **Métricas Globales de Estructura:**
  - **Silhouette Score:** `0.3536` (Separación limpia y sin clusters difusos negativos).
  - **Calinski-Harabasz Index:** `1203.58` (Alta varianza inter-cluster respecto a intra-cluster).
  - **Davies-Bouldin Index:** `1.2517` (Baja similitud media entre clusters).
  - **Inercia:** `14278.35`.

---

## 2. Descripción Exhaustiva de los Segmentos Descubiertos

### Segmento 0: Base Masiva Tradicional (Bajo Consumo / Estándar)
- **Representación:** 1,938 clientes (**61.5%** de la base).
- **Métricas Medias:**
  - *Segundos de Uso:* 2,297.63 s (~38 minutos totales).
  - *Frecuencia de Uso:* 41.43 llamadas.
  - *Mensajes SMS:* 16.81 mensajes.
  - *Contactos Distintos:* 16.93 personas.
  - *Edad Promedio:* 30.65 años.
  - *Customer Value:* **159.30 USD**.
  - *Nivel de Cargo:* 0.45.
- **Interpretación del Perfil:** Clientes con uso esporádico y funcional de su línea telefónica. Tienen poca interacción por SMS y bajo volumen de minutos. Su valor unitario es bajo, pero representan la base transaccional más grande de la compañía.
- **Acción Estratégica:** Evitar costos excesivos de contacto. Implementar campañas automatizadas de baja fricción (descuentos por lealtad digital, bundles de datos para incentivar el consumo de datos).

---

### Segmento 1: Digital Champions & High-Value SMS Lovers
- **Representación:** 562 clientes (**17.8%** de la base).
- **Métricas Medias:**
  - *Segundos de Uso:* 4,277.33 s (~71 minutos).
  - *Frecuencia de Uso:* 65.79 llamadas.
  - *Mensajes SMS:* **281.82 mensajes** (17 veces más que el Segmento 0).
  - *Contactos Distintos:* 22.71 personas.
  - *Edad Promedio:* 28.59 años (el segmento más joven).
  - *Customer Value:* **1,334.92 USD** (el más rentable de todo el portafolio).
  - *Nivel de Cargo:* 1.09.
- **Interpretación del Perfil:** Clientes jóvenes, altamente digitales y monetizables. Aunque su uso de voz es moderado, son los campeones indiscutibles en mensajería e interacciones de alto valor agregado. Son los clientes de mayor rentabilidad neta por usuario.
- **Acción Estratégica:** Fidelización VIP de alta prioridad. Programas de recompensas digitales, accesos exclusivos a eventos o streaming, y blindaje proactivo contra ofertas de la competencia.

---

### Segmento 2: Heavy Callers / Ejecutivos y Negocios de Voz
- **Representación:** 650 clientes (**20.6%** de la base).
- **Métricas Medias:**
  - *Segundos de Uso:* **11,125.50 s** (~185 minutos / más de 3 horas).
  - *Frecuencia de Uso:* **156.22 llamadas** (4 veces más que el Segmento 0).
  - *Mensajes SMS:* 60.84 mensajes.
  - *Contactos Distintos:* **43.83 personas** (red social amplia).
  - *Edad Promedio:* 34.12 años (segmento más maduro/profesional).
  - *Customer Value:* **653.25 USD** (segundo mayor margen).
  - *Nivel de Cargo:* **2.28** (el cargo mensual directo más alto).
- **Interpretación del Perfil:** Usuarios corporativos, comerciales o profesionales independientes dependientes de la telefonía tradicional. Red de contactos amplia y consumo intensivo de voz.
- **Acción Estratégica:** Planes corporativos ilimitados, soluciones multidispositivo, servicio de atención telefónica ejecutiva sin colas de espera (*Gold Tier*) y facturación consolidada.

