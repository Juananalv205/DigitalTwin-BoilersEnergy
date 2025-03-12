# 🚀 DigitalTwin-BoilersEnergy

Repositorio de un **Gemelo Digital** aplicado al análisis energético de calderas industriales. Se divide en tres etapas principales:

1. 🧪 **Exploratory Data Analysis (EDA)** — `000_EDA.ipynb`  
2. 🔄 **Extract-Transform-Load (ETL)** — `001_ETL.ipynb`  
3. 🤖 **Modelado & Análisis de Anomalías** — `002_model.ipynb`  

---

## 📖 Descripción

Este proyecto implementa un flujo completo para construir un **Digital Twin** de calderas de energía:

- 🔍 **EDA**: exploración y visualización de datos crudos (outliers, series de tiempo, nulos, formatos).  
- 🧹 **ETL**: limpieza, normalización de fechas y columnas, conversión de tipos, validación de rangos físicos y exportación de datos procesados.  
- ⚙️ **Modelado**:  
  - Autoencoder para reducción de dimensión.  
  - LSTM para detección de anomalías en series temporales.  
  - Evaluación de métricas (MAE, MSE, R²) y exportación de modelo y escaladores.

---

## 📂 Estructura del Proyecto

```plaintext
DigitalTwin-BoilersEnergy/
│
├── data/
│   ├── raw/         # Datos originales (.csv)
│   └── processed/   # Datos procesados por ETL
│
├── notebooks/
│   ├── 000_EDA.ipynb   # Análisis exploratorio
│   ├── 001_ETL.ipynb   # Pipeline de limpieza y transformación
│   └── 002_model.ipynb # Entrenamiento y evaluación de modelos
│
├── src/
│   └── utils/        # Funciones de limpieza, análisis y visualización
│
├── requirements.txt  # Dependencias pip
└── README.md         # Este archivo
```

---

# 💡 Uso

1. EDA

    - Abre notebooks/000_EDA.ipynb

    - Explora estadísticas, distribuciones y series de tiempo.

2. ETL

    - Ejecuta notebooks/001_ETL.ipynb

    - Normaliza fechas, limpia nulos y guarda CSV en data/processed/.

3. Modelo Hibrido

    - Abre notebooks/002_model.ipynb

    - Normaliza datos, entrena Autoencoder y LSTM, evalúa anomalías y guarda artefactos con joblib.

---

# 📝 Conclusiones

1. 🔄 ETL (001_ETL.ipynb)
    - 🔠 Columnas Estandarizadas y Reordenadas: nombres armonizados y secuencia lógica para análisis.

    - 🔢 Conversión Robusta de Tipos: variables de medición a float para preservar integridad.

    - 🧹 Gestión de Valores Nulos: filas con excesivos nulos imputadas con pyampute para retener observaciones críticas.

    - 📏 Validación de Rangos Físicos: se corrigieron/removieron valores anómalos de procesos.

    - 💾 Resultados Guardados: dataset limpio exportado a CSV, listo para modelado o visualización.

2. 🤖 Modelado & Análisis (002_model.ipynb)

    - 📊 Predicción vs Real — Energía Generada

    - 🔄 Alineación General: curvas coinciden en tendencias y estacionalidades.

    - 📐 Magnitud del Error: desviaciones contenidas en banda estrecha alrededor de la serie real.

    - ⚡ Respuesta a Picos y Valles: captura casi todo, con leve retardo en valles profundos.

    - 🌊 Ruido y Estabilidad: serie de predicción mantiene variabilidad controlada.

    - 🏁 Conclusión: superposición casi completa y desviación de ruido (~0.05) indica desempeño excelente.

---

# 📜 Licencia
Este proyecto está bajo MIT License.
Consulta LICENSE para más detalles.

