# 🌾 Proyecto Agro-Clima: Predicción de Rendimiento Agrícola y Simulador Interactivo

Proyecto integral de Ciencia de Datos y Machine Learning orientado a analizar, modelar y simular el **rendimiento agrícola (kg/ha)** en Argentina mediante el cruce histórico de estadísticas de producción agropecuaria y variables meteorológicas satelitales (**NASA POWER**).

Además del análisis exploratorio, la segmentación no supervisada de zonas agroclimáticas y la evaluación comparativa de algoritmos supervisados y redes neuronales, el repositorio incorpora un **Simulador Predictivo Interactivo en Streamlit** impulsado por un pipeline serializado de **LightGBM**.

## 🎯 Problema de Negocio

Predecir el rendimiento esperado por hectárea (kg/ha) a nivel de **cultivo, provincia y departamento** permite:
* **Estimar volúmenes de cosecha** y logística asociada antes del cierre de campaña.
* **Evaluar sensibilidad climática** frente a variaciones de temperatura, precipitación, humedad y radiación solar.
* **Comparar el desempeño productivo** de un departamento frente a su promedio histórico regional.

## 📊 Datos y Cobertura

* **Fuente agrícola:** Estimaciones Agrícolas — Secretaría de Agricultura, Ganadería y Pesca de la Nación (Argentina).
* **Fuente climática:** NASA POWER.
* **Cobertura:** Serie histórica 1981–2024 (~40 cultivos, 23 provincias). Dataset maestro (`data/processed/df_recortado.csv`) con **83.558 observaciones válidas** (`superficie_cosechada_ha > 0`), sin imputación artificial de campañas faltantes.

## 🏗️ Arquitectura del Repositorio

* `app/app.py`: Simulador predictivo interactivo (Streamlit + Plotly).
* `data/processed/`: Datasets limpios (`df_clima_clean.csv`, `df_produccion_clean.csv`, `df_recortado.csv`).
* `models/exportar_modelo.py`: Entrenamiento y serialización del Pipeline ganador.
* `models/lgbm_rendimiento.joblib`: Modelo serializado (`ColumnTransformer` + `LightGBM`).
* `models/modelado_rendimiento_kgxha.ipynb`: Comparativa de regresores, validación cruzada 5-Fold y SHAP.
* `models/no_supervisado_zonas_agroclimaticas.ipynb`: PCA + Clustering (K-Means) de zonas agroclimáticas.
* `models/red_neuronal_mlp_rendimiento (1).ipynb`: Red Neuronal Multicapa (MLP).
* `notebooks/`: Limpieza, unión de datasets, EDA y dashboard exploratorio.

## 📈 Resultados Comparativos

| Modelo | R² (Test) | MAE Test (kg/ha) | Tiempo (s) |
| :--- | :---: | :---: | :---: |
| Regresión Lineal (Baseline) | 0.821 | 1.020 | 0.4 |
| MLP (Red Neuronal) | 0.916 | 687 | 128.5 |
| Random Forest | 0.942 | **484** | 105.0 |
| **LightGBM (Optimizado — Ganador)** | **0.942** | 500 | **7.0** |

**Selección:** **LightGBM** alcanza un $R^2 = 0.942$ (reduciendo el error medio en un 51% frente al modelo base), entrena **15 veces más rápido** que Random Forest y mantiene mayor estabilidad en validación cruzada ($R^2\text{ CV} = 0.901 \pm 0.005$).

## 🚀 Ejecución Local del Simulador

```bash
git clone [https://github.com/M-Linzuain/prediccion-rendimiento-agricola-clima.git](https://github.com/M-Linzuain/prediccion-rendimiento-agricola-clima.git)
cd prediccion-rendimiento-agricola-clima
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py
```

## 👥 Autoría y Contexto

* **Proyecto Base (Trabajo Colaborativo):** Desarrollado en equipo (junto a Santiago Scozziero y compañeros de cursada) en la **Tecnicatura Universitaria en Ciencia de Datos e IA — Universidad del Gran Rosario (UGR)**.
* **Serialización y Simulador Web:** Implementación del pipeline exportable (`exportar_modelo.py`) y aplicación interactiva en Streamlit (`app/app.py`) por **Máximo Linzuain**.