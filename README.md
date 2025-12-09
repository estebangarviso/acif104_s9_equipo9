# Predicción de Demanda en E-commerce - Equipo 9 (ACIF104)

Este repositorio contiene el proyecto final para la asignatura **Aprendizaje de Máquinas (ACIF104)** de la Universidad Andrés Bello. El objetivo es desarrollar un sistema robusto de predicción de demanda para retail utilizando:

- 🧠 **Machine Learning Avanzado**: Ensemble Stacking (Random Forest + XGBoost + meta-estimador) + Deep Learning (MLP + LSTM-DNN)
- 📊 **Ingeniería de Features Avanzada**: 
  - Clustering K-Means para segmentación de tiendas
  - **24+ features engineered**: Momentum (deltas, aceleración), Sensibilidad al Precio (elasticidad, ingresos), Desviaciones (z-scores, volatilidad)
  - **Exactamente 2 ventanas rolling parametrizables** (default: 3 y 6 meses)
  - Balanceo con SMOTE opcional
- 🌐 **Arquitectura Desacoplada**: Backend REST API (FastAPI) + Frontend (Streamlit) con comunicación HTTP
- 🔄 **MLOps Best Practices**: Validación temporal, sincronización automática de dependencias, versionado de modelos

## Integrantes del Equipo

* **Esteban Garviso**
* **Felipe Ortega**

## Estructura del Proyecto

El proyecto sigue una arquitectura modular que desacopla la lógica de negocio (Backend REST API) de la capa de presentación (Frontend Streamlit), facilitando la mantenibilidad y escalabilidad:

```text
acif104_s9_equipo9/
│
├── README.md               # Documentación completa del proyecto
├── Pipfile                 # Gestión de dependencias con Pipenv
├── Pipfile.lock            # Árbol de dependencias exacto (reproducibilidad)
├── requirements.txt        # Dependencias (generado automáticamente)
├── requirements-dev.txt    # Dependencias de desarrollo (generado automáticamente)
├── Makefile                # Comandos de automatización (install, train, api, start)
├── pyproject.toml          # Configuración de QA (Black, Isort, Mypy)
│
├── .githooks/              # Git hooks personalizados
│   └── pre-commit          # Auto-sincronización de requirements.txt al commitear
│
├── data/                   # Datasets con sistema de respaldo automático
│   ├── .gitkeep            # Los datos se descargan automáticamente vía KaggleHub
│   └── [*.csv]             # Respaldo local: sales_train, items, shops, item_categories
│
├── models/                 # Modelos entrenados y metadatos
│   ├── stacking_model.pkl  # Ensemble Stacking (Random Forest + XGBoost)
│   ├── mlp_model.keras     # Red Neuronal MLP (3 capas densas)
│   ├── lstm_model.keras    # Red Neuronal LSTM-DNN simplificada
│   ├── scaler.pkl          # StandardScaler para normalización
│   └── metrics.json        # Métricas comparativas (RMSE, MAE, R²)
│
├── notebooks/              # Prototipado y análisis exploratorio
│   ├── 01_EDA_Clustering.ipynb      # K-Means, Outliers y patrones temporales
│   └── 02_Modelado_Ensemble.ipynb   # Experimentos con Stacking y Deep Learning
│
├── src/                    # Backend: Lógica de Negocio y Modelado
│   ├── __init__.py         # Inicialización del paquete
│   ├── data_processing.py  # Pipeline ETL: SMOTE, Rolling Windows, TimeSeriesSplit
│   ├── train.py            # Entrenamiento de 5 modelos (RF, XGB, MLP, LSTM, Stacking)
│   ├── inference.py        # Motor de inferencia con sistema de respaldo
│   └── api.py              # FastAPI REST API (5 endpoints con Pydantic)
│
├── app/                    # Frontend: Interfaz de Usuario con Streamlit
│   ├── README.md           # Documentación de arquitectura modular
│   ├── app.py              # Punto de entrada principal
│   ├── config.py           # Configuraciones centralizadas
│   ├── state_manager.py    # Gestión de estado (Singleton)
│   │
│   ├── services/           # Lógica de negocio
│   │   ├── pricing_service.py       # Precios dinámicos por categoría
│   │   ├── prediction_service.py    # Cliente HTTP para API REST
│   │   └── trend_analyzer.py        # Análisis de tendencias
│   │
│   ├── components/         # Componentes de visualización
│   │   ├── chart_builder.py         # Gráficos Plotly reutilizables
│   │   ├── shap_renderer.py         # Renderizado SHAP (dark/light theme)
│   │   └── dataframe_builder.py     # Construcción de DataFrames
│   │
│   ├── ui_components/      # Componentes UI
│   │   ├── header.py       # Encabezado con branding
│   │   └── sidebar.py      # Formulario de predicción
│   │
│   └── views/              # Vistas de navegación
│       ├── prediction_view.py       # Vista principal de predicción
│       ├── monitoring_view.py       # Dashboard de monitoreo
│       └── about_view.py            # Información del proyecto
```
│   │
│   ├── views/              # Vistas principales
│   │   ├── prediction_view.py       # Análisis predictivo con KPIs y SHAP
│   │   ├── monitoring_view.py       # Salud del modelo + Mantenimiento
│   │   └── architecture_view.py     # Documentación técnica
│   │
│   └── ui_components/      # Componentes de UI
│       ├── sidebar.py               # Controles laterales y formularios
│       └── header.py                # Encabezado de la aplicación
│
└── models/                 # Artefactos serializados (Persistencia)
    ├── stacking_model.pkl  # Modelo final de ensamble (RF + XGBoost)
    ├── features.pkl        # Metadatos de columnas
    ├── xgb_simple_shap.pkl # Modelo proxy para explicabilidad
    └── category_prices.pkl # Precios promedio por categoría
```

## Inicio Rápido

```bash
# 1. Clonar repositorio
git clone https://github.com/estebangarviso/acif104_s9_equipo9.git
cd acif104_s9_equipo9

# 2. Instalar dependencias
pipenv install --ignore-pipfile

# 3. Iniciar Backend (Terminal 1)
pipenv run api

# 4. Iniciar Frontend (Terminal 2)
pipenv run start
```

📖 **Documentación completa:** Ver [docs/INSTALLATION.md](docs/INSTALLATION.md)

## Características Principales

- **5 Modelos ML/DL:** Random Forest, XGBoost, MLP, LSTM-DNN, Stacking Ensemble
- **Ingeniería de Features Avanzada (24+ variables):**
  - **Momentum:** Deltas (delta_1_2, evolution_3m), promedios y dirección de tendencia
  - **Sensibilidad al Precio:** Cambios porcentuales, elasticidad precio-demanda, ingreso potencial
  - **Desviaciones:** Z-scores, diferencias vs promedio, coeficientes de volatilidad
  - **Rolling Windows:** 2 ventanas temporales parametrizables (mean + std)
  - **Clustering K-Means:** Segmentación automática de tiendas
  - **Balanceo SMOTE:** Opcional para clases desbalanceadas
- **API REST con FastAPI:** 5 endpoints documentados con Swagger UI
- **Frontend Streamlit:** Interfaz interactiva con explicabilidad SHAP
- **Validación Temporal:** TimeSeriesSplit para prevenir data leakage
- **Sistema de Respaldo:** Gestión automática de datasets con KaggleHub

📖 **Detalles técnicos:** Ver [docs/TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md)  
📖 **Documentación API:** Ver [docs/API.md](docs/API.md)

## Capturas de Pantalla

### Vista de Predicción
![Vista de Predicción](docs/screenshots/prediction-view.png)

### Panel de Monitoreo
![Panel de Monitoreo](docs/screenshots/monitoring-view.png)

## Tecnologías Utilizadas

**Machine Learning:** scikit-learn, XGBoost, TensorFlow, imbalanced-learn, SHAP  
**Backend:** FastAPI, Pydantic, uvicorn  
**Frontend:** Streamlit, Plotly, httpx  
**Data:** pandas, numpy, KaggleHub  
**QA:** Black, Pylint, Mypy, Isort, pytest

📖 **Ver versiones completas:** [docs/INSTALLATION.md](docs/INSTALLATION.md)

## Métricas de los Modelos

Comparativa de rendimiento (dataset de validación con TimeSeriesSplit):

| Modelo            | RMSE  | MAE   | R²        | Tipo              | Estado             |
| :---------------- | :---- | :---- | :-------- | :---------------- | :----------------- |
| **Random Forest** | 0.028 | 0.017 | **0.999** | Tree-based        | ✅ Óptimo           |
| XGBoost           | 0.120 | 0.052 | 0.984     | Gradient Boosting | ✅ Excelente        |
| Stacking Ensemble | 0.821 | 0.807 | 0.276     | Ensemble          | ⚠️ Bajo rendimiento |
| MLP               | 0.791 | 0.591 | 0.327     | Neural Network    | ⚠️ Requiere ajuste  |
| LSTM-DNN          | 6.348 | 6.271 | -42.330   | Neural Network    | ❌ Fallo crítico    |

**Conclusiones:**
- **Random Forest es el modelo ganador** con R²=0.999, superando incluso al Stacking Ensemble
- Los modelos tree-based (RF, XGBoost) superan significativamente a Deep Learning en datos tabulares pequeños
- **El Stacking Ensemble tiene rendimiento inferior** (R²=0.276) a sus estimadores base, posiblemente por:
  - Overfitting del meta-estimador en validación temporal
  - Desbalance en los pesos de combinación
  - Incompatibilidad entre predicciones de estimadores heterogéneos
- **LSTM-DNN falló completamente** (R²=-42.33) indicando divergencia en entrenamiento
- Deep Learning requiere datasets más grandes para convergencia óptima
- TimeSeriesSplit previene overfitting temporal y data leakage

**Recomendación:** Usar **Random Forest** como modelo de producción por su estabilidad y rendimiento superior

## Documentación Adicional

- 📘 [Guía de Instalación](docs/INSTALLATION.md) - Configuración completa del entorno
- 🔧 [Detalles Técnicos](docs/TECHNICAL_DETAILS.md) - Metodología, arquitectura y features
- 🌐 [Documentación API](docs/API.md) - Endpoints y ejemplos de uso
- 🏗️ [Arquitectura Frontend](app/README.md) - Patrones SOLID y estructura modular

## Universidad Andrés Bello - 2025

**Asignatura:** ACIF104 - Aprendizaje de Máquinas  
**Docente:** OMAR IVÁN SALINAS SILVA  
**Periodo:** Sexto Trimestre 2025
