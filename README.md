# ecommerce-demand-prediction

## Descripción
Proyecto de predicción de demanda en e-commerce utilizando técnicas de Machine Learning y Deep Learning. Incluye análisis exploratorio, comparación de modelos, explicabilidad con SHAP y despliegue en Google Colab.

Este proyecto forma parte de la segunda fase del curso de Aprendizaje Automático. El objetivo es desarrollar un sistema predictivo que anticipe la demanda de productos en un negocio de e-commerce, optimizando la gestión de inventarios y reduciendo costos.

## Características
- Análisis exploratorio de datos.
- Implementación de modelos ML (Random Forest, XGBoost) y DL (LSTM).
- Evaluación con métricas (MAE, RMSE, F1-score).
- Explicabilidad con SHAP.
- Integración con Google Colab.

## Requisitos
- Python 3.10+
- pipenv para gestión de dependencias.

## Instalación
```bash
# Clonar el repositorio
git clone https://github.com/usuario/ecommerce-demand-prediction.git
cd ecommerce-demand-prediction

# Instalar dependencias con pipenv
pip install pipenv
pipenv install
```

## Ejecución en Google Colab
1. Subir los notebooks a Google Colab.
2. Instalar dependencias:
```python
!pip install kagglehub[pandas-datasets]
!pip install shap scikit-learn xgboost tensorflow
```
3. Ejecutar las celdas en orden.

## Estructura del repositorio
```
ecommerce-demand-prediction/
│
├── data/                # Datasets
├── notebooks/           # Jupyter notebooks
├── src/                 # Código fuente
├── Pipfile              # Dependencias
├── Pipfile.lock         # Bloqueo de dependencias
└── README.md            # Documentación
```

## Licencia
Este proyecto se distribuye bajo la licencia MIT.
