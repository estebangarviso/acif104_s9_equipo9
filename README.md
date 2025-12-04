# 📈 Predicción de Demanda en E-commerce - Equipo 9 (ACIF104)

Este repositorio contiene el proyecto final para la asignatura **Aprendizaje de Máquinas (ACIF104)** de la Universidad Andrés Bello. El objetivo es desarrollar un sistema robusto de predicción de demanda para retail utilizando una arquitectura de **Ensemble Learning (Stacking)**, enriquecida con **Clustering Particional** y desplegada mediante una aplicación web interactiva con **Streamlit**.

## 👥 Integrantes del Equipo

* **Esteban Garviso**
* **Felipe Ortega**

---

## 📂 Estructura del Proyecto

El repositorio sigue una arquitectura modular que separa claramente la lógica de negocio (Backend) de la interfaz de usuario (Frontend), cumpliendo con los estándares de ingeniería de software.

```text
acif104_s9_equipo9/
│
├── README.md               # Documentación principal y manual de ejecución.
├── Pipfile                 # Definición de dependencias y scripts del entorno.
├── Pipfile.lock            # Árbol de dependencias exacto (Hash) para reproducibilidad.
├── pyproject.toml          # Configuración centralizada de Linters (Black, Isort, Mypy).
│
├── data/                   # Almacenamiento local de datos.
│   ├── raw/                # Los datos se descargan aquí automáticamente vía KaggleHub.
│   └── processed/          # Datos transformados listos para entrenamiento.
│
├── notebooks/              # Análisis exploratorio y prototipado rápido.
│   ├── 01_EDA_Clustering.ipynb      # Análisis de outliers, K-Means y patrones temporales.
│   └── 02_Modelado_Ensemble.ipynb   # Experimentos con Stacking y Deep Learning.
│
├── src/                    # Backend: Lógica de Negocio y Modelado.
│   ├── __init__.py
│   ├── data_processing.py  # Pipeline de limpieza, clustering y feature engineering.
│   ├── train.py            # Script de entrenamiento, validación y serialización.
│   └── inference.py        # Motor de inferencia para la aplicación.
│
├── app/                    # Frontend: Interfaz de Usuario.
│   └── app.py              # Aplicación web interactiva (Streamlit).
│
└── models/                 # Artefactos serializados (Modelos entrenados).
    ├── stacking_model.pkl  # Modelo final de ensamble.
    ├── features.pkl        # Metadatos de columnas.
    └── xgb_simple_shap.pkl # Modelo proxy para explicabilidad SHAP.
````

## 🛠️ Instalación y Configuración

Este proyecto utiliza **Pipenv** para asegurar un entorno determinista y **KaggleHub** para la gestión automática del dataset.

### 1\. Prerrequisitos

* **Python:** Versión 3.10 (Requerido).
* **Gestor de Paquetes:** `pipenv` instalado globalmente.

```bash
pip install pipenv
```

### 2\. Clonar el Repositorio

```bash
git clone https://github.com/estebangarviso/acif104_s9_equipo9.git
cd acif104_s9_equipo9
```

### 3\. Instalar Dependencias

Ejecuta el siguiente comando para crear el entorno virtual e instalar las librerías exactas definidas en el `Pipfile.lock`:

```bash
pipenv install --ignore-pipfile
```

*(Para desarrollo, incluye las herramientas de calidad de código: `pipenv install --dev`)*

## 🚀 Manual de Ejecución

Hemos configurado **scripts automatizados** en Pipenv para facilitar el ciclo de vida del desarrollo. No es necesario activar el shell manualmente si usas `pipenv run`.

### A. Entrenamiento del Modelo (Backend)

Este comando descarga automáticamente el dataset desde Kaggle (si no existe), aplica el preprocesamiento (Clustering + Lags), entrena el Ensemble y guarda los modelos en la carpeta `models/`.

```bash
pipenv run train
```

*Salida esperada:* Archivos `stacking_model.pkl` y `features.pkl` generados en `models/`.

### B. Iniciar la Aplicación Web (Frontend)

Despliega la interfaz gráfica para interactuar con el modelo y visualizar la explicabilidad (SHAP).

```bash
pipenv run start
```

*La aplicación se abrirá automáticamente en tu navegador (<http://localhost:8501>).*

## 🛡️ Calidad de Código (QA)

Para garantizar la mantenibilidad y robustez del código, utilizamos un set estricto de herramientas de análisis estático. Puedes ejecutar la suite completa con un solo comando:

```bash
pipenv run check-all
```

O ejecutar herramientas individuales:

* **Formato:** `pipenv run format` (Aplica **Black** e **Isort**).
* **Linting:** `pipenv run lint` (Analiza el código con **Pylint**).
* **Tipado:** `pipenv run type-check` (Valida tipos estáticos con **Mypy**).

## 🧠 Descripción Técnica del Sistema

### 1\. Metodología

El proyecto sigue la metodología **CRISP-DM**, abarcando desde el entendimiento del negocio hasta el despliegue del prototipo.

### 2\. Arquitectura del Modelo (Stacking)

Para maximizar la capacidad predictiva, implementamos un **Ensemble Heterogéneo**:

* **Nivel Base:**
  * *Random Forest Regressor:* Captura relaciones no lineales y reduce varianza.
  * *XGBoost:* Optimiza el sesgo mediante Gradient Boosting.
* **Meta-Modelo:**
  * *Regresión Lineal:* Combina las predicciones base para generar la estimación final.

### 3\. Ingeniería de Características Avanzada

* **Clustering Particional (K-Means):** Segmentación de tiendas basada en volumen de ventas histórico para agrupar comportamientos similares.
* **Lags Temporales:** Variables de rezago (t-1, t-2, t-3) para capturar la tendencia secuencial.
* **Balanceo de Datos:** Transformación Logarítmica (`log1p`) sobre la variable objetivo para normalizar la distribución de ventas.

### 4\. Explicabilidad

Se integra **SHAP (SHapley Additive exPlanations)** en el Frontend para proporcionar transparencia algorítmica, permitiendo al usuario entender qué variables (precio, categoría, mes) influyen positiva o negativamente en cada predicción.

---

## Universidad Andrés Bello - 2025
