# Predicción de Demanda en E-commerce - Equipo 9 (ACIF104)

Este repositorio contiene el proyecto final para la asignatura **Aprendizaje de Máquinas (ACIF104)** de la Universidad Andrés Bello. El objetivo es desarrollar un sistema robusto de predicción de demanda para retail utilizando una arquitectura de **Ensemble Learning (Stacking)**, enriquecida con **Clustering Particional** y desplegada mediante una aplicación web interactiva con **Streamlit**.

## Integrantes del Equipo

* **Esteban Garviso**
* **Felipe Ortega**

---

##  Estructura del Proyecto

El proyecto está organizado en carpetas que separan claramente el backend del frontend. Esto permite ordenar el desarrollo y facilita entender dónde está cada parte del sistema:

```text
acif104_s9_equipo9/
│
├── README.md               # Explica cómo usar y ejecutar el proyecto.
├── Pipfile / Pipfile.lock  # Dependencias del entorno (reproducibles).
├── pyproject.toml          # Configuración de herramientas (formato, linting, etc.)
│
├── data/                   
│   ├── raw/                # Datos descargados directamente desde KaggleHub.
│   └── processed/          # Datos limpios listos para entrenar modelos.
│
├── notebooks/              # Exploración inicial y pruebas.
│   ├── 01_EDA_Clustering.ipynb      # Outliers, K-Means y análisis temporal.
│   └── 02_Modelado_Ensemble.ipynb   # Pruebas con stacking y modelos más complejos.
│
├── src/                    # Código principal del backend.
│   ├── data_processing.py  # Limpieza, features y clustering.
│   ├── train.py            # Entrenamiento y validación del modelo.
│   └── inference.py        # Predicciones (inferencia).
│
├── app/                    # Frontend en Streamlit.
│   └── app.py              
│
└── models/                 # Modelos entrenados que quedan guardados.
    ├── stacking_model.pkl  
    ├── features.pkl        
    └── xgb_simple_shap.pkl 

````

## Instalación y Configuración

Se utiliza **Pipenv** para mantener las dependencias controladas, y **KaggleHub** para obtener el dataset sin necesidad de descargarlo manualmente.

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

## Manual de Ejecución

El proyecto incluye **scripts automatizados** en Pipenv para simplificar el ciclo de desarrollo. Al ejecutar los comandos con `pipenv run` **no es necesario activar el shell manualmente.**


### A. Entrenamiento del Modelo (Backend)

Este comando descarga automáticamente el dataset desde Kaggle (si no existe), aplica el preprocesamiento (Clustering + Lags), entrena el Ensemble y guarda los modelos en la carpeta `models/`.

```bash
pipenv run train
```
Si ocurre algún problema durante la ejecución, se puede invocar directamente el script:

```bash
pipenv run python src/train.py
```

Salida esperada: los archivos stacking_model.pkl y features.pkl generados en la carpeta `models/`.

### B. Iniciar la Aplicación Web (Frontend)

Despliega la interfaz gráfica que permite interactuar con el modelo y visualizar la explicabilidad (SHAP).

```bash
pipenv run start
```

*Luego de aceptar el mensaje inicial de Streamlit (o simplemente continuar sin ingresar correo), la aplicación se abrirá automáticamente en el navegador, en la dirección: http://localhost:8501*

## Calidad de Código (QA)

Para asegurar que el código sea mantenible y robusto, se utiliza un conjunto de herramientas de análisis estático.
La suite completa puede ejecutarse con:

```bash
pipenv run check-all
```

También es posible ejecutar herramientas individuales:

* **Formato:** `pipenv run format` (Aplica **Black** e **Isort**).
* **Linting:** `pipenv run lint` (Analiza el código con **Pylint**).
* **Tipado:** `pipenv run type-check` (Valida tipos estáticos con **Mypy**).

## Descripción Técnica del Sistema

### 1\. Metodología

El proyecto utiliza la metodología CRISP-DM, cubriendo desde el entendimiento del problema hasta el desarrollo del prototipo.

### 2\. Arquitectura del Modelo (Stacking)

Se implementa un **Ensemble Heterogéneo** con el objetivo de mejorar la capacidad predictiva:

* **Modelos Base:**
  * *Random Forest Regressor:* Captura relaciones no lineales y reduce varianza.
  * *XGBoost:* Optimiza el sesgo mediante Gradient Boosting.
* **Meta-Modelo:**
  * *Regresión Lineal:* combina las predicciones de los modelos base para producir la estimación final.

### 3\. Ingeniería de Características Avanzada

* **Clustering Particional (K-Means):** permite segmentar tiendas según su historial de ventas, agrupando patrones similares.
* **Lags Temporales:** se incorporan variables de rezago (t-1, t-2, t-3) para capturar la dinámica secuencial.
* **Balanceo de Datos:** se aplica una transformación logarítmica (log1p) a la variable objetivo para reducir la asimetría de la distribución.

### 4\. Explicabilidad

El Frontend integra **SHAP (SHapley Additive exPlanations)** para analizar el aporte de cada variable (por ejemplo, precio, categoría o mes) en las predicciones del modelo.

---

## Universidad Andrés Bello - 2025
