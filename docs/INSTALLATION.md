# Guía de Instalación y Configuración

## Prerrequisitos

* **Python:** Versión 3.10 (Requerido)
* **Gestor de Paquetes:** `pipenv` (recomendado) o `pip`

```bash
# Instalar Pipenv
pip install pipenv
```

## Clonar el Repositorio

```bash
git clone https://github.com/estebangarviso/acif104_s9_equipo9.git
cd acif104_s9_equipo9
```

## Instalar Dependencias

### Opción A: Con Pipenv (Recomendado)

```bash
# Instalar dependencias y generar requirements.txt automáticamente
pipenv install --ignore-pipfile

# Para desarrollo (incluye herramientas de QA)
pipenv install --dev
```

### Opción B: Con pip tradicional

```bash
# Crear entorno virtual
python3.10 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias desde requirements.txt
pip install -r requirements.txt

# Para desarrollo
pip install -r requirements-dev.txt
```

## Activar Git Hooks (Automático con Pipenv)

Los hooks ya están configurados para sincronizar `requirements.txt` automáticamente cuando hagas commit de cambios en `Pipfile`.

## Manual de Comandos

### Comandos Principales

| Comando                        | Descripción                                                                                                                                                                                                                                                                 |
| :----------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`pipenv run api`**           | **Inicia el Backend (API REST).** FastAPI en [http://localhost:8000](http://localhost:8000). Documentación interactiva: [http://localhost:8000/docs](http://localhost:8000/docs)                                                                                            |
| **`pipenv run start`**         | **Inicia el Frontend (Streamlit).** Interfaz web con explicabilidad SHAP en [http://localhost:8501](http://localhost:8501). **Requiere que el Backend esté corriendo.**                                                                                                     |
| **`pipenv run train`**         | **Ejecuta Pipeline de Entrenamiento.** 1. Descarga/carga datos 2. Clustering K-Means + Features 3. Entrena 5 modelos (RF, XGB, MLP, LSTM, Stacking) 4. Guarda en `/models` con métricas en `metrics.json`. **Ventanas rolling parametrizables** (ver documentación técnica) |
| `pipenv run sync-requirements` | Sincroniza `requirements.txt` manualmente desde `Pipfile`                                                                                                                                                                                                                   |
| `pipenv run install`           | Instala nueva dependencia y sincroniza automáticamente                                                                                                                                                                                                                      |

### Calidad de Código (QA)

| Comando                    | Descripción                                                                   |
| :------------------------- | :---------------------------------------------------------------------------- |
| **`pipenv run check-all`** | **Suite Completa.** Ejecuta formato, linting y chequeo de tipos en secuencia. |
| `pipenv run format`        | Aplica formato automático con **Black** e **Isort**.                          |
| `pipenv run lint`          | Analiza el código estáticamente con **Pylint**.                               |
| `pipenv run type-check`    | Valida tipos estáticos con **Mypy**.                                          |

### Gestión de Dependencias

**Agregar nuevas dependencias (con sincronización automática):**

```bash
# Método 1: Script automático (recomendado)
pipenv run install requests

# Método 2: Instalación tradicional + sincronización manual
pipenv install requests
pipenv run sync-requirements

# Método 3: Dependencias de desarrollo
pipenv install --dev pytest
pipenv run sync-requirements
```

**Los `requirements.txt` se actualizan automáticamente:**
- ✅ Al hacer `git commit` con cambios en `Pipfile` (git hook)
- ✅ Al usar `pipenv run install` (script personalizado)

### Testing y Validación

**Tests Unitarios con pytest:**

```bash
# Ejecutar todos los tests
pipenv run pytest tests/ -v

# Ejecutar tests de rolling windows específicamente
pipenv run pytest tests/test_rolling_windows.py -v

# Ejecutar un test específico
pipenv run pytest tests/test_rolling_windows.py::TestRollingWindowsValidation::test_valid_default_windows -v
```

**Debugging en VS Code:**
- Usa las configuraciones de launch para depurar tests específicos
- `Python: Run All Tests` - Ejecuta todos los tests del proyecto
- `Python: Run Current Test File` - Ejecuta el archivo de test actual
- `Python: Run Test at Cursor` - Ejecuta solo el test bajo el cursor
