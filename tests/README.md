# Guía de Testing - Sistema Predictivo de Demanda

## Estructura de Tests

```
tests/
├── README.md                    # Esta guía
├── conftest.py                  # Fixtures compartidos y configuración pytest
├── mocks/                       # Mocks y datos de prueba
│   ├── __init__.py
│   ├── mock_data.py            # Datos sintéticos para tests
│   └── mock_services.py        # Mocks de servicios externos
│
├── unit/                        # Tests unitarios (componentes aislados)
│   ├── __init__.py
│   ├── test_data_processing.py # Tests de src/data_processing.py
│   ├── test_inference.py       # Tests de src/inference.py
│   │
│   ├── services/               # Tests de app/services/
│   │   ├── __init__.py
│   │   ├── test_pricing_service.py
│   │   ├── test_prediction_service.py
│   │   └── test_trend_analyzer.py
│   │
│   ├── components/             # Tests de app/components/
│   │   ├── __init__.py
│   │   ├── test_chart_builder.py
│   │   ├── test_dataframe_builder.py
│   │   └── test_shap_renderer.py
│   │
│   └── utils/                  # Tests de utilidades
│       ├── __init__.py
│       └── test_state_manager.py
│
├── integration/                # Tests de integración (múltiples componentes)
│   ├── __init__.py
│   ├── test_api_integration.py
│   └── test_prediction_flow.py
│
└── e2e/                        # Tests end-to-end (flujo completo)
    ├── __init__.py
    └── test_complete_prediction.py
```

## Principios de Testing

### 1. **Aislamiento (Unit Tests)**
- Cada test debe ser independiente
- Usa mocks para dependencias externas
- No accedas a archivos reales ni APIs

### 2. **AAA Pattern (Arrange-Act-Assert)**
```python
def test_example():
    # Arrange: Preparar datos y mocks
    service = PricingService(mock_prices, ...)
    
    # Act: Ejecutar la función a testear
    result = service.get_price(category_id=1)
    
    # Assert: Verificar el resultado
    assert result == expected_value
```

### 3. **Nomenclatura Clara**
```python
# ✅ Bueno
def test_pricing_service_returns_average_when_category_exists():
    pass

# ❌ Malo
def test_1():
    pass
```

### 4. **Fixtures Reutilizables**
```python
# En conftest.py
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"col": [1, 2, 3]})

# En test
def test_function(sample_dataframe):
    assert len(sample_dataframe) == 3
```

## Comandos de Ejecución

### Ejecutar Todos los Tests
```bash
pipenv run pytest tests/ -v
```

### Ejecutar Tests por Categoría
```bash
# Solo tests unitarios
pipenv run pytest tests/unit/ -v

# Solo tests de integración
pipenv run pytest tests/integration/ -v

# Solo tests end-to-end
pipenv run pytest tests/e2e/ -v
```

### Ejecutar Tests con Cobertura
```bash
pipenv run pytest tests/ --cov=src --cov=app --cov-report=html
```

### Ejecutar Tests Específicos
```bash
# Un archivo
pipenv run pytest tests/unit/test_data_processing.py -v

# Una clase
pipenv run pytest tests/unit/test_data_processing.py::TestFeatureEngineering -v

# Un test específico
pipenv run pytest tests/unit/test_data_processing.py::TestFeatureEngineering::test_price_rel_category -v
```

### Ejecutar Tests con Filtros
```bash
# Tests que contengan "price" en el nombre
pipenv run pytest tests/ -k "price" -v

# Tests marcados como "slow"
pipenv run pytest tests/ -m "slow" -v
```

## Mejores Prácticas

### 1. Usa Fixtures para Datos Repetitivos
```python
# conftest.py
@pytest.fixture
def mock_sales_data():
    return pd.DataFrame({
        "date_block_num": [1, 1, 2],
        "shop_id": [1, 2, 1],
        "item_id": [100, 100, 100],
        "item_price": [1500, 1600, 1550],
        "item_cnt_day": [5, 3, 7]
    })
```

### 2. Mockea Dependencias Externas
```python
# test_prediction_service.py
def test_predict_calls_api(mocker):
    mock_post = mocker.patch("httpx.Client.post")
    mock_post.return_value.json.return_value = {"prediction": 10.5}
    
    service = PredictionService(None)
    result = service.predict({"shop_cluster": 0})
    
    assert result == 10.5
    mock_post.assert_called_once()
```

### 3. Parametriza Tests con Múltiples Casos
```python
@pytest.mark.parametrize("windows,expected", [
    ([3, 6], [3, 6]),
    ([6, 3], [3, 6]),
    ([3], [3]),
])
def test_validate_rolling_windows(windows, expected):
    result = validate_rolling_windows(windows)
    assert result == expected
```

### 4. Usa Marcadores para Categorizar Tests
```python
@pytest.mark.slow
def test_expensive_computation():
    # Test que toma mucho tiempo
    pass

@pytest.mark.integration
def test_api_connection():
    # Test de integración
    pass
```

### 5. Verifica Excepciones
```python
def test_invalid_input_raises_error():
    with pytest.raises(ValueError, match="debe ser positivo"):
        function_that_validates(-1)
```

## Estructura de un Test Completo

```python
"""
Tests para PricingService.
"""

import pytest
from app.services import PricingService


class TestPricingService:
    """Suite de tests para PricingService."""
    
    @pytest.fixture
    def mock_category_prices(self):
        """Fixture con precios mock."""
        return {
            1: 1500.0,
            2: 2500.0,
            3: 500.0
        }
    
    @pytest.fixture
    def service(self, mock_category_prices):
        """Fixture del servicio configurado."""
        return PricingService(
            cat_prices=mock_category_prices,
            default_price=1000.0,
            min_price=0.0,
            max_price=50000.0,
            range_multiplier=0.33,
            max_multiplier=3.0
        )
    
    def test_get_price_for_existing_category(self, service):
        """Debe retornar el precio promedio de la categoría."""
        # Act
        price = service.get_price_for_category(1)
        
        # Assert
        assert price == 1500.0
    
    def test_get_price_for_missing_category(self, service):
        """Debe retornar el precio por defecto."""
        # Act
        price = service.get_price_for_category(999)
        
        # Assert
        assert price == 1000.0
    
    def test_update_price_range_calculates_correctly(self, service):
        """Debe calcular el rango como ±200% del precio."""
        # Act
        service.update_price_for_category(1)  # precio = 1500
        min_price, max_price = service.get_current_price_range()
        
        # Assert
        assert min_price == pytest.approx(495.0)  # 1500 * 0.33
        assert max_price == pytest.approx(4500.0) # 1500 * 3.0
```

## Cobertura de Código

### Objetivo: ≥ 80% de cobertura

Para verificar la cobertura:
```bash
pipenv run pytest tests/ --cov=src --cov=app --cov-report=term-missing
```

### Interpretar el Reporte
```
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
src/data_processing.py    150     10    93%   45-47, 89-92
app/services/pricing.py    45      2    96%   12, 34
```

- **Stmts**: Total de líneas ejecutables
- **Miss**: Líneas no cubiertas por tests
- **Cover**: Porcentaje de cobertura
- **Missing**: Números de línea sin cobertura

## Debugging de Tests

### 1. Usar `-s` para Ver Prints
```bash
pipenv run pytest tests/unit/test_pricing_service.py -s
```

### 2. Usar `--pdb` para Debugger Interactivo
```bash
pipenv run pytest tests/unit/test_pricing_service.py --pdb
```

### 3. Usar VS Code Debugger
- Coloca breakpoints en el código
- Ejecuta "Python: Debug Test at Cursor"

## Convenciones de Nombres

| Tipo    | Patrón                    | Ejemplo                        |
| ------- | ------------------------- | ------------------------------ |
| Archivo | `test_*.py`               | `test_pricing_service.py`      |
| Clase   | `Test*`                   | `TestPricingService`           |
| Método  | `test_*`                  | `test_get_price_returns_float` |
| Fixture | `*_fixture` o descriptivo | `mock_sales_data`              |

## Referencias

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-mock](https://pytest-mock.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)

---

**Desarrollado por:** Equipo 9 - ACIF104  
**Universidad:** Andrés Bello  
**Año:** 2025
