"""
Configuración global de pytest y fixtures compartidos.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from typing import Dict

# Agregar el directorio raíz al PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


# ============================================================================
# CONFIGURACIÓN DE PYTEST
# ============================================================================


def pytest_configure(config):
    """Configuración personalizada de pytest."""
    config.addinivalue_line("markers", "slow: marca tests lentos")
    config.addinivalue_line("markers", "integration: tests de integración")
    config.addinivalue_line("markers", "e2e: tests end-to-end")


# ============================================================================
# FIXTURES DE DATOS
# ============================================================================


@pytest.fixture
def sample_sales_data() -> pd.DataFrame:
    """DataFrame de ventas sintético para tests."""
    return pd.DataFrame(
        {
            "date_block_num": [0, 0, 1, 1, 2, 2],
            "shop_id": [1, 2, 1, 2, 1, 2],
            "item_id": [100, 100, 100, 100, 100, 100],
            "item_price": [1500, 1600, 1550, 1650, 1500, 1700],
            "item_cnt_day": [5, 3, 7, 4, 6, 5],
        }
    )


@pytest.fixture
def sample_items_data() -> pd.DataFrame:
    """DataFrame de items sintético."""
    return pd.DataFrame(
        {
            "item_id": [100, 101, 102],
            "item_name": ["Product A", "Product B", "Product C"],
            "item_category_id": [40, 41, 40],
        }
    )


@pytest.fixture
def sample_shops_data() -> pd.DataFrame:
    """DataFrame de tiendas sintético."""
    return pd.DataFrame({"shop_id": [1, 2, 3], "shop_name": ["Shop A", "Shop B", "Shop C"]})


@pytest.fixture
def sample_categories_data() -> pd.DataFrame:
    """DataFrame de categorías sintético."""
    return pd.DataFrame(
        {"item_category_id": [40, 41, 42], "item_category_name": ["Electronics", "Books", "Games"]}
    )


@pytest.fixture
def sample_shops_clusters() -> pd.DataFrame:
    """DataFrame de clusters de tiendas."""
    return pd.DataFrame({"shop_id": [1, 2, 3], "shop_cluster": [0, 1, 2]})


@pytest.fixture
def mock_category_prices() -> Dict[int, float]:
    """Diccionario de precios por categoría."""
    return {40: 1500.0, 41: 2500.0, 42: 500.0}


@pytest.fixture
def sample_input_data() -> Dict:
    """Datos de entrada para predicción."""
    return {
        "shop_cluster": 1,
        "item_category_id": 40,
        "item_price": 1500.0,
        "item_cnt_lag_1": 10,
        "item_cnt_lag_2": 8,
        "item_cnt_lag_3": 12,
        "rolling_windows": [3, 6],
    }


# ============================================================================
# FIXTURES DE NUMPY ARRAYS
# ============================================================================


@pytest.fixture
def sample_shap_values() -> np.ndarray:
    """Valores SHAP sintéticos."""
    return np.array([[0.5, -0.3, 0.8, 0.2, -0.1, 0.4]])


@pytest.fixture
def sample_feature_values() -> np.ndarray:
    """Valores de features para SHAP."""
    return np.array([[1, 40, 1500, 10, 8, 12]])


# ============================================================================
# FIXTURES DE CONFIGURACIÓN
# ============================================================================


@pytest.fixture
def default_config() -> Dict:
    """Configuración por defecto del sistema."""
    return {
        "default_price": 1500.0,
        "min_price": 0.0,
        "max_price": 50000.0,
        "price_range_multiplier": 0.33,
        "price_range_max_multiplier": 3.0,
        "rolling_windows": [3, 6],
    }


@pytest.fixture
def cluster_map() -> Dict[int, str]:
    """Mapeo de clusters."""
    return {
        0: "Tienda Pequeña / Kiosco (Bajo Volumen)",
        1: "Supermercado / Mall (Volumen Medio)",
        2: "Megatienda / Online (Alto Volumen)",
    }
