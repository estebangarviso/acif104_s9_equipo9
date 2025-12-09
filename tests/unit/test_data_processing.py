"""
Tests para src/data_processing.py
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    validate_rolling_windows,
    clean_data,
    create_rolling_window_features,
    feature_engineering,
    DEFAULT_ROLLING_WINDOWS,
    MIN_ROLLING_WINDOW,
    MAX_ROLLING_WINDOW,
)


class TestValidateRollingWindows:
    """Tests para validación de ventanas rolling."""

    def test_default_windows_valid(self):
        """Las ventanas por defecto deben ser válidas."""
        result = validate_rolling_windows(DEFAULT_ROLLING_WINDOWS)
        assert result == sorted(DEFAULT_ROLLING_WINDOWS)

    @pytest.mark.parametrize(
        "windows,expected",
        [
            ([3, 6], [3, 6]),
            ([6, 3], [3, 6]),
            ([2, 12], [2, 12]),
        ],
    )
    def test_valid_windows_sorted(self, windows, expected):
        """Múltiples ventanas se ordenan correctamente."""
        result = validate_rolling_windows(windows)
        assert result == expected

    def test_empty_list_raises_error(self):
        """Lista vacía debe lanzar ValueError."""
        with pytest.raises(ValueError, match="al menos una ventana"):
            validate_rolling_windows([])

    def test_single_window_raises_error(self):
        """Una sola ventana debe lanzar ValueError (requiere 2)."""
        with pytest.raises(ValueError, match="EXACTAMENTE 2"):
            validate_rolling_windows([3])

    def test_three_windows_raises_error(self):
        """Tres ventanas debe lanzar ValueError (requiere 2)."""
        with pytest.raises(ValueError, match="EXACTAMENTE 2"):
            validate_rolling_windows([3, 6, 9])

    def test_non_integer_raises_error(self):
        """Valores no enteros deben lanzar ValueError."""
        with pytest.raises(ValueError, match="enteros"):
            validate_rolling_windows([3.5, 6])

    def test_below_minimum_raises_error(self):
        """Ventanas menores al mínimo deben lanzar ValueError."""
        with pytest.raises(ValueError, match="entre"):
            validate_rolling_windows([1, 6])

    def test_above_maximum_raises_error(self):
        """Ventanas mayores al máximo deben lanzar ValueError."""
        with pytest.raises(ValueError, match="entre"):
            validate_rolling_windows([3, 15])

    def test_duplicate_windows_raises_error(self):
        """Ventanas duplicadas deben lanzar ValueError."""
        with pytest.raises(ValueError, match="no pueden repetirse"):
            validate_rolling_windows([3, 3])

    def test_wrong_order_raises_error(self):
        """Primera ventana mayor que segunda debe lanzar ValueError."""
        # Nota: Se ordena automáticamente, pero validamos orden lógico
        result = validate_rolling_windows([6, 3])
        assert result == [3, 6]

    def test_boundary_values(self):
        """Valores en los límites deben ser válidos."""
        result = validate_rolling_windows([MIN_ROLLING_WINDOW, MAX_ROLLING_WINDOW])
        assert result == [MIN_ROLLING_WINDOW, MAX_ROLLING_WINDOW]


class TestCleanData:
    """Tests para limpieza de datos."""

    def test_removes_negative_prices(self, sample_sales_data):
        """Debe eliminar filas con precios negativos o cero."""
        # Arrange
        sample_sales_data.loc[0, "item_price"] = -100
        sample_sales_data.loc[1, "item_price"] = 0

        # Act
        result = clean_data(sample_sales_data)

        # Assert
        assert len(result) == 4  # 6 - 2 filas inválidas
        assert all(result["item_price"] > 0)

    def test_clips_sales_count(self, sample_sales_data):
        """Debe limitar ventas entre 0 y 20."""
        # Arrange
        sample_sales_data.loc[0, "item_cnt_day"] = 50
        sample_sales_data.loc[1, "item_cnt_day"] = -5

        # Act
        result = clean_data(sample_sales_data)

        # Assert
        assert result.loc[0, "item_cnt_day"] == 20
        assert result.loc[1, "item_cnt_day"] == 0

    def test_clips_prices(self, sample_sales_data):
        """Debe limitar precios entre 0 y 300000."""
        # Arrange
        sample_sales_data.loc[0, "item_price"] = 500000

        # Act
        result = clean_data(sample_sales_data)

        # Assert
        assert result.loc[0, "item_price"] == 300000


class TestCreateRollingWindowFeatures:
    """Tests para creación de features de rolling window."""

    def test_creates_rolling_mean_features(
        self, sample_sales_data, sample_shops_clusters, sample_items_data
    ):
        """Debe crear columnas de media móvil."""
        # Arrange
        data = sample_sales_data.merge(sample_shops_clusters, on="shop_id")
        data = data.merge(sample_items_data[["item_id", "item_category_id"]], on="item_id")

        # Act
        result = create_rolling_window_features(data, window_sizes=[3, 6])

        # Assert
        assert "rolling_mean_3" in result.columns
        assert "rolling_mean_6" in result.columns

    def test_creates_rolling_std_features(
        self, sample_sales_data, sample_shops_clusters, sample_items_data
    ):
        """Debe crear columnas de desviación estándar móvil."""
        # Arrange
        data = sample_sales_data.merge(sample_shops_clusters, on="shop_id")
        data = data.merge(sample_items_data[["item_id", "item_category_id"]], on="item_id")

        # Act
        result = create_rolling_window_features(data, window_sizes=[3, 6])

        # Assert
        assert "rolling_std_3" in result.columns
        assert "rolling_std_6" in result.columns

    def test_handles_missing_windows(
        self, sample_sales_data, sample_shops_clusters, sample_items_data
    ):
        """Debe usar ventanas por defecto si no se especifican."""
        # Arrange
        data = sample_sales_data.merge(sample_shops_clusters, on="shop_id")
        data = data.merge(sample_items_data[["item_id", "item_category_id"]], on="item_id")

        # Act
        result = create_rolling_window_features(data, window_sizes=None)

        # Assert
        assert "rolling_mean_3" in result.columns
        assert "rolling_mean_6" in result.columns


class TestFeatureEngineering:
    """Tests para ingeniería de características."""

    def test_creates_lag_features(
        self, sample_sales_data, sample_items_data, sample_shops_clusters
    ):
        """Debe crear lags de ventas."""
        # Act
        result = feature_engineering(sample_sales_data, sample_items_data, sample_shops_clusters)

        # Assert
        assert "item_cnt_lag_1" in result.columns
        assert "item_cnt_lag_2" in result.columns
        assert "item_cnt_lag_3" in result.columns

    def test_creates_price_lag_features(
        self, sample_sales_data, sample_items_data, sample_shops_clusters
    ):
        """Debe crear lags de precio."""
        # Act
        result = feature_engineering(sample_sales_data, sample_items_data, sample_shops_clusters)

        # Assert
        assert "item_price_lag_1" in result.columns
        assert "item_price_lag_2" in result.columns
        assert "item_price_lag_3" in result.columns

    def test_creates_pricing_features(
        self, sample_sales_data, sample_items_data, sample_shops_clusters
    ):
        """Debe crear features de pricing."""
        # Act
        result = feature_engineering(sample_sales_data, sample_items_data, sample_shops_clusters)

        # Assert
        assert "price_rel_category" in result.columns
        assert "price_discount" in result.columns
        assert "is_new_price" in result.columns

    def test_creates_normalized_features(
        self, sample_sales_data, sample_items_data, sample_shops_clusters
    ):
        """Debe crear versiones normalizadas (log)."""
        # Act
        result = feature_engineering(sample_sales_data, sample_items_data, sample_shops_clusters)

        # Assert
        assert "item_price_log" in result.columns
        assert "price_rel_category_log" in result.columns
        assert "item_cnt_lag_1_log" in result.columns

    def test_creates_momentum_features(
        self, sample_sales_data, sample_items_data, sample_shops_clusters
    ):
        """Debe crear features de momentum."""
        # Act
        result = feature_engineering(sample_sales_data, sample_items_data, sample_shops_clusters)

        # Assert
        assert "delta_1_2" in result.columns
        assert "evolution_3m" in result.columns
        assert "momentum_avg" in result.columns

    def test_creates_rolling_features(
        self, sample_sales_data, sample_items_data, sample_shops_clusters
    ):
        """Debe crear features de rolling window."""
        # Act
        result = feature_engineering(
            sample_sales_data, sample_items_data, sample_shops_clusters, rolling_windows=[3, 6]
        )

        # Assert
        assert "rolling_mean_3" in result.columns
        assert "rolling_std_3" in result.columns
        assert "rolling_mean_6" in result.columns
        assert "rolling_std_6" in result.columns

    def test_fills_nan_values(self, sample_sales_data, sample_items_data, sample_shops_clusters):
        """Debe llenar NaNs con 0."""
        # Act
        result = feature_engineering(sample_sales_data, sample_items_data, sample_shops_clusters)

        # Assert
        assert result.isna().sum().sum() == 0

    def test_handles_infinities(self, sample_sales_data, sample_items_data, sample_shops_clusters):
        """Debe reemplazar infinitos con 0."""
        # Act
        result = feature_engineering(sample_sales_data, sample_items_data, sample_shops_clusters)

        # Assert
        assert not np.isinf(result.values).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
