"""
Tests para app/components/dataframe_builder.py
"""

import pytest
import pandas as pd
from app.components import DataFrameBuilder


class TestDataFrameBuilder:
    """Suite de tests para DataFrameBuilder."""

    @pytest.fixture
    def builder(self):
        """Fixture del builder."""
        return DataFrameBuilder()

    def test_create_trend_dataframe_returns_dataframe(self, builder):
        """Debe retornar un DataFrame."""
        # Act
        result = builder.create_trend_dataframe(lag_3=10, lag_2=12, lag_1=15)

        # Assert
        assert isinstance(result, pd.DataFrame)

    def test_create_trend_dataframe_has_correct_columns(self, builder):
        """Debe tener las columnas correctas."""
        # Act
        result = builder.create_trend_dataframe(lag_3=10, lag_2=12, lag_1=15)

        # Assert
        assert "Mes" in result.columns
        assert "Ventas" in result.columns

    def test_create_trend_dataframe_has_three_rows(self, builder):
        """Debe tener 3 filas (t-3, t-2, t-1)."""
        # Act
        result = builder.create_trend_dataframe(lag_3=10, lag_2=12, lag_1=15)

        # Assert
        assert len(result) == 3

    def test_create_trend_dataframe_values_match_input(self, builder):
        """Los valores deben coincidir con los inputs."""
        # Act
        result = builder.create_trend_dataframe(lag_3=10, lag_2=12, lag_1=15)

        # Assert
        assert result["Ventas"].tolist() == [10, 12, 15]

    def test_create_temporal_dataframe_returns_dataframe(self, builder):
        """Debe retornar un DataFrame."""
        # Act
        result = builder.create_temporal_dataframe(lag_3=10, lag_2=12, lag_1=15, prediction=18)

        # Assert
        assert isinstance(result, pd.DataFrame)

    def test_create_temporal_dataframe_has_four_rows(self, builder):
        """Debe tener 4 filas (t-3, t-2, t-1, t)."""
        # Act
        result = builder.create_temporal_dataframe(lag_3=10, lag_2=12, lag_1=15, prediction=18)

        # Assert
        assert len(result) == 4

    def test_create_temporal_dataframe_includes_prediction(self, builder):
        """Debe incluir la predicción como última fila."""
        # Act
        result = builder.create_temporal_dataframe(lag_3=10, lag_2=12, lag_1=15, prediction=18)

        # Assert
        assert result["Ventas"].iloc[-1] == 18

    def test_create_temporal_dataframe_has_correct_labels(self, builder):
        """Debe tener etiquetas de tiempo correctas."""
        # Act
        result = builder.create_temporal_dataframe(lag_3=10, lag_2=12, lag_1=15, prediction=18)

        # Assert
        expected_labels = ["Mes t-3", "Mes t-2", "Mes t-1", "Predicción (t)"]
        assert result["Periodo"].tolist() == expected_labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
