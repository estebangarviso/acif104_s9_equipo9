"""
Tests para app/services/trend_analyzer.py
"""

import pytest
from app.services import TrendAnalyzer


class TestTrendAnalyzer:
    """Suite de tests para TrendAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Fixture del analizador."""
        return TrendAnalyzer()

    def test_calculate_delta_positive(self, analyzer):
        """Debe calcular delta positivo correctamente."""
        # Act
        result = analyzer.calculate_delta(15.0, 10.0)

        # Assert
        assert result == 5.0

    def test_calculate_delta_negative(self, analyzer):
        """Debe calcular delta negativo correctamente."""
        # Act
        result = analyzer.calculate_delta(8.0, 12.0)

        # Assert
        assert result == -4.0

    def test_calculate_delta_zero(self, analyzer):
        """Debe calcular delta cero correctamente."""
        # Act
        result = analyzer.calculate_delta(10.0, 10.0)

        # Assert
        assert result == 0.0

    def test_get_trend_status_positive(self, analyzer):
        """Debe retornar estado de tendencia alcista."""
        # Act
        message, icon, delta_mode = analyzer.get_trend_status(5.0)

        # Assert
        assert "alza" in message.lower() or "trending_up" in icon
        assert delta_mode == "normal"

    def test_get_trend_status_negative(self, analyzer):
        """Debe retornar estado de tendencia bajista."""
        # Act
        message, icon, delta_mode = analyzer.get_trend_status(-5.0)

        # Assert
        assert "baja" in message.lower() or "trending_down" in icon
        assert delta_mode == "normal"

    def test_get_trend_status_neutral(self, analyzer):
        """Debe retornar estado de tendencia neutra."""
        # Act
        message, icon, delta_mode = analyzer.get_trend_status(0.0)

        # Assert
        assert "estable" in message.lower() or "igual" in message.lower()
        assert delta_mode == "off"

    def test_get_chart_colors_positive_trend(self, analyzer):
        """Debe retornar color positivo cuando la predicción sube."""
        # Arrange
        positive_color = "#00FF00"
        negative_color = "#FF0000"
        historical_color = "#808080"

        # Act
        colors = analyzer.get_chart_colors(
            prediction=15.0,
            last_value=10.0,
            historical_color=historical_color,
            positive_color=positive_color,
            negative_color=negative_color,
        )

        # Assert
        assert colors[-1] == positive_color  # Último color es de predicción
        assert all(c == historical_color for c in colors[:-1])

    def test_get_chart_colors_negative_trend(self, analyzer):
        """Debe retornar color negativo cuando la predicción baja."""
        # Arrange
        positive_color = "#00FF00"
        negative_color = "#FF0000"
        historical_color = "#808080"

        # Act
        colors = analyzer.get_chart_colors(
            prediction=8.0,
            last_value=10.0,
            historical_color=historical_color,
            positive_color=positive_color,
            negative_color=negative_color,
        )

        # Assert
        assert colors[-1] == negative_color

    def test_get_chart_colors_returns_correct_length(self, analyzer):
        """Debe retornar lista con 4 colores (3 históricos + 1 predicción)."""
        # Act
        colors = analyzer.get_chart_colors(
            prediction=10.0,
            last_value=10.0,
            historical_color="#808080",
            positive_color="#00FF00",
            negative_color="#FF0000",
        )

        # Assert
        assert len(colors) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
