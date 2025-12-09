"""
Tests para app/services/prediction_service.py
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from app.services import PredictionService
from tests.mocks import MockHTTPClient


class TestPredictionService:
    """Suite de tests para PredictionService."""

    @pytest.fixture
    def mock_shap_model(self):
        """Mock de modelo SHAP."""
        mock_model = Mock()
        mock_model.n_features_in_ = 10
        return mock_model

    @pytest.fixture
    def service(self, mock_shap_model):
        """Fixture del servicio configurado."""
        return PredictionService(shap_model=mock_shap_model, api_url="http://localhost:8000")

    def test_predict_calls_api_and_returns_prediction(self, service, mocker):
        """Debe llamar a la API y retornar la predicción."""
        # Arrange
        mock_client = MockHTTPClient({"prediction": 10.5})
        mocker.patch("httpx.Client", return_value=mock_client)
        input_data = {"shop_cluster": 1, "item_price": 1500}

        # Act
        result = service.predict(input_data)

        # Assert
        assert result == 10.5
        assert mock_client.last_request_data == input_data

    def test_predict_raises_error_on_api_failure(self, service, mocker):
        """Debe manejar errores de la API."""
        # Arrange
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.post = Mock(return_value=mock_response)
        mocker.patch("httpx.Client", return_value=mock_client)

        # Mock de streamlit para evitar st.stop()
        mocker.patch("streamlit.error")
        mocker.patch("streamlit.stop")

        # Act & Assert
        service.predict({"shop_cluster": 1})

    def test_check_api_health_returns_true_when_healthy(self, service, mocker):
        """Debe retornar True si la API está saludable."""
        # Arrange
        mock_client = MockHTTPClient()
        mocker.patch("httpx.Client", return_value=mock_client)

        # Act
        result = service.check_api_health()

        # Assert
        assert result is True

    def test_check_api_health_returns_false_on_error(self, service, mocker):
        """Debe retornar False si hay error de conexión."""
        # Arrange
        mock_client = Mock()
        mock_client.__enter__ = Mock(side_effect=Exception("Connection failed"))
        mock_client.__exit__ = Mock(return_value=False)
        mocker.patch("httpx.Client", return_value=mock_client)

        # Act
        result = service.check_api_health()

        # Assert
        assert result is False

    def test_calculate_shap_values_returns_correct_format(self, service, mocker, sample_input_data):
        """Debe retornar SHAP values, DataFrame y valor base."""
        # Arrange
        mock_explainer = Mock()
        mock_explainer.shap_values = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
        mock_explainer.expected_value = 5.0
        mocker.patch("shap.TreeExplainer", return_value=mock_explainer)

        # Mockear _fetch_model_schema
        service._rolling_windows = [3, 6]
        service._feature_names = ["shop_cluster", "item_category_id", "item_price"]

        # Act
        shap_values, feat_df, expected_value = service.calculate_shap_values(sample_input_data)

        # Assert
        assert isinstance(shap_values, np.ndarray)
        assert isinstance(feat_df, pd.DataFrame)
        assert isinstance(expected_value, (int, float))

    def test_prepare_features_includes_pricing_features(self, service, sample_input_data):
        """Debe preparar features incluyendo pricing."""
        # Arrange
        service._rolling_windows = [3, 6]

        # Act
        result = service._prepare_features_for_shap(sample_input_data)

        # Assert
        assert "item_price_log" in result
        assert "price_rel_category" in result
        assert "revenue_potential_log" in result

    def test_prepare_features_includes_rolling_features(self, service, sample_input_data):
        """Debe calcular rolling features si no están presentes."""
        # Arrange
        service._rolling_windows = [3, 6]

        # Act
        result = service._prepare_features_for_shap(sample_input_data)

        # Assert
        assert "rolling_mean_3" in result
        assert "rolling_std_3" in result
        assert "rolling_mean_6" in result
        assert "rolling_std_6" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
