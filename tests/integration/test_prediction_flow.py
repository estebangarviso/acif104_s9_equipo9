"""
Tests de integración para flujo completo de predicción.
"""

import pytest
from unittest.mock import Mock
from tests.mocks import MockHTTPClient, create_mock_feature_engineered_data


@pytest.mark.integration
class TestPredictionFlow:
    """Tests de integración del flujo de predicción."""

    def test_complete_prediction_flow_with_mocked_api(self, mocker, sample_input_data):
        """Debe completar el flujo de predicción usando API mockeada."""
        # Arrange
        from app.services import PredictionService

        mock_client = MockHTTPClient({"prediction": 12.5})
        mocker.patch("httpx.Client", return_value=mock_client)

        mock_shap_model = Mock()
        service = PredictionService(shap_model=mock_shap_model)

        # Act
        result = service.predict(sample_input_data)

        # Assert
        assert result == 12.5
        assert mock_client.last_request_data == sample_input_data

    def test_pricing_and_prediction_services_integration(self, mocker, mock_category_prices):
        """Debe integrar pricing service con prediction service."""
        # Arrange
        from app.services import PricingService, PredictionService

        pricing_service = PricingService(
            cat_prices=mock_category_prices,
            default_price=1000.0,
            default_min=0.0,
            default_max=50000.0,
            min_multiplier=0.33,
            max_multiplier=3.0,
        )

        mock_shap_model = Mock()
        prediction_service = PredictionService(shap_model=mock_shap_model)

        mock_client = MockHTTPClient({"prediction": 15.0})
        mocker.patch("httpx.Client", return_value=mock_client)

        # Act
        pricing_service.update_price_for_category(40)
        current_price, _, _ = pricing_service.get_current_price_range()

        prediction = prediction_service.predict(
            {
                "shop_cluster": 1,
                "item_category_id": 40,
                "item_price": current_price,
                "item_cnt_lag_1": 10,
                "item_cnt_lag_2": 8,
                "item_cnt_lag_3": 12,
            }
        )

        # Assert
        assert current_price > 0  # Precio actualizado desde la categoría
        assert prediction == 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
