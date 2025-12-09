"""
Mocks de servicios externos para testing.
"""

from typing import Dict, Any, Optional


class MockAPIResponse:
    """Mock de respuesta HTTP."""

    def __init__(self, json_data: Dict, status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self) -> Dict:
        """Retorna datos JSON."""
        return self.json_data

    def raise_for_status(self) -> None:
        """Simula validación de status HTTP."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class MockHTTPClient:
    """Mock de cliente HTTP para tests."""

    def __init__(self, response_data: Optional[Dict] = None):
        self.response_data = response_data or {"prediction": 10.5}
        self.last_request_data = None

    def post(self, url: str, json: Dict) -> MockAPIResponse:
        """Simula POST request."""
        self.last_request_data = json
        return MockAPIResponse(self.response_data)

    def get(self, url: str) -> MockAPIResponse:
        """Simula GET request."""
        if "health" in url:
            return MockAPIResponse(
                {
                    "status": "healthy",
                    "models_loaded": True,
                    "model_metrics": {"rolling_windows": [3, 6]},
                }
            )
        return MockAPIResponse({"data": "mock"})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
