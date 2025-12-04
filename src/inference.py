import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional


def load_system() -> Tuple[Optional[Any], Optional[list], Optional[Any]]:
    """Carga el modelo, la lista de features y el modelo SHAP."""
    try:
        model = joblib.load("models/stacking_model.pkl")
        features = joblib.load("models/features.pkl")
        shap_model = joblib.load("models/xgb_simple_shap.pkl")
        return model, features, shap_model
    except FileNotFoundError:
        return None, None, None


def predict_demand(model: Any, input_data: Dict[str, float]) -> float:
    """
    Realiza la predicción con tipos fuertes.
    input_data: diccionario con claves 'item_price', 'shop_cluster', etc.
    """
    df = pd.DataFrame([input_data])

    # Predecir logaritmo
    pred_log = model.predict(df)

    # Invertir transformación logarítmica
    pred_real = np.expm1(pred_log)

    # Retornar float asegurando que no sea negativo
    return float(max(0.0, pred_real[0]))
