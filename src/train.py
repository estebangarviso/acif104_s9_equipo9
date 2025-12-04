import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import data_processing

# Crear carpeta models si no existe (en el directorio padre)
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)


def train_models() -> None:
    # 1. Obtener datos procesados
    train_data, val_data, _ = data_processing.prepare_full_pipeline()

    features = [
        "shop_cluster",
        "item_category_id",
        "item_price",
        "item_cnt_lag_1",
        "item_cnt_lag_2",
        "item_cnt_lag_3",
    ]
    target = "target_log"

    # Si val está vacío, usar una estrategia de validación alternativa
    if len(val_data) == 0:
        print("⚠️ Val vacío. Usando últimos 20% de train como validación...")
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    X_train = train_data[features]
    y_train = train_data[target]
    X_val = val_data[features]
    y_val = val_data[target]

    print(f"Entrenando con {X_train.shape[0]} muestras...")

    # 2. Definir Modelos Base
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
        ("xgb", XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)),
    ]

    # 3. Stacking (Ensemble Learning)
    stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

    print("Entrenando Stacking Ensemble...")
    stacking_model.fit(X_train, y_train)

    # 4. Evaluación
    preds_log = stacking_model.predict(X_val)
    preds = np.expm1(preds_log)  # Invertir logaritmo
    y_true = np.expm1(y_val)

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)

    print("--- Resultados Validación ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    # 5. Guardar Modelo y columnas
    joblib.dump(stacking_model, os.path.join(models_dir, "stacking_model.pkl"))
    joblib.dump(features, os.path.join(models_dir, "features.pkl"))
    print(f"Modelo guardado en {os.path.join(models_dir, 'stacking_model.pkl')}")

    # Guardar un modelo simple XGBoost para SHAP
    xgb_simple = XGBRegressor(n_estimators=50, max_depth=5).fit(X_train, y_train)
    joblib.dump(xgb_simple, os.path.join(models_dir, "xgb_simple_shap.pkl"))


if __name__ == "__main__":
    train_models()
