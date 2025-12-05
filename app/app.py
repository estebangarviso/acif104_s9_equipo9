import streamlit as st
import pandas as pd
import shap
import sys
import os

# Agregar directorio raíz al path para importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import load_system, predict_demand

# Configuración de página
st.set_page_config(page_title="Predicción de Demanda AI", layout="wide")

st.title("Sistema de Predicción de Demanda E-commerce")
st.markdown("---")

# Cargar Backend
model, features, shap_model = load_system()


# Helper para renderizar gráficos SHAP
def st_shap(plot, height=None):
    """Helper para renderizar gráficos JS de SHAP en Streamlit"""
    import streamlit.components.v1 as components

    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height if height else 400)


if model is None:
    st.error("Error: No se encontraron los modelos. Ejecuta 'python src/train.py' primero.")
else:
    # --- Sidebar: Inputs del Usuario ---
    st.sidebar.header("Parámetros de Entrada")

    # Simulamos inputs (en producción vendrían de una base de datos)
    shop_cluster = st.sidebar.selectbox("Cluster de Tienda (Segmento)", [0, 1, 2])
    item_category = st.sidebar.number_input("ID Categoría", min_value=0, max_value=83, value=19)
    item_price = st.sidebar.slider("Precio del Producto", 0.0, 50000.0, 1500.0)

    st.sidebar.subheader("Historial de Ventas (Lags)")
    lag_1 = st.sidebar.number_input("Ventas Mes Anterior", 0, 100, 5)
    lag_2 = st.sidebar.number_input("Ventas hace 2 Meses", 0, 100, 4)
    lag_3 = st.sidebar.number_input("Ventas hace 3 Meses", 0, 100, 4)

    # Botón de Predicción
    if st.sidebar.button("Predecir Demanda"):
        # Construir diccionario de datos
        input_data = {
            "shop_cluster": shop_cluster,
            "item_category_id": item_category,
            "item_price": item_price,
            "item_cnt_lag_1": lag_1,
            "item_cnt_lag_2": lag_2,
            "item_cnt_lag_3": lag_3,
        }

        # --- Backend Call ---
        prediction = predict_demand(model, input_data)

        # --- Frontend Display ---
        col1, col2 = st.columns([1, 2])

        with col1:
            st.success("Predicción Exitosa")
            st.metric(label="Demanda Estimada (Mes Siguiente)", value=f"{prediction:.2f} Unidades")

        with col2:
            st.subheader("🔍 Explicabilidad del Modelo (SHAP)")
            st.info(
                "Este gráfico muestra cómo cada variable empujó la predicción hacia arriba (rojo) o abajo (azul)."
            )

            # Generar SHAP Force Plot
            # Usamos el modelo simple XGBoost porque TreeExplainer es rápido y compatible
            explainer = shap.TreeExplainer(shap_model)
            features_df = pd.DataFrame([input_data])
            shap_values = explainer.shap_values(features_df)

            # Renderizar Force Plot (sin matplotlib para obtener HTML interactivo)
            st_shap(shap.force_plot(explainer.expected_value, shap_values[0], features_df.iloc[0]))
