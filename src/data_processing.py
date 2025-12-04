import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.cluster import KMeans
from typing import Tuple


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Descarga los datasets automáticamente desde KaggleHub y los carga en Pandas.
    Dataset: jaklinmalkoc/predict-future-sales-retail-dataset-en
    """
    print("⏳ Descargando dataset desde KaggleHub...")

    # Descarga el dataset y obtiene la ruta local de la caché
    path = kagglehub.dataset_download("jaklinmalkoc/predict-future-sales-retail-dataset-en")
    print(f"✅ Dataset descargado en: {path}")

    # Cargar archivos usando la ruta obtenida
    sales = pd.read_csv(os.path.join(path, "sales_train.csv"))
    items = pd.read_csv(os.path.join(path, "items.csv"))
    shops = pd.read_csv(os.path.join(path, "shops.csv"))
    cats = pd.read_csv(os.path.join(path, "item_categories.csv"))

    return sales, items, shops, cats


def clean_data(sales: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica y tratamiento de outliers (Clipping)."""
    # Eliminar precios negativos
    sales = sales[sales["item_price"] > 0]

    # Clipping: Limitar ventas extremas (Balanceo de datos)
    sales["item_cnt_day"] = sales["item_cnt_day"].clip(0, 20)
    sales["item_price"] = sales["item_price"].clip(0, 300000)

    # Convertir fecha
    if "date" in sales.columns:
        sales["date"] = pd.to_datetime(sales["date"], dayfirst=True)

    return sales


def generate_clusters(
    shops: pd.DataFrame, sales: pd.DataFrame, n_clusters: int = 2
) -> pd.DataFrame:
    """
    APRENDIZAJE NO SUPERVISADO (Clustering).
    Agrupa tiendas según su volumen de ventas para usarlo como feature.
    """
    shop_sales = sales.groupby("shop_id")["item_cnt_day"].sum().reset_index()

    # K-Means para agrupar tiendas por volumen de venta
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    shop_sales["shop_cluster"] = kmeans.fit_predict(shop_sales[["item_cnt_day"]])

    return shop_sales[["shop_id", "shop_cluster"]]


def feature_engineering(
    sales: pd.DataFrame, items: pd.DataFrame, shops_clusters: pd.DataFrame
) -> pd.DataFrame:
    """Genera la matriz de entrenamiento con Lags."""
    # Agrupar por mes (date_block_num), tienda e item
    monthly_sales = (
        sales.groupby(["date_block_num", "shop_id", "item_id"])
        .agg({"item_cnt_day": "sum", "item_price": "mean"})
        .reset_index()
    )

    # Clip de ventas mensuales (Target range 0-20)
    monthly_sales["item_cnt_day"] = monthly_sales["item_cnt_day"].clip(0, 20)

    # Unir con clusters y categorías
    data = monthly_sales.merge(shops_clusters, on="shop_id", how="left")
    data = data.merge(items[["item_id", "item_category_id"]], on="item_id", how="left")

    # Generar Lags (Rezagos)
    data_shifted = data.copy()

    for lag in [1, 2, 3]:
        shifted = data_shifted[["date_block_num", "shop_id", "item_id", "item_cnt_day"]].copy()
        shifted.columns = ["date_block_num", "shop_id", "item_id", f"item_cnt_lag_{lag}"]
        shifted["date_block_num"] += lag
        data = data.merge(shifted, on=["date_block_num", "shop_id", "item_id"], how="left")

    # Llenar NaNs generados por los lags con 0
    data = data.fillna(0)

    return data


def prepare_full_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta todo el pipeline y retorna datos listos para Train/Val/Test."""
    sales, items, shops, cats = load_data()

    print("Limpiando datos...")
    sales = clean_data(sales)

    print("Generando Clusters (Unsupervised Learning)...")
    shops_clusters = generate_clusters(shops, sales)

    print("Ingeniería de Características...")
    df_final = feature_engineering(sales, items, shops_clusters)

    # Transformación Logarítmica del Target (Técnica de Balanceo #2)
    df_final["target_log"] = np.log1p(df_final["item_cnt_day"])

    # Separar Train/Val/Test (Simulación temporal)
    # Encontrar el rango de meses disponible
    max_month = df_final["date_block_num"].max()

    # Train: meses hasta max_month-2 | Val: max_month-1 | Test: max_month
    train = df_final[df_final["date_block_num"] < max_month - 1]
    val = df_final[df_final["date_block_num"] == max_month - 1]
    test = df_final[df_final["date_block_num"] == max_month]

    print(f"Train meses: 0-{max_month-2} ({len(train)} registros)")
    print(f"Val mes: {max_month-1} ({len(val)} registros)")
    print(f"Test mes: {max_month} ({len(test)} registros)")

    return train, val, test
