import pandas as pd
import numpy as np
import kagglehub
import os
import shutil
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
from typing import Tuple, List

# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Archivos requeridos del dataset
REQUIRED_FILES = ["sales_train.csv", "items.csv", "shops.csv", "item_categories.csv"]

# Configuración de rolling windows (ventanas temporales)
DEFAULT_ROLLING_WINDOWS = [3, 6]  # Ventanas de 3 y 6 meses por defecto
MIN_ROLLING_WINDOW = 2  # Mínimo tamaño de ventana
MAX_ROLLING_WINDOW = 12  # Máximo tamaño de ventana


def validate_rolling_windows(window_sizes: List[int]) -> List[int]:
    """Valida que las ventanas rolling sean válidas.

    Parámetros:
        window_sizes: lista de tamaños de ventana (DEBE tener exactamente 2 elementos)

    Retorna:
        Lista validada de tamaños de ventana

    Raises:
        ValueError: si las ventanas no son válidas o no son exactamente 2
    """
    if not window_sizes:
        raise ValueError("Se requiere al menos una ventana rolling")

    # NUEVA RESTRICCIÓN: Exactamente 2 ventanas
    if len(window_sizes) != 2:
        raise ValueError(
            f"rolling_windows debe tener EXACTAMENTE 2 elementos. "
            f"Recibido: {len(window_sizes)} elementos {window_sizes}"
        )

    if not all(isinstance(w, int) for w in window_sizes):
        raise ValueError("Todas las ventanas deben ser enteros")

    if not all(MIN_ROLLING_WINDOW <= w <= MAX_ROLLING_WINDOW for w in window_sizes):
        raise ValueError(
            f"Las ventanas deben estar entre {MIN_ROLLING_WINDOW} y {MAX_ROLLING_WINDOW}. "
            f"Recibido: {window_sizes}"
        )

    if len(window_sizes) != len(set(window_sizes)):
        raise ValueError(f"Las ventanas no pueden repetirse: {window_sizes}")

    # Ordenar ventanas de menor a mayor
    window_sizes_sorted = sorted(window_sizes)

    # Validar orden correcto
    if window_sizes_sorted[0] >= window_sizes_sorted[1]:
        raise ValueError(
            f"La primera ventana ({window_sizes_sorted[0]}) debe ser menor que "
            f"la segunda ({window_sizes_sorted[1]})"
        )

    print(f"✅ Ventanas rolling validadas: {window_sizes_sorted}")
    return window_sizes_sorted


def get_data_path() -> str:
    """
    Obtiene la ruta de los datos con sistema de respaldo:
    1. Intenta usar archivos locales en data/ si existen y son válidos
    2. Si no, descarga desde KaggleHub y los copia a data/
    3. Si KaggleHub falla, usa data/ como último recurso
    """
    # Verificar si data/ tiene todos los archivos
    local_files_exist = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED_FILES)

    if local_files_exist:
        print("✅ Usando archivos locales desde data/")
        return DATA_DIR

    # Intentar descargar desde KaggleHub
    try:
        print("⏳ Descargando dataset desde KaggleHub...")
        kaggle_path = kagglehub.dataset_download(
            "jaklinmalkoc/predict-future-sales-retail-dataset-en"
        )

        # Verificar que los archivos descargados no estén vacíos
        all_valid = True
        for file in REQUIRED_FILES:
            file_path = os.path.join(kaggle_path, file)
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                all_valid = False
                break

        if not all_valid:
            raise ValueError("Archivos descargados están vacíos o incompletos")

        print(f"✅ Dataset descargado en: {kaggle_path}")

        # Copiar archivos a data/ como respaldo
        print("💾 Creando copia de respaldo en data/...")
        os.makedirs(DATA_DIR, exist_ok=True)
        for file in REQUIRED_FILES:
            src = os.path.join(kaggle_path, file)
            dst = os.path.join(DATA_DIR, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"   ✓ {file} respaldado")

        return kaggle_path

    except Exception as e:
        print(f"⚠️  Error al descargar desde KaggleHub: {e}")

        # Verificar si hay respaldo local
        if local_files_exist:
            print("✅ Usando archivos de respaldo desde data/")
            return DATA_DIR
        else:
            raise FileNotFoundError(
                "No se pudo descargar desde KaggleHub y no hay archivos de respaldo en data/. "
                "Por favor, descarga manualmente los archivos y colócalos en la carpeta data/"
            )


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los datasets con sistema de respaldo automático.
    Intenta KaggleHub primero, si falla usa data/ local.
    """
    path = get_data_path()

    # Cargar archivos
    print("📂 Cargando archivos CSV...")
    sales = pd.read_csv(os.path.join(path, "sales_train.csv"))
    items = pd.read_csv(os.path.join(path, "items.csv"))
    shops = pd.read_csv(os.path.join(path, "shops.csv"))
    cats = pd.read_csv(os.path.join(path, "item_categories.csv"))

    # Validar que no estén vacíos
    if any(df.empty for df in [sales, items, shops, cats]):
        raise ValueError("Uno o más datasets están vacíos")

    print(f"✅ Datos cargados exitosamente ({len(sales):,} registros de ventas)")
    return sales, items, shops, cats


def force_download_datasets() -> bool:
    """
    Fuerza la descarga de datasets desde KaggleHub y los guarda en data/.
    Elimina archivos existentes para garantizar datos frescos.

    Returns:
        True si la descarga fue exitosa, False en caso contrario
    """
    try:
        print("🔄 Eliminando archivos antiguos de data/...")
        # Eliminar archivos antiguos
        for file in REQUIRED_FILES:
            file_path = os.path.join(DATA_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"   ✓ {file} eliminado")

        print("⏳ Descargando dataset fresco desde KaggleHub...")
        kaggle_path = kagglehub.dataset_download(
            "jaklinmalkoc/predict-future-sales-retail-dataset-en"
        )

        # Verificar que los archivos descargados sean válidos
        all_valid = True
        for file in REQUIRED_FILES:
            file_path = os.path.join(kaggle_path, file)
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                all_valid = False
                break

        if not all_valid:
            raise ValueError("Archivos descargados están vacíos o incompletos")

        print(f"✅ Dataset descargado en: {kaggle_path}")

        # Copiar archivos a data/
        print("💾 Guardando en data/...")
        os.makedirs(DATA_DIR, exist_ok=True)
        for file in REQUIRED_FILES:
            src = os.path.join(kaggle_path, file)
            dst = os.path.join(DATA_DIR, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"   ✓ {file} guardado")

        print("✅ Datasets regenerados exitosamente en data/")
        return True

    except Exception as e:
        print(f"❌ Error al regenerar datasets: {e}")
        return False


def clean_data(sales: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica y tratamiento de outliers (Clipping)."""
    # Eliminar precios negativos o cero
    sales = sales[sales["item_price"] > 0]

    # Clipping: Limitar ventas extremas (Balanceo de datos) para evitar sesgos
    sales["item_cnt_day"] = sales["item_cnt_day"].clip(0, 20)
    sales["item_price"] = sales["item_price"].clip(0, 300000)

    # Convertir fecha
    if "date" in sales.columns:
        sales["date"] = pd.to_datetime(sales["date"], dayfirst=True)

    return sales


def generate_clusters(
    shops: pd.DataFrame, sales: pd.DataFrame, n_clusters: int = 3
) -> pd.DataFrame:
    """
    APRENDIZAJE NO SUPERVISADO (Clustering).
    Agrupa tiendas según su volumen de ventas para usarlo como feature.
    """
    # Agrupar ventas totales por tienda
    shop_sales = sales.groupby("shop_id")["item_cnt_day"].sum().reset_index()

    # K-Means para agrupar tiendas por volumen de venta
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    shop_sales["shop_cluster"] = kmeans.fit_predict(shop_sales[["item_cnt_day"]])

    return shop_sales[["shop_id", "shop_cluster"]]


def create_demand_bins(target: pd.Series, n_bins: int = 5) -> pd.Series:
    """Discretiza valores continuos de demanda en categorías para facilitar el balanceo.

    Parámetros:
        target: valores de demanda
        n_bins: cantidad de categorías (default 5)
    """
    bins = pd.qcut(target, q=n_bins, labels=False, duplicates="drop")
    return bins


def balance_data_smote(
    X: pd.DataFrame, y: pd.Series, use_balancing: bool = True, sampling_strategy: str = "auto"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Balancea el dataset usando SMOTE sobre bins de demanda.

    Nota: Solo aplicar en train set para evitar data leakage en validación.

    Parámetros:
        X: features de entrada
        y: target continuo
        use_balancing: activar/desactivar SMOTE
        sampling_strategy: estrategia de sobremuestreo
    """
    if not use_balancing or len(y) < 100:
        print("⚠️ Balanceo omitido (dataset pequeño o deshabilitado)")
        return X, y

    # Crear bins temporales para identificar clases minoritarias
    y_bins = create_demand_bins(y, n_bins=5)

    # Verificar distribución antes del balanceo
    print(f"📊 Distribución ANTES del balanceo:")
    print(y_bins.value_counts().sort_index())

    # Aplicar SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=3)

    try:
        X_balanced, y_bins_balanced = smote.fit_resample(X, y_bins)

        # Reconstruir target continuo: usar valores promedio de cada bin
        bin_means = y.groupby(y_bins).mean().to_dict()
        y_balanced = pd.Series(
            [bin_means.get(b, y.mean()) for b in y_bins_balanced], index=X_balanced.index
        )

        print(f"✅ Balanceo SMOTE aplicado: {len(X)} → {len(X_balanced)} muestras")
        print(f"📊 Distribución DESPUÉS del balanceo:")
        print(pd.Series(y_bins_balanced).value_counts().sort_index())

        return X_balanced, y_balanced

    except ValueError as e:
        print(f"⚠️ Error en SMOTE: {e}. Retornando datos originales.")
        return X, y


def create_rolling_window_features(
    df: pd.DataFrame, window_sizes: List[int] = None
) -> pd.DataFrame:
    """Calcula estadísticas de ventana móvil para capturar tendencias temporales.

    Parámetros:
        df: datos temporales ordenados
        window_sizes: tamaños de ventanas (meses). Si es None, usa DEFAULT_ROLLING_WINDOWS

    Raises:
        ValueError: si las ventanas no son válidas
    """
    # Usar ventanas por defecto si no se especifican
    if window_sizes is None:
        window_sizes = DEFAULT_ROLLING_WINDOWS

    # Validar ventanas
    window_sizes = validate_rolling_windows(window_sizes)

    df_rolled = df.copy()

    # Ordenar por fecha para asegurar continuidad temporal
    df_rolled = df_rolled.sort_values(["shop_id", "item_id", "date_block_num"])

    for window in window_sizes:
        # Rolling mean de ventas
        df_rolled[f"rolling_mean_{window}"] = df_rolled.groupby(["shop_id", "item_id"])[
            "item_cnt_day"
        ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

        # Rolling std de ventas
        df_rolled[f"rolling_std_{window}"] = df_rolled.groupby(["shop_id", "item_id"])[
            "item_cnt_day"
        ].transform(lambda x: x.rolling(window=window, min_periods=1).std())

    # Llenar NaN con 0
    df_rolled = df_rolled.fillna(0)

    print(f"✅ Features de rolling window creadas (ventanas: {window_sizes})")
    return df_rolled


def feature_engineering(
    sales: pd.DataFrame,
    items: pd.DataFrame,
    shops_clusters: pd.DataFrame,
    rolling_windows: List[int] = None,
) -> pd.DataFrame:
    """Genera la matriz de entrenamiento con Lags (Variables temporales).

    Parámetros:
        sales: datos de ventas
        items: catálogo de productos
        shops_clusters: clusters de tiendas
        rolling_windows: tamaños de ventanas rolling (None = usar DEFAULT_ROLLING_WINDOWS)
    """
    # Agrupar por mes (date_block_num), tienda e item
    monthly_sales = (
        sales.groupby(["date_block_num", "shop_id", "item_id"])
        .agg({"item_cnt_day": "sum", "item_price": "mean"})  # Precio promedio del mes
        .reset_index()
    )

    # Clip de ventas mensuales (Target range 0-20)
    monthly_sales["item_cnt_day"] = monthly_sales["item_cnt_day"].clip(0, 20)

    # Unir con clusters y categorías
    data = monthly_sales.merge(shops_clusters, on="shop_id", how="left")
    data = data.merge(items[["item_id", "item_category_id"]], on="item_id", how="left")

    # Precio relativo por categoría
    # Calcula el posicionamiento del producto dentro de su segmento de mercado
    category_avg_price = data.groupby(["date_block_num", "item_category_id"])[
        "item_price"
    ].transform("mean")
    data["price_rel_category"] = data["item_price"] / (category_avg_price + 1e-5)

    # Máximo histórico de precio por producto para detectar descuentos
    data = data.sort_values(["item_id", "date_block_num"])
    data["item_price_rolling_max"] = data.groupby("item_id")["item_price"].transform(
        lambda x: x.expanding().max()
    )

    # Ratio de descuento respecto al precio máximo (valores negativos = descuento)
    data["price_discount"] = (data["item_price"] / (data["item_price_rolling_max"] + 1e-5)) - 1

    # Generar Lags (Rezagos: t-1, t-2, t-3)
    data_shifted = data.copy()

    for lag in [1, 2, 3]:
        shifted = data_shifted[["date_block_num", "shop_id", "item_id", "item_cnt_day"]].copy()
        shifted.columns = ["date_block_num", "shop_id", "item_id", f"item_cnt_lag_{lag}"]
        shifted["date_block_num"] += lag
        data = data.merge(shifted, on=["date_block_num", "shop_id", "item_id"], how="left")

    # Lags de precio para capturar elasticidad y cambios temporales
    for lag in [1, 2, 3]:
        shifted_price = data_shifted[["date_block_num", "shop_id", "item_id", "item_price"]].copy()
        shifted_price.columns = ["date_block_num", "shop_id", "item_id", f"item_price_lag_{lag}"]
        shifted_price["date_block_num"] += lag
        data = data.merge(shifted_price, on=["date_block_num", "shop_id", "item_id"], how="left")

    # Llenar NaNs generados por los lags con 0 (meses iniciales)
    data = data.fillna(0)

    # Indicador binario de cambio de precio mensual
    data["is_new_price"] = (data["item_price"] != data["item_price_lag_1"]).astype(int)

    # ========== FEATURES DE MOMENTUM (TENDENCIA DIRECCIONAL) ==========
    print("🔥 Generando features de MOMENTUM...")

    # Delta inmediato: diferencia entre mes anterior y trasanterior
    data["delta_1_2"] = data["item_cnt_lag_1"] - data["item_cnt_lag_2"]

    # Aceleración/evolución en 3 meses (captura curvatura de la tendencia)
    data["evolution_3m"] = data["item_cnt_lag_1"] - data["item_cnt_lag_3"]

    # Momentum promedio (combina señales de corto y mediano plazo)
    data["momentum_avg"] = (data["delta_1_2"] + data["evolution_3m"]) / 2.0

    # Dirección de tendencia (1=subiendo, -1=bajando, 0=estable)
    data["trend_direction"] = np.sign(data["delta_1_2"])

    # ========== FEATURES DE SENSIBILIDAD AL PRECIO ==========
    print("💰 Generando features de SENSIBILIDAD AL PRECIO...")

    # Cambio porcentual de precio respecto al mes anterior
    data["price_change_pct"] = (data["item_price"] - data["item_price_lag_1"]) / (
        data["item_price_lag_1"] + 1e-6
    )

    # Cambio porcentual de precio en 2 meses (tendencia más amplia)
    data["price_change_2m_pct"] = (data["item_price"] - data["item_price_lag_2"]) / (
        data["item_price_lag_2"] + 1e-6
    )

    # Ingreso potencial (interacción ventas × precio)
    data["revenue_potential"] = data["item_cnt_lag_1"] * data["item_price"]

    # Elasticidad precio-demanda aproximada
    data["price_demand_elasticity"] = np.where(
        data["price_change_pct"] != 0, data["delta_1_2"] / (data["price_change_pct"] + 1e-6), 0
    )

    # Agregar rolling window features para capturar tendencias
    data = create_rolling_window_features(data, window_sizes=rolling_windows)

    # ========== FEATURES DE DESVIACIONES RESPECTO A PROMEDIOS ==========
    print("📊 Generando features de DESVIACIÓN respecto a promedios...")

    # Usar ventanas rolling validadas
    if rolling_windows is None:
        rolling_windows = DEFAULT_ROLLING_WINDOWS
    else:
        rolling_windows = validate_rolling_windows(rolling_windows)

    for window in rolling_windows:
        mean_col = f"rolling_mean_{window}"
        std_col = f"rolling_std_{window}"

        # Desviación absoluta del último mes vs promedio
        data[f"diff_to_mean_{window}"] = data["item_cnt_lag_1"] - data[mean_col]

        # Z-score: cuántas desviaciones estándar de distancia
        data[f"zscore_{window}"] = np.where(
            data[std_col] > 0, (data["item_cnt_lag_1"] - data[mean_col]) / (data[std_col] + 1e-6), 0
        )

        # Coeficiente de variación (volatilidad relativa)
        data[f"volatility_coef_{window}"] = np.where(
            data[mean_col] > 0, data[std_col] / (data[mean_col] + 1e-6), 0
        )

    # Limpieza final: reemplazar infinitos y NaNs
    data = data.replace([np.inf, -np.inf], 0)
    data = data.fillna(0)

    # Normalización de variables de precio y ventas para comparabilidad
    print("📐 Aplicando normalización a variables de precio y ventas...")

    # Transformación logarítmica para variables con distribución sesgada
    # Esto reduce el impacto de outliers y estabiliza la varianza
    price_features = ["item_price", "price_rel_category", "revenue_potential"]
    for feat in price_features:
        if feat in data.columns:
            data[f"{feat}_log"] = np.log1p(data[feat])

    # Normalización de lags de ventas para mantener escala consistente
    lag_features = ["item_cnt_lag_1", "item_cnt_lag_2", "item_cnt_lag_3"]
    for feat in lag_features:
        if feat in data.columns:
            data[f"{feat}_log"] = np.log1p(data[feat])

    print(f"✅ Feature Engineering completado: {data.shape[1]} columnas, {data.shape[0]} filas")

    return data


def prepare_full_pipeline(
    use_balancing: bool = False, balance_strategy: str = "auto", rolling_windows: List[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TimeSeriesSplit]:
    """Pipeline completo de procesamiento: limpieza, features, splits temporales.

    Retorna splits train/val/test + TimeSeriesSplit para validación cruzada.

    Parámetros:
        use_balancing: activar SMOTE en train
        balance_strategy: estrategia de sobremuestreo
        rolling_windows: tamaños de ventanas rolling (None = usar DEFAULT_ROLLING_WINDOWS)
    """
    # Validar ventanas rolling al inicio
    if rolling_windows is None:
        rolling_windows = DEFAULT_ROLLING_WINDOWS
    rolling_windows = validate_rolling_windows(rolling_windows)
    sales, items, shops, cats = load_data()

    print("🧹 Limpiando datos...")
    sales = clean_data(sales)

    print("🤖 Generando Clusters (K-Means)...")
    shops_clusters = generate_clusters(shops, sales)

    print(f"⚙️ Ingeniería de Características (Lags + Rolling Windows {rolling_windows})...")
    df_final = feature_engineering(sales, items, shops_clusters, rolling_windows=rolling_windows)

    # Transformación logarítmica para estabilizar la varianza del target
    df_final["target_log"] = np.log1p(df_final["item_cnt_day"])

    # Configurar splits respetando el orden temporal de los datos
    print("📅 Configurando Time Series Split (ventana temporal)...")

    # Ordenar por fecha para respetar cronología
    df_final = df_final.sort_values("date_block_num")

    # Definir splits temporales (últimos 2 meses para val y test)
    max_month = df_final["date_block_num"].max()

    train = df_final[df_final["date_block_num"] < max_month - 1]
    val = df_final[df_final["date_block_num"] == max_month - 1]
    test = df_final[df_final["date_block_num"] == max_month]

    # Crear generador TimeSeriesSplit para validación cruzada (opcional)
    tscv = TimeSeriesSplit(n_splits=5)

    # Aplicar balanceo solo en train para evitar contaminar val/test
    if use_balancing and len(train) > 100:
        print("⚖️ Aplicando SMOTE en conjunto de entrenamiento...")
        features = [
            col
            for col in train.columns
            if col not in ["target_log", "item_cnt_day", "date_block_num"]
        ]

        X_train = train[features]
        y_train = train["target_log"]

        X_train_balanced, y_train_balanced = balance_data_smote(
            X_train, y_train, use_balancing=True, sampling_strategy=balance_strategy
        )

        # Reconstruir train balanceado
        train = X_train_balanced.copy()
        train["target_log"] = y_train_balanced
        train["item_cnt_day"] = np.expm1(y_train_balanced)  # Invertir log para consistencia

    print(f"📊 Dataset listo: Train ({len(train)}), Val ({len(val)}), Test ({len(test)})")
    print(f"🔄 TimeSeriesSplit configurado con {tscv.n_splits} splits para validación cruzada")

    return train, val, test, tscv
