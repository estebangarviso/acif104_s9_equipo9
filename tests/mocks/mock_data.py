"""
Generadores de datos sintéticos para tests.
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def create_mock_dataframe(
    n_rows: int = 100, date_blocks: int = 10, n_shops: int = 5, n_items: int = 20
) -> pd.DataFrame:
    """Crea un DataFrame mock de ventas."""
    np.random.seed(42)

    return pd.DataFrame(
        {
            "date_block_num": np.random.randint(0, date_blocks, n_rows),
            "shop_id": np.random.randint(0, n_shops, n_rows),
            "item_id": np.random.randint(0, n_items, n_rows),
            "item_price": np.random.uniform(100, 5000, n_rows),
            "item_cnt_day": np.random.randint(0, 20, n_rows),
        }
    )


def create_mock_sales_data(n_months: int = 12) -> pd.DataFrame:
    """Crea datos de ventas mensuales sintéticos."""
    np.random.seed(42)

    data = []
    for month in range(n_months):
        for shop in [1, 2]:
            for item in [100, 101]:
                data.append(
                    {
                        "date_block_num": month,
                        "shop_id": shop,
                        "item_id": item,
                        "item_category_id": 40,
                        "item_price": 1500 + np.random.normal(0, 100),
                        "item_cnt_day": max(0, 10 + np.random.normal(0, 3)),
                    }
                )

    return pd.DataFrame(data)


def create_mock_feature_engineered_data(n_rows: int = 50) -> pd.DataFrame:
    """Crea datos con features de ingeniería."""
    np.random.seed(42)

    return pd.DataFrame(
        {
            "shop_cluster": np.random.randint(0, 3, n_rows),
            "item_category_id": np.random.randint(40, 50, n_rows),
            "item_price": np.random.uniform(500, 3000, n_rows),
            "item_cnt_lag_1": np.random.randint(0, 20, n_rows),
            "item_cnt_lag_2": np.random.randint(0, 20, n_rows),
            "item_cnt_lag_3": np.random.randint(0, 20, n_rows),
            "price_rel_category": np.random.uniform(0.5, 1.5, n_rows),
            "price_discount": np.random.uniform(-0.3, 0.1, n_rows),
            "is_new_price": np.random.randint(0, 2, n_rows),
            "rolling_mean_3": np.random.uniform(5, 15, n_rows),
            "rolling_std_3": np.random.uniform(1, 5, n_rows),
            "rolling_mean_6": np.random.uniform(5, 15, n_rows),
            "rolling_std_6": np.random.uniform(1, 5, n_rows),
        }
    )
