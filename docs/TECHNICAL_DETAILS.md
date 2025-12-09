# Detalles Técnicos del Sistema

## 1. Metodología

El proyecto se rige por la metodología **CRISP-DM**, abarcando desde la comprensión del negocio y datos hasta el despliegue del prototipo funcional.

## 2. Arquitectura del Modelo (Stacking)

Implementamos una estrategia de **Ensemble Learning Heterogéneo** para reducir la varianza y el sesgo:

* **Nivel Base (Weak Learners):**
  * *Random Forest:* Captura no-linealidades robustas mediante agregación de árboles (n_estimators=50, max_depth=10)
  * *XGBoost:* Optimiza el error residual mediante Gradient Boosting (n_estimators=100, learning_rate=0.1)
* **Meta-Modelo (Nivel 1):**
  * *Regresión Lineal:* Pondera las predicciones base para generar la estimación final

## 3. Aprendizaje No Supervisado

**Clustering Particional (K-Means):**

* Segmentación automática de tiendas según volumen de venta histórico
* k=2 clusters determinados por el método del codo
* Feature adicional: `shop_cluster` (0: Bajo Volumen, 1: Volumen Medio, 2: Alto Volumen)
* Implementado en `src/data_processing.py::generate_clusters()`

## 4. Ingeniería de Características Avanzada

El sistema genera **24+ features engineered** para capturar patrones complejos:

### 4.1. Variables Base
* **Variables Temporales (Lags):** Rezagos (t-1, t-2, t-3) para capturar inercia de demanda
* **Lags de Precio:** Precios históricos (t-1, t-2) para calcular cambios porcentuales
* **Balanceo de Target:** Transformación `log1p` para normalizar distribución
* **Clipping de Outliers:** Ventas (0-20), precios (0-300,000)
* **Agregación Temporal:** Ventas mensuales con precio promedio

### 4.2. Features de Momentum (Tendencia Direccional)

**Problema resuelto:** El modelo detecta si la demanda está acelerando o desacelerando, en lugar de solo regresar al promedio histórico.

```python
# Delta inmediato (velocidad del cambio)
delta_1_2 = item_cnt_lag_1 - item_cnt_lag_2

# Evolución en 3 meses (aceleración/curvatura)
evolution_3m = item_cnt_lag_1 - item_cnt_lag_3

# Momentum promedio
momentum_avg = (delta_1_2 + evolution_3m) / 2.0

# Dirección de tendencia (1=subiendo, -1=bajando, 0=estable)
trend_direction = sign(delta_1_2)
```

**Impacto:** Si las ventas cayeron 2 meses consecutivos, el modelo predice continuar la caída (momentum negativo) en lugar de regresar al promedio histórico.

### 4.3. Features de Sensibilidad al Precio

**Problema resuelto:** El modelo aprende la relación inversa entre precio y demanda (elasticidad).

```python
# Cambio porcentual de precio
price_change_pct = (precio_actual - precio_lag_1) / precio_lag_1

# Cambio en 2 meses (tendencia más amplia)
price_change_2m_pct = (precio_actual - precio_lag_2) / precio_lag_2

# Ingreso potencial (interacción ventas × precio)
revenue_potential = item_cnt_lag_1 * item_price

# Elasticidad precio-demanda aproximada
price_demand_elasticity = delta_ventas / delta_precio
```

**Impacto:** 
- Si precio subió 10% y ventas cayeron 20% → **alta elasticidad** (producto sensible al precio)
- Si precio subió pero ventas también subieron → **producto premium/inelástico**

### 4.4. Features de Desviaciones (Detección de Anomalías)

**Problema resuelto:** El modelo distingue entre comportamiento normal y picos/caídas anómalas.

```python
# Desviación absoluta vs promedio
diff_to_mean_3 = item_cnt_lag_1 - rolling_mean_3

# Z-score (cuántas desviaciones estándar de distancia)
zscore_3 = (item_cnt_lag_1 - rolling_mean_3) / rolling_std_3

# Coeficiente de variación (volatilidad relativa)
volatility_coef_3 = rolling_std_3 / rolling_mean_3
```

**Impacto:**
- **Picos anómalos** (zscore > 2): Un mes excepcionalmente alto que no se repetirá
- **Caídas drásticas** (zscore < -2): Un mes mucho peor que el promedio
- **Productos estables** (volatility_coef bajo): Demanda predecible
- **Productos volátiles** (volatility_coef alto): Alta variabilidad

### 4.5. Resumen de Features por Categoría

| **Categoría**       | **Features**                                                         | **Propósito**                |
| ------------------- | -------------------------------------------------------------------- | ---------------------------- |
| **Base**            | shop_cluster, item_category_id, item_price                           | Contexto del producto        |
| **Lags**            | item_cnt_lag_1, lag_2, lag_3                                         | Inercia de demanda           |
| **Momentum**        | delta_1_2, evolution_3m, momentum_avg, trend_direction               | Tendencia direccional        |
| **Precio**          | price_change_pct, price_change_2m_pct, revenue_potential, elasticity | Sensibilidad al precio       |
| **Rolling Windows** | rolling_mean_*, rolling_std_* (×2 ventanas)                          | Promedios históricos         |
| **Desviaciones**    | diff_to_mean_*, zscore_*, volatility_coef_* (×2 ventanas)            | Detección de anomalías       |
| **TOTAL**           | **24+ features**                                                     | Captura completa de patrones |

### 4.6. Ventajas del Nuevo Sistema de Features

✅ **Detecta tendencias recientes** (momentum) en lugar de solo promedios históricos  
✅ **Captura elasticidad precio-demanda** para productos sensibles al precio  
✅ **Identifica picos anómalos** que no deben propagarse a futuras predicciones  
✅ **Distingue productos estables vs volátiles** mediante coeficientes de variación  
✅ **Previene regresión a la media** cuando hay momentum sostenido

### Ventanas Temporales (Rolling Windows) - Configuración Fija de 2 Ventanas

```python
# Configuración por defecto (SIEMPRE 2 ventanas)
DEFAULT_ROLLING_WINDOWS = [3, 6]  # Exactamente 2 ventanas

# Features generadas automáticamente
rolling_mean_{window}   # Media móvil de N meses
rolling_std_{window}    # Desviación estándar de N meses
```

**RESTRICCIÓN IMPORTANTE: EXACTAMENTE 2 VENTANAS**

El sistema requiere **obligatoriamente 2 ventanas rolling** para mantener consistencia dimensional:
- ✅ Válido: `[3, 6]`, `[2, 4]`, `[4, 8]`, `[3, 9]`
- ❌ Inválido: `[3]` (1 ventana), `[3, 6, 9]` (3 ventanas), `[2, 4, 6, 8]` (4 ventanas)

**Validaciones automáticas:**
- **Longitud exacta:** Debe ser una lista con exactamente 2 elementos
- **Rango permitido:** Entre 2 y 12 meses (inclusive)
- **Sin duplicados:** `[3, 3]` no es válido
- **Orden ascendente:** Primera ventana < Segunda ventana (ej: `[3, 6]` ✅, `[6, 3]` ❌)
- **Enteros positivos:** Todos los valores deben ser enteros positivos

**Casos de uso recomendados:**
- `[2, 4]`: Productos con alta variabilidad (ropa de moda, tecnología)
- `[3, 6]`: Balance general (⭐ **RECOMENDADO** para la mayoría de casos)
- `[4, 8]`: Productos con tendencias medias
- `[6, 12]`: Productos estacionales (decoración navideña, útiles escolares)

**¿Por qué exactamente 2 ventanas?**
- **Consistencia dimensional:** Todos los modelos entrenados usan 10 features fijas (6 base + 4 rolling)
- **Performance SHAP:** La explicabilidad requiere dimensiones consistentes
- **Eficiencia computacional:** Balance entre captura de patrones y tiempo de cálculo
- **Simplicidad de API:** Schema dinámico predecible y fácil de documentar

### Balanceo de Datos con SMOTE

- Discretización de demanda en 5 bins
- SMOTE aplicado sobre bins
- Reconstrucción de valores continuos

### Validación Temporal (TimeSeriesSplit)

- 5 splits con expansión progresiva
- Previene data leakage temporal
- Respeta cronología de datos

## 5. Explicabilidad (XAI)

El sistema integra **SHAP (SHapley Additive exPlanations)** en el frontend, proporcionando transparencia algorítmica al desglosar el impacto marginal de cada variable en la predicción final.

* **Modelo Proxy:** XGBoost simplificado para compatibilidad con TreeExplainer
* **Visualización Dinámica:** Waterfall charts con soporte para temas dark/light
* **Interpretabilidad:** Muestra cómo cada feature contribuye a la predicción

## 6. Arquitectura de Software

El frontend sigue los principios **SOLID** con una arquitectura modular:

* **18 archivos Python** con una clase por archivo
* **Separación de responsabilidades:** Services (lógica de negocio), Components (visualización), Views (vistas), UI Components (interfaz)
* **Patrones de diseño:** Singleton (SessionStateManager), Builder (ChartBuilder), Service Layer, Dependency Injection
* Para más detalles, ver [Documentación de Arquitectura](../app/README.md)

## 7. Sistema de Respaldo de Datos

Implementación robusta de gestión de datasets con múltiples capas de seguridad:

### Prioridad de carga:
* ✅ Si `data/` tiene todos los archivos → los usa directamente (más rápido)
* ⏳ Si no → descarga desde KaggleHub
* 💾 Copia automáticamente a `data/` como respaldo
* ⚠️ Si KaggleHub falla → usa `data/` como último recurso

### Validaciones automáticas:
* Verifica existencia de archivos requeridos
* Valida que no estén vacíos (tamaño > 0)
* Comprueba que los DataFrames cargados contengan datos

### Archivos gestionados:
* `sales_train.csv` - Registros históricos de ventas
* `items.csv` - Catálogo de productos
* `shops.csv` - Información de tiendas
* `item_categories.csv` - Categorías de productos
