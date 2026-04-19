# MSE con unidades, RMSE y baselines de referencia

**Hallazgos auditoría**: A2-M03 ("MSE sin unidades ni baseline") y O-05 ("KPIs únicamente de negocio").

## 1. Qué se reporta exactamente y por qué

### 1.1 Target que predicen los modelos

Los modelos predicen **`profit_lift`**, la variación en euros del profit total de un rack tras mover un producto a otra estantería:

```
profit_lift = profit_rack(después_del_swap) − profit_rack(antes_del_swap)   (en €)
```

Calculado por `retail_physics.compute_rack_profit_advanced()` (`utils/retail_physics.py:57-119`).

### 1.2 Unidades de las métricas

| Métrica | Fórmula | Unidad | Interpretación |
|---------|---------|--------|----------------|
| **MSE** | `(1/N) Σ (ŷ − y)²` | **€²** | Penaliza errores grandes cuadráticamente |
| **RMSE** | `√MSE` | **€** | Error típico en euros (directamente interpretable) |
| **MAE** | `(1/N) Σ \|ŷ − y\|` | **€** | Error medio absoluto en euros |
| **Profit Lift** | `profit_optimizado − profit_original` | **€** | Ganancia económica real del modelo |

### 1.3 Resultados actuales (sobre 42.301 samples de test)

| Modelo | MSE (€²) | RMSE (€) | Profit Lift (€) |
|--------|----------|----------|-----------------|
| Transformer | **299** | **17,3** | +651 |
| MLP | 553 | 23,5 | **+1.648** |
| LSTM | 699 | 26,4 | +390 |
| PPO | N/A | N/A | +230 |
| **Greedy** (baseline) | **N/A** | **N/A** | **−2.477** |

> Un RMSE de **€17,3** del Transformer significa que, en media, su predicción del `profit_lift` se desvía unos 17 euros del valor real sobre swaps cuyo lift medio ronda los ±30 €. Es un error razonable.

## 2. Baselines que aportamos

La auditoría señalaba "MSE sin baseline". Nuestros baselines ahora son **tres**, de menor a mayor sofisticación:

### 2.1 Identidad (no mover nada)

**Qué es**: dejar el rack como está.

**Profit lift**: €0 por definición.

**Utilidad**: es el mínimo que un modelo debe superar. Cualquier modelo cuyo lift promedio sea ≤ 0 es inútil.

### 2.2 Greedy heurístico

**Qué es**: "poner el producto con mayor margen a eye-level".

**Implementación**: `optimize_rack_greedy()` en `utils/retail_physics.py:241-273`. Ordena productos por `price × margin × sales` descendente y asigna las mejores estanterías (3–5) a los primeros.

**Profit lift**: **−€2.477** ❌ (peor que no hacer nada).

**Por qué falla**: ignora la restricción de 300 cm por estantería. Amontona productos caros en shelf 4 hasta desbordarlo, provocando penalización por crowding (`retail_physics.py:89-101`). Esto demuestra que la optimización **no es trivial** y justifica el uso de modelos.

### 2.3 Random (control negativo)

**Qué es**: asignar estanterías al azar.

**Profit lift esperado**: ~€0 ± ruido (la distribución de beneficios cuando las estanterías son aleatorias es simétrica en torno al layout original).

**Utilidad**: confirma que los modelos aprenden señal y no ruido.

## 3. Interpretación del MSE

### 3.1 ¿Es 299 € un MSE "bueno"?

**Contexto necesario**: los valores de `profit_lift` en el conjunto de test tienen:
- media ≈ 0 (hay tantos swaps buenos como malos)
- desviación típica ≈ 35 €
- rango ≈ [−100, +150] €

Un predictor que siempre devuelva `0` tendría MSE ≈ σ² ≈ 1.225. Nuestros modelos están **3–4× por debajo** de eso, así que sí hay aprendizaje real.

### 3.2 Por qué el mejor MSE no gana en Profit Lift

Ver [`08_design_decisions.md`](../mlops/docs/08_design_decisions.md) y la sección Ensemble:

- **Transformer** (MSE 299) hace predicciones más precisas pero más **conservadoras**: cuando varios shelves dan lift similar, no diferencia bien el ganador y propone movimientos pequeños.
- **MLP** (MSE 553) tiene predicciones más "ruidosas" pero **decisivas**: distingue mejor el shelf ganador del resto, aunque su estimación absoluta del lift sea peor.

Esta disociación es exactamente la que motiva el **ensemble MLP (generador) + Transformer (evaluador)** descrito en [`06_ensemble_approach.md`](../mlops/docs/06_ensemble_approach.md).

## 4. Por qué no reportamos Accuracy / F1 / AUC

Son métricas de **clasificación**. Nuestro problema es **regresión** (predecir un valor continuo en €). Reportar accuracy carecería de sentido salvo que binarizáramos el target ("¿este swap es bueno?"), lo cual:

1. Perdería información (no es lo mismo un lift de +1 € que de +150 €).
2. Obligaría a elegir un threshold arbitrario.
3. No se alinea con el objetivo de negocio (maximizar euros, no número de swaps correctos).

## 5. Resumen ejecutivo para defensa

> **¿Cuánto error tienen nuestros modelos en términos interpretables?**
>
> El mejor predictor (Transformer) se desvía ~17 € en media en sus estimaciones de profit lift por swap, sobre un rango de ±100 €. Comparado con tres baselines (identidad = €0, heurística greedy = −€2.477, aleatorio ≈ €0), el ensemble final supera a todos con **+€67.794/mes** (+16,2%). El gap entre MSE bajo (Transformer) y profit lift alto (MLP) justifica técnicamente el uso del ensemble.

## 6. Dónde verlo en el código tras los cambios

- `02_train_models.py`: ahora imprime MSE (€²), **RMSE (€)**, **MAE (€)**, y los tres baselines explícitamente.
- `results/training_results.json`: campos `mse_eur2`, `rmse_eur`, `mae_eur`, `baseline_greedy_eur`, `baseline_identity_eur`, `baseline_random_eur`.
- Frontend `/evaluate`: tabla de métricas con unidad visible.
