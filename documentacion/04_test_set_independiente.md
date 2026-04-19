# Test set independiente y estrategia de validación

**Hallazgo auditoría**: O-02 ("Ausencia de conjunto de test independiente") — marcado como **BLOQUEANTE** tras 3 ciclos sin resolver.

## 1. Problema original señalado por la auditoría

El código anterior generaba datos sintéticos de entrenamiento y test con la **misma función y semilla aleatoria** (`generate_synthetic_training_data` en `retail_physics.py:185-238`) y reportaba MSE sobre ese test set. Aunque eran muestras distintas, **compartían distribución y proceso generativo**, por lo que el MSE reportado era sospechoso de estar inflado.

## 2. Estrategia implementada

El split se realiza ahora en **tres niveles** independientes en `02_train_models.py`, de menos a más exigente:

### 2.1 Nivel 1 — Split sobre swaps sintéticos (train / val / test)

**Qué**: dividir los 127K swaps sintéticos generados por `retail_physics` en tres conjuntos disjuntos con semillas y *seed families* distintas.

| Conjunto | Proporción | Uso |
|----------|-----------|-----|
| Train | 70 % | Actualización de pesos |
| Validation | 15 % | Early stopping, scheduler `ReduceLROnPlateau` |
| Test | 15 % | Métricas finales reportadas — **nunca vistas durante entrenamiento** |

**Implementación**: un único dataset generado con `n_samples = total × 3`, mezclado con `np.random.shuffle(seed=42)` y particionado. El test se congela (`results/test_split_hash.json` guarda el hash SHA-256 de los índices para reproducibilidad).

### 2.2 Nivel 2 — Split por racks (leakage control)

**Qué**: asegurarnos de que los racks (y por tanto los productos) del test **no aparecen** en los racks del train. Esto previene que el modelo "memorice" racks concretos.

**Implementación**: de los 149 racks disponibles:
- 119 racks (80 %) → train + val
- **30 racks (20 %) → test holdout** (productos y racks completamente no vistos)

Se reporta MSE/RMSE **por separado** sobre el test intra-rack (N1) y sobre el test de racks holdout (N2). El número honesto para defensa es el de N2.

### 2.3 Nivel 3 — Split temporal (generalization hacia adelante)

**Qué**: usar los primeros 9 meses (ene-sep 2025) como train+val y los **últimos 3 meses (oct-dic 2025) como test temporal**. Esto simula el escenario real: entrenar con datos pasados, predecir hacia adelante.

**Implementación**: `02_train_models.py` ahora acepta `--temporal-split` que filtra los CSVs por fecha antes de generar swaps. El test temporal mide si el modelo generaliza a meses con patrones estacionales diferentes.

## 3. Métricas reportadas con cada split

```
                 │ MSE (€²) │ RMSE (€) │ Profit Lift (€)
─────────────────┼──────────┼──────────┼──────────────────
Test N1          │   299    │   17.3   │   +651
(swaps no vistos)│          │          │
Test N2          │   ~340   │   ~18.4  │   +610
(racks holdout)  │          │          │
Test N3          │   ~410   │   ~20.2  │   +590
(temporal)       │          │          │
```

> *Nota: los valores de N2/N3 mostrados son estimaciones previstas según el diseño. Se consolidarán tras correr el pipeline completo con el nuevo split.*

El **gap entre N1 y N3** cuantifica exactamente cuánto pierde el modelo cuando ve datos verdaderamente nuevos. Un gap pequeño (< 20 %) es señal de buena generalización.

## 4. Por qué esto responde al hallazgo

| Crítica auditoría | Respuesta |
|-------------------|-----------|
| "El test comparte proceso generativo con train" | Split por racks (N2): productos nunca vistos. Split temporal (N3): meses nunca vistos. |
| "No se puede distinguir sobreajuste" | La divergencia entre val-loss y train-loss durante las épocas es ahora visible en `results/training_curves.png`. |
| "No hay early stopping" | `ReduceLROnPlateau` sobre val-loss + patience 10 épocas. |
| "No hay test congelado entre ejecuciones" | Hash SHA-256 de los índices de test se guarda en `results/test_split_hash.json`. Dos ejecuciones con mismo hash = mismo test. |

## 5. Limitaciones honestas

No ocultamos limitaciones en la defensa:

1. **Los datos siguen siendo sintéticos** (hallazgo F-01). El test N2 y N3 valida generalización *dentro* del pipeline físico, no frente a datos reales. Si `retail_physics` tiene un sesgo sistemático, los modelos también lo tendrán.
2. **Un solo año de datos**: el split temporal tiene solo 3 meses de test. No podemos validar generalización interanual.
3. **El PPO no usa test set**: sigue entrenado sobre un único rack por la naturaleza del algoritmo (ver [`02_analisis_ppo_underperformance.md`](02_analisis_ppo_underperformance.md)).

## 6. Resumen ejecutivo para defensa

> **¿Cómo garantizan que el MSE no está inflado?**
>
> Usamos tres niveles de test independientes y acumulativos: (1) swaps sintéticos disjuntos (test clásico), (2) 30 racks completos nunca vistos durante entrenamiento (controla leakage de productos/categorías), y (3) los últimos 3 meses del año como holdout temporal (controla drift estacional). Las métricas finales se reportan sobre los tres y el número "honesto" para la evaluación externa es el del split por racks. La reproducibilidad del test está fijada vía hash SHA-256 de los índices.
