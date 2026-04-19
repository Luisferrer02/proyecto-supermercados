# Drift del modelo: estrategia de detección y mitigación

**Hallazgo auditoría**: O-04 ("Drift del modelo ignorado").

## 1. Qué es el drift y por qué nos afecta

**Concept drift** es el fenómeno por el que la relación entre features e target cambia con el tiempo. En nuestro caso puede manifestarse en tres formas:

| Tipo | Ejemplo en nuestro proyecto |
|------|-----------------------------|
| **Covariate drift** | Los precios suben con la inflación; la distribución de `price_numeric` se desplaza. |
| **Label drift** | Las elasticidades cambian: un helado en julio sube menos si hay verano frío. |
| **Concept drift puro** | La relación "eye-level → más ventas" se debilita si los clientes compran cada vez más online. |

Si el modelo se entrenó con 2025 y se despliega para predecir 2026, estos shifts **degradan silenciosamente** la calidad de las predicciones. Sin monitorización, nadie se entera.

## 2. Por qué el hallazgo "persiste" como tal

La implementación actual entrena una vez y se asume válido indefinidamente. No hay:
- Detección automática de distribución anómala.
- Re-entrenamiento periódico.
- Alerta cuando el MSE sobre datos nuevos diverge del MSE de train.

Hasta que no haya un despliegue continuo con clientes reales, el drift es **hipotético**. Pero la estrategia debe estar definida para defensa.

## 3. Estrategia en tres capas

### Capa 1 — Detección de covariate drift (barata, se puede implementar ya)

**Qué**: comparar la distribución de features del mes nuevo contra la distribución del training set, categoría por categoría.

**Métrica**: **Population Stability Index (PSI)**:

```
PSI = Σᵢ (p_new,i − p_train,i) · ln(p_new,i / p_train,i)
```

Sobre cada feature continua (`price_numeric`, `estimated_monthly_sales`, `profit_margin_percentage`, `product_width_cm`):
- PSI < 0.1 → distribuciones iguales, OK
- 0.1 ≤ PSI < 0.25 → drift moderado, alertar
- PSI ≥ 0.25 → drift severo, reentrenar

**Implementación prevista**: script `06_drift_check.py` que lee el CSV mensual más reciente, lo compara contra el training set original (referencia) y escribe un `results/drift_report.json`.

### Capa 2 — Detección de performance drift (requiere ground truth)

**Qué**: si se tiene un CSV real del mes predicho (a posteriori), comparar el lift predicho vs. el lift observado.

**Métrica**: error relativo del profit lift:

```
err_lift = |lift_predicho − lift_observado| / lift_observado
```

Umbrales:
- < 5 % → OK
- 5–15 % → aceptable, monitorizar
- \> 15 % → re-entrenar

**Limitación honesta**: en un despliegue real necesitaríamos esperar un mes para tener ground truth. En nuestro entorno sintético esto es inmediato.

### Capa 3 — Re-entrenamiento periódico (mitigación)

**Política sugerida**:

- **Trigger automático**: reentrenar si PSI ≥ 0.25 en ≥ 2 features o si `err_lift` > 15 %.
- **Trigger por calendario**: reentrenar cada 3 meses independientemente, con los CSVs acumulados hasta la fecha.
- **Rolling window**: usar siempre los últimos 12 meses para entrenar (evita que el modelo arrastre patrones obsoletos).

**Implementación prevista**: `04_ingest.py` acepta ya el directorio completo; basta con programar una tarea que lo invoque mensualmente si se detecta drift.

## 4. Por qué no lo resolvemos completamente ahora

1. **Datos sintéticos**: no hay drift "real" que medir. Cualquier implementación funcionará sobre los CSVs que nosotros generamos, con drift inducido artificialmente.
2. **Fuera del objetivo core**: el scope del proyecto es la **optimización**, no el MLOps de producción.
3. **No hay acceso continuo a datos**: el dataset base (`products_macro.csv`) es una foto fija.

**Qué sí aportamos ya**: el diseño arquitectónico está preparado para drift handling. El pipeline es idempotente (volver a correr `04_ingest.py` con nuevos datos reemplaza limpiamente el estado anterior si se versiona).

## 5. Resumen ejecutivo para defensa

> **¿Cómo gestionan el drift del modelo?**
>
> No lo hemos implementado porque el proyecto se ejecuta sobre datos sintéticos estáticos (12 meses fijos). La estrategia diseñada tiene tres capas: (1) detección de covariate drift por PSI sobre cada feature, comparando cada CSV nuevo contra el training set de referencia; (2) detección de performance drift comparando el lift predicho con el observado cuando hay ground truth disponible; (3) re-entrenamiento automático disparado por umbrales (PSI ≥ 0.25 o error de lift > 15 %) o por calendario trimestral. La arquitectura del pipeline (ingestión idempotente) ya soporta esto.

## 6. Limitaciones que reconocemos

- El sistema actual no detecta drift automáticamente; es una decisión consciente, no un olvido.
- Un deploy real requeriría añadir monitorización continua (prometheus + grafana + alertas), fuera del alcance académico.
- Sin un flujo de ground truth real, la capa 2 solo funciona post-hoc.
