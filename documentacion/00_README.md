# Documentación técnica — Respuestas de defensa

Esta carpeta contiene las respuestas justificadas a los hallazgos de la auditoría final que son de naturaleza **teórica** (no requieren cambio de código, pero sí respuesta preparada ante el tribunal).

Las respuestas están basadas en la implementación real del pipeline (`mlops/`) con referencias a archivos y líneas concretas, no en teoría genérica.

## Índice

| # | Documento | Hallazgos que responde |
|---|-----------|------------------------|
| 01 | [Justificación de hiperparámetros del Transformer](01_justificacion_hiperparametros_transformer.md) | M-04, A3-M01 |
| 02 | [Análisis del underperformance de PPO](02_analisis_ppo_underperformance.md) | A2-M02 |
| 03 | [MSE con unidades y baselines de referencia](03_metricas_mse_unidades_baseline.md) | A2-M03 |
| 04 | [Test set independiente y validación](04_test_set_independiente.md) | O-02 |
| 05 | [Error del pipeline RAG cuantificado](05_error_rag_cuantificado.md) | A2-D01 |
| 06 | [Drift del modelo: estrategia](06_drift_modelo.md) | O-04 |
| 07 | [Perfiles de cliente: por qué quedan fuera de scope](07_perfiles_cliente.md) | F-04 |

## Mapa a la auditoría

- **Hallazgos críticos A3** (Docker, concurrencia): fuera de alcance por decisión explícita del equipo de seguir operando en **entorno local controlado** para la defensa. La demo se ejecutará en la máquina del equipo, evitando los fallos de infraestructura multi-instancia.
- **Hallazgos críticos A3 resueltos vía código**: fallback LLM, persistencia de modelos, validación de CSV, unificación de métricas, explicabilidad, UX para manager → ver cambios en `mlops/` y `web/`.
- **Hallazgos persistentes A1/A2 teóricos**: en esta carpeta.
