# ShelfOpt — Optimización inteligente de estanterías de supermercado

Un sistema inteligente que **determina automáticamente la disposición más rentable de productos en las estanterías de un supermercado** mediante modelos de aprendizaje automático. Se han experimentado con 4 arquitecturas distintas (MLP, LSTM, Transformer y PPO) para encontrar el enfoque más efectivo.

## Estado actual del proyecto

Resultados obtenidos con el dataset sintético generado por nuestro propio pipeline:
| Métrica | Valor |
|---------|-------|
| Productos optimizados | 3.133 |
| Estanterías cubiertas | 149 |
| **Incremento mensual de beneficio** | **+67.794 € (+16,2 %)** |
| Mejor modelo de predicción | Transformer (MSE: 299) |
| Mejor modelo de optimización | MLP (incremento: +1.648 €) |
| Enfoque final | Ensemble MLP + Transformer |

## Estructura del repositorio

```
├── mlops/          # Pipeline de ML: generación de datos → entrenamiento → evaluación → predicción
│   ├── models/     #   Arquitecturas de modelos (MLP, LSTM, Transformer, PPO)
│   ├── utils/      #   Motor de física de estanterías y base de conocimiento ChromaDB
│   ├── data/       #   Datos mensuales de ventas generados
│   ├── results/    #   Resultados de entrenamiento: pesos, gráficos, métricas
│   └── docs/       #   Documentación técnica
├── web/            # Dashboard web full-stack
│   ├── frontend/   #   Aplicación Next.js (subida, entrenamiento, evaluación, predicción)
│   └── backend/    #   API Express (ejecuta scripts de Python mediante procesos hijos)
├── auditorias/     # Informes y presentaciones de auditoría
└── archive/        # Prototipos iniciales y propuestas anteriores
```

## Inicio rápido

### Pipeline de MLOps

```bash
cd mlops
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Añade tu clave de API de OpenRouter

# Ejecutar el pipeline (en orden)
python 01_generate_monthly_sales.py
python 02_train_models.py
python 03_evaluate.py
python 04_ingest.py
python 05_predict.py
```

### Dashboard web

```bash
cd web
npm run install:all

# En terminales separados:
npm run dev:backend    # API Express en :3001
npm run dev:frontend   # Aplicación Next.js en :3000
```

## Stack tecnológico

- **ML**: PyTorch (MLP, LSTM, Transformer, PPO)
- **RAG**: ChromaDB + OpenRouter
- **Frontend**: Next.js + TypeScript + Tailwind CSS
- **Backend**: Express + TypeScript
- **Datos**: ~4.500 productos en 149 categorías de una cadena de supermercados española
