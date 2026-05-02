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
├── mlops/              # Pipeline de ML
│   ├── models/         #   Arquitecturas (MLP, LSTM, Transformer, PPO)
│   ├── utils/          #   Módulos compartidos (física de estanterías, datos, entrenamiento, LLM)
│   ├── tests/          #   Tests unitarios (pytest)
│   ├── data/           #   Datos mensuales de ventas generados
│   ├── results/        #   Pesos de modelos, gráficos, métricas
│   └── docs/           #   Documentación técnica
├── web/                # Dashboard web full-stack
│   ├── frontend/       #   Next.js (subida, entrenamiento, evaluación, predicción)
│   └── backend/        #   API Express (ejecuta scripts Python mediante procesos hijos)
├── .github/workflows/  # CI/CD (GitHub Actions)
│   ├── ci.yml          #   Lint, tests, typecheck, build Docker (en cada push)
│   └── release.yml     #   Publica imágenes en GHCR (en tags v*)
├── docker-compose.yml  # Stack local: backend + frontend + chromadb
├── auditorias/         # Informes y presentaciones de auditoría
└── archive/            # Prototipos iniciales y propuestas anteriores
```

## Inicio rápido

### Pipeline de MLOps

```bash
cd mlops
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .  # Instala mlops/ como paquete para que utils/ y models/ sean importables
cp .env.example .env  # Añade tu clave de API de OpenRouter

# Ejecutar el pipeline (en orden)
python 01_generate_monthly_sales.py
python 02_train_models.py
python 03_evaluate.py
python 04_ingest.py
python 05_predict.py
```

### Tests

```bash
cd mlops
pip install pytest
python -m pytest tests/ -v
```

### Dashboard web

```bash
cd web
npm run install:all

# En terminales separados:
npm run dev:backend    # API Express en :3001
npm run dev:frontend   # Aplicación Next.js en :3000
```

### Docker

```bash
# Levantar todo el stack
docker compose up --build

# Solo frontend + backend (sin chromadb standalone)
docker compose up backend frontend

# Producción (imágenes de GHCR)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## CI/CD

Cada push ejecuta automáticamente (solo los jobs afectados por los archivos cambiados):

| Job | Qué hace | Se ejecuta si cambia |
|-----|----------|---------------------|
| Python quality | ruff + bandit + pip-audit + pytest (61 tests) | `mlops/` |
| Backend quality | TypeScript typecheck + build + npm audit | `web/backend/` |
| Frontend quality | TypeScript typecheck + ESLint + Next.js build | `web/frontend/` |
| Docker build | Construye ambas imágenes (sin push) | Cualquiera de los anteriores o Dockerfiles |

Al crear un tag `v*` se publican las imágenes en GitHub Container Registry (GHCR).

## Stack tecnológico

- **ML**: PyTorch (MLP, LSTM, Transformer, PPO)
- **RAG**: ChromaDB + OpenRouter
- **Frontend**: Next.js + TypeScript + Tailwind CSS
- **Backend**: Express + TypeScript
- **CI/CD**: GitHub Actions + Docker Compose + GHCR
- **Testing**: pytest (Python), ESLint + tsc (TypeScript)
- **Datos**: ~4.500 productos en 149 categorías de una cadena de supermercados española
