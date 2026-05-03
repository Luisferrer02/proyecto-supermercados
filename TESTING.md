# Testing — ShelfOpt

## Resumen

| Suite | Tests | Cobertura | Tiempo |
|-------|------:|----------:|-------:|
| MLOps (Python) | 94 | 81% | ~29s |
| Backend (Node/Express) | 85 | 80% | ~20s |
| Frontend (React/Next.js) | 110 | ~75% | ~18s |
| E2E (Playwright) | 9 specs | n/a | ~25s |
| **Total** | **298+** | | |

---

## Estructura de tests

```
proyecto-supermercados/
├── mlops/tests/                    # Python: unit + integration
│   ├── test_models.py              # MLP, Transformer, LSTM
│   ├── test_training.py            # optimize_rack_mlp, on_epoch_callback
│   ├── test_retail_physics.py      # scoring, placement, constraints
│   ├── test_data_io.py             # CSV I/O, forecast parsing
│   ├── test_csv_schema.py          # validation rules
│   ├── test_explainability.py      # explanation generation
│   ├── test_llm_client.py          # LLM client with mocked HTTP
│   ├── test_model_persistence.py   # save/load/archive model lifecycle
│   └── test_knowledge_base.py      # vector store operations
│
├── web/backend/
│   ├── src/routes/*.test.ts        # Route-level (supertest)
│   ├── src/services/*.test.ts      # Service unit tests
│   └── tests/contract.test.ts      # Contract tests vs @shared/api-types
│
├── web/frontend/
│   ├── app/**/page.test.tsx        # Page-level (RTL + fetch mock + SSE)
│   ├── components/*.test.tsx       # Component unit tests
│   ├── lib/*.test.ts               # Utility/library tests
│   └── mocks/                      # Shared test infrastructure
│       ├── handlers.ts             # Fetch mock con route matching
│       └── sse-helpers.ts          # MockEventSource utilities
│
├── web/shared/
│   └── api-types.ts                # Source of truth para contratos front↔back
│
└── web/e2e/
    ├── tests/                      # Playwright specs
    │   ├── home-optimize.spec.ts
    │   ├── upload-manage.spec.ts
    │   ├── train-evaluate.spec.ts
    │   └── predict-flow.spec.ts
    └── fixtures/                   # CSV de prueba para uploads
```

---

## Comandos

### MLOps (Python)

```bash
cd mlops

# Instalar el paquete en modo desarrollo (necesario la primera vez)
pip install -e .
pip install pytest pytest-cov

# Ejecutar todos los tests con cobertura
python -m pytest tests/ -v \
  --cov=utils --cov=models \
  --cov-report=term-missing \
  --cov-fail-under=70 \
  --cov-config=.coveragerc

# Solo un archivo
python -m pytest tests/test_models.py -v

# Generar HTML de cobertura
python -m pytest tests/ --cov=utils --cov=models --cov-report=html
# Abrir: htmlcov/index.html
```

### Backend (Node.js)

```bash
cd web/backend

# Ejecutar todos los tests
npx jest

# Con cobertura
npx jest --coverage

# Solo contract tests
npx jest --testPathPatterns="contract"

# Solo un archivo de ruta
npx jest --testPathPatterns="upload.test"

# Watch mode durante desarrollo
npx jest --watch
```

**Cobertura**: se genera en `web/backend/coverage/` (text + lcov + html).

### Frontend (Next.js/React)

```bash
cd web/frontend

# Ejecutar todos los tests
npx jest

# Con cobertura
npx jest --coverage

# Solo page-level tests
npx jest --testPathPatterns="app/.*page\\.test"

# Solo component tests
npx jest --testPathPatterns="components/"

# Un test específico
npx jest --testPathPatterns="evaluate/page"

# Watch mode
npx jest --watch
```

**Cobertura**: se genera en `web/frontend/coverage/` (text + lcov + html).

### E2E (Playwright)

```bash
cd web/e2e

# Instalar dependencias y browsers (solo la primera vez)
npm install
npx playwright install --with-deps chromium

# Ejecutar todos los tests (arranca backend + frontend automáticamente)
npx playwright test

# Con UI interactiva (debug visual)
npx playwright test --ui

# Un spec específico
npx playwright test tests/home-optimize.spec.ts

# Ver el último report HTML
npx playwright show-report
```

**Requisito**: El backend se arranca con `MOCK_PYTHON=1` (no necesita Python real).

### Todo junto (desarrollo local)

```bash
# Desde la raíz del proyecto
cd mlops && python -m pytest tests/ -v && cd ..
cd web/backend && npx jest --coverage && cd ..
cd web/frontend && npx jest --coverage && cd ..
cd web/e2e && npx playwright test && cd ..
```

---

## Integración Continua (CI)

El workflow está en `.github/workflows/ci.yml` y se ejecuta en cada push/PR.

### Jobs

| Job | Trigger | Qué hace |
|-----|---------|----------|
| `changes` | siempre | Detecta qué directorios cambiaron |
| `python-quality` | `mlops/**` | Ruff + Bandit + pip-audit + pytest con cobertura |
| `backend-quality` | `web/backend/**` | tsc + build + npm audit |
| `backend-test` | `web/backend/**` | Jest con cobertura |
| `frontend-quality` | `web/frontend/**` | tsc + lint + Next.js build + npm audit |
| `frontend-test` | `web/frontend/**` | Jest con cobertura |
| `e2e-test` | backend o frontend | Playwright con MOCK_PYTHON |
| `docker-build` | Dockerfiles o source | Build de imágenes (sin push) |

### Cómo ver resultados

1. **GitHub Actions** > pestaña "Actions" del repositorio
2. Click en el run del commit/PR
3. Cada job muestra su output; tests fallidos aparecen en rojo
4. E2E: si falla, el artifact `playwright-report` se sube automáticamente (7 días)

### Path filtering

Los jobs solo se ejecutan si cambiaron archivos relevantes:
- Cambio solo en `mlops/` → solo corre `python-quality`
- Cambio en `web/frontend/` → corre `frontend-quality` + `frontend-test` + `e2e-test`
- Cambio en ambos → corren todos

---

## Arquitectura de testing

### Pirámide

```
         ┌─────────────┐
         │    E2E      │  9 specs — flujo completo con browser real
         │ (Playwright)│  MOCK_PYTHON=1, sin Python real
         ├─────────────┤
         │   Page      │  45 tests — transiciones de estado,
         │  (RTL)      │  API calls, SSE streams, UI behavior
         ├─────────────┤
         │ Integration │  85 tests — supertest contra Express,
         │ (Jest+node) │  contract tests vs shared types
         ├─────────────┤
         │    Unit     │  94+65 — funciones puras, componentes
         │(pytest/jest)│  aislados, lógica de negocio
         └─────────────┘
```

### Mocking strategy

| Capa | Qué se mockea | Cómo |
|------|---------------|------|
| MLOps unit | HTTP (LLM API), filesystem | `unittest.mock.patch`, `tmp_path` |
| Backend route | `pythonRunner` (no spawn real) | `jest.mock('../services/pythonRunner')` |
| Frontend page | `global.fetch` + `EventSource` | `mocks/handlers.ts` + `MockEventSource` |
| E2E | Python processes | `MOCK_PYTHON=1` en el backend |

### Contract tests

El archivo `web/shared/api-types.ts` es la fuente de verdad de los tipos de respuesta de la API. Los contract tests (`web/backend/tests/contract.test.ts`) validan en runtime que las respuestas reales del backend conforman esas interfaces.

Si cambias la forma de un endpoint:
1. Actualiza `web/shared/api-types.ts`
2. Los contract tests pasan
3. El frontend usa los mismos tipos → TypeScript detecta incompatibilidades

### MOCK_PYTHON mode

Cuando `MOCK_PYTHON=1`:
- `pythonRunner.ts` delega a `mockPython.ts`
- Emite eventos SSE simulados (log, step, done) con ~30ms de delay
- Copia fixtures predefinidas al `RESULTS_DIR`
- Los endpoints de resultados funcionan normal (leen las fixtures copiadas)

Esto permite E2E tests rápidos sin instalar Python ni modelos ML.

---

## Añadir nuevos tests

### Nuevo endpoint en backend

1. Crea el route test co-located: `src/routes/newRoute.test.ts`
2. Usa `buildApp()` + `createTmpDirs()` del helper
3. Añade el handler al contract test si tiene response shape nuevo
4. Actualiza `web/shared/api-types.ts` con la interfaz

### Nueva página en frontend

1. Crea `app/nueva-pagina/page.test.tsx` co-located
2. Mockea `LiveLog` si usa SSE streaming
3. Usa `setupFetchMock()` + las helpers de `mocks/sse-helpers.ts`
4. Añade handler en `mocks/handlers.ts` para nuevos endpoints

### Nuevo script Python

1. Añade tests en `mlops/tests/test_nuevo.py`
2. Si se expone vía backend, añade fixture en `web/backend/src/services/fixtures/`
3. Actualiza `mockPython.ts` si necesita mock para E2E

---

## Thresholds y quality gates

| Suite | Mínimo cobertura | Configurado en |
|-------|-----------------|----------------|
| MLOps | 70% | `mlops/.coveragerc` + `--cov-fail-under=70` |
| Backend | sin mínimo (0%) | `web/backend/jest.config.ts` |
| Frontend | sin mínimo (0%) | `web/frontend/jest.config.ts` |

Para activar un mínimo en backend/frontend, editar `coverageThreshold` en el `jest.config.ts` correspondiente:

```typescript
coverageThreshold: {
  global: {
    statements: 75,
    branches: 60,
    functions: 70,
    lines: 75,
  },
},
```

---

## Troubleshooting

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: No module named 'utils'` | Ejecuta `pip install -e .` en `mlops/` |
| Tests E2E fallan por timeout | Verifica que puertos 3000/3001 estén libres |
| `scrollIntoView is not a function` | Ya resuelto en `jest.setup.ts` |
| Frontend tests lentos | Usa `--maxWorkers=50%` en CI si hay OOM |
| Coverage no refleja cambios | Borra `coverage/` y vuelve a ejecutar |
