# Defensa individual — Guía de estudio · Julián

> Documento de uso personal. Si no quieres que se suba al repo, añade
> `documentacion/_defensa_julian.md` a `.gitignore` o bórralo tras la
> defensa.

---

## A. El proyecto en 60 segundos

**ShelfOPT** es un sistema MLOps que recomienda **la disposición óptima de productos en las estanterías de un supermercado** para maximizar el beneficio mensual.

- **Input**: catálogo Mercadona (~4 500 productos en 149 categorías, dataset Kaggle).
- **Pipeline**: 5 scripts Python encadenados (genera datos sintéticos mensuales → entrena modelos → evalúa → ingesta a base vectorial → predice y optimiza para un mes futuro).
- **Capa web**: backend Express (TypeScript) que ejecuta los scripts vía `child_process.spawn` y un frontend Next.js 16 que ofrece dos modos — **Optimizar** (flujo simple para el manager) y **Avanzado** (cada paso del pipeline expuesto).
- **Resultado** (con dataset sintético del propio pipeline):
  - 3 133 productos optimizados / 149 estanterías
  - **+67 794 €/mes (+16,2 %)** de mejora frente al layout original
  - Mejor predictor: Transformer (MSE 299)
  - Mejor optimizador: MLP (lift +1 648 €)
  - Enfoque final: **ensemble MLP + Transformer**

---

## B. Arquitectura (lo que debes saber dibujar en un papel)

```
                          ┌──────────────────────┐
                          │  Frontend Next.js    │
                          │  (web/frontend/)     │
                          └──────────┬───────────┘
                                     │ HTTP + SSE
                          ┌──────────▼───────────┐
                          │  Backend Express     │
                          │  (web/backend/)      │
                          └──────────┬───────────┘
                                     │ spawn() Python
                          ┌──────────▼───────────┐
                          │  Pipeline MLOps      │
                          │  (mlops/)            │
                          │                      │
                          │  01 → 02 → 03 → 04 → 05
                          │                      │
                          │  ChromaDB (.chromadb/)
                          │  OpenRouter LLM      │
                          └──────────────────────┘
```

### Tres capas que tienes que saber distinguir

| Capa | Carpeta | Lenguaje | Función |
|------|---------|----------|---------|
| Datos + ML | `mlops/` | Python 3.12 | 5 scripts del pipeline + `utils/` compartidos + `tests/` |
| API | `web/backend/` | TypeScript / Express | Recibe peticiones del frontend, lanza scripts Python, devuelve resultados |
| Interfaz | `web/frontend/` | TypeScript / Next.js 16 | UI para el usuario final (Optimizar) y para el equipo (Avanzado) |

### CI/CD y empaquetado

- `.github/workflows/ci.yml` — corre lint+tests+build en cada push con jobs condicionales por carpeta cambiada (`dorny/paths-filter`).
- `.github/workflows/release.yml` — al taggear `v*` publica imágenes Docker a GHCR.
- `docker-compose.yml` (+ overlay `prod`) — levanta backend + frontend + ChromaDB.

---

## C. Pipeline ML — paso a paso

| # | Script | Qué hace | Tecnología clave |
|---|--------|----------|------------------|
| 01 | `01_generate_monthly_sales.py` | Genera 12 CSVs (Ene-Dic 2025) a partir del catálogo base. 60-90 % de productos por mes, multiplicadores estacionales. Modo heurístico (rápido) o `--use-llm` (más realista). | LLM cascade vía OpenRouter |
| 02 | `02_train_models.py` | Entrena los **4 modelos** y los compara: MLP, LSTM, Transformer, PPO. Genera samples sintéticos con `retail_physics.generate_synthetic_training_data()`. | PyTorch |
| 03 | `03_evaluate.py` | Produce 4 gráficas PNG (MSE comparison, profit comparison, rack comparison, alluvial diagram). | Matplotlib |
| 04 | `04_ingest.py` | Dos hilos en paralelo: (1) embeddings de las categorías a ChromaDB; (2) entrena MLP+Transformer de producción. | ChromaDB + sentence-transformers |
| 05 | `05_predict.py` | Para un mes futuro: RAG retrieval → LLM forecast (cascada) → ensemble MLP+Transformer → produce CSV optimizado + explicaciones rule-based. | RAG + LLM + ensemble |

### Por qué ensemble MLP + Transformer (paradoja a recordar)

- **Transformer** tiene el mejor MSE (≈299) pero solo +€651 de lift.
- **MLP** tiene peor MSE (≈553) pero +€1 648 de lift (2,5×).
- Tomamos lo bueno de cada uno: **MLP genera 5 candidatos** (con ruido gaussiano creciente, "creative explorer") y el **Transformer puntúa cada candidato** ("careful judge"). El ganador es el de mayor score.

---

## D. La web — qué hace cada página

### Modo Optimizar (`/`) — para el usuario final

Tres fases en una sola pantalla:

1. **Upload**: drop zone para los `sales_*.csv`.
2. **Running**: barra de progreso con mensajes amigables (el parser convierte logs Python crudos en frases tipo "Entrenando modelo rápido", "Calculando la nueva disposición"). Internamente abre un `EventSource` al endpoint `/api/optimize/run` que devuelve eventos SSE `step`/`log`/`done`.
3. **Results**: 4 KPI cards + Sankey 7→7 + tabla top racks.

### Modo Avanzado (dropdown del TopNav)

| Página | Qué ofrece |
|--------|------------|
| `/upload` | Drop zone + listado/borrado de CSVs subidos |
| `/ingest` | Ejecutar `04_ingest.py` con LiveLog |
| `/train` | Ejecutar `02_train_models.py` + tabla MSE/RMSE/profit por modelo |
| `/evaluate` | Ver las 4 PNG generadas, con modal lightbox al hacer click |
| `/predict` | Ejecutar `05_predict.py` + Sankey global + ShelfMap por rack + tabla explicaciones |

---

## E. Tecnologías que tienes que poder defender

### 1. Next.js 16 (App Router) + React 19

- **App Router** = la organización por carpetas dentro de `app/`. Cada `page.tsx` es una ruta.
- **Server Components vs Client Components**: nuestras páginas son `"use client"` porque usamos `useState`, `useEffect`, `EventSource`. Esto es necesario para SSE.
- **Por qué Next y no Vite**: routing por archivos, optimización de imágenes y SSR si lo necesitamos en el futuro.
- **Turbopack**: el bundler nuevo de Vercel (replaza webpack). Más rápido en `npm run dev`.

> **Si te preguntan**: "¿Por qué Next en lugar de React puro?" → File-system routing, SSR opcional, configuración mínima, build optimizado con Turbopack.

### 2. Tailwind CSS 4 + shadcn/ui

- **Tailwind**: utility-first CSS. Las clases (`bg-primary`, `rounded-lg`) se traducen a CSS en build-time.
- **shadcn/ui**: NO es una librería instalada, son **componentes copiados** dentro de `components/ui/`. Usamos: `Card`, `Button`, `Badge`, `Progress`, `Table`, `Tabs`.
- **Por qué shadcn y no MUI/Chakra**: control total del código (no es una dependencia opaca), se adapta al tema con CSS variables (`globals.css`).

> **Si te preguntan**: "¿Cómo está el sistema de colores?" → CSS variables en `app/globals.css` (`--primary`, `--secondary`, `--accent`, `--background`...). Verde bosque `#09543D` primario, marrón `#461E10` secundario, cream `#FFFCF1` de fondo.

### 3. SSE (Server-Sent Events)

- **Qué es**: conexión HTTP **unidireccional** persistente del servidor al cliente. El servidor mantiene la conexión abierta y empuja eventos cuando ocurren.
- **Por qué SSE y no WebSockets**: nuestros logs van solo de Python → navegador. WebSockets serían bidireccionales y excesivos.
- **Por qué SSE y no polling**: latencia 0 frente a polling cada N segundos.
- **Headers clave** que pone el backend: `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: keep-alive`.
- **API navegador**: `new EventSource(url)`, luego `es.addEventListener("log", …)`. El navegador maneja auto-reconnect.

> **Si te preguntan**: "¿Cómo se reciben los logs en tiempo real?" → SSE, una conexión persistente HTTP con `text/event-stream` que emite eventos `step`/`log`/`done`. Cliente los consume con `EventSource`.

### 4. Express + TypeScript

- Backend muy fino: 6 rutas (`upload`, `train`, `evaluate`, `ingest`, `predict`, `optimize`).
- Usa `child_process.spawn` para lanzar los scripts Python. El stdout del Python se enruta al cliente vía SSE.
- `multer` v2 para los uploads multipart (después del fix de seguridad).

### 5. PyTorch (los 4 modelos)

- **MLP** (Multi-Layer Perceptron) — red feedforward, 3 capas (256/128/64).
- **LSTM** — recurrent, procesa una secuencia de productos uno a uno (forced ordering).
- **Transformer** — atención bidireccional. d_model=128, 4 layers, 4 heads, Pre-LN, BatchNorm en input, GELU. 150 epochs, LR 1e-4, grad-clip 1.0.
- **PPO** (Reinforcement Learning) — actor-critic. Underperforma porque el problema tiene reward inmediato y calculable; RL brilla con reward retardado.

> **Si te preguntan los hiperparámetros del Transformer**: el doc `documentacion/01_justificacion_hiperparametros_transformer.md` los explica con tablas. Los tres cambios que más MSE redujeron: **BatchNorm** sobre input, **Pre-LN** en vez de Post-LN, y **4 layers con d_model=128** (de los 3 niveles probados, el v3).

### 6. RAG + ChromaDB + LLM cascade

- **ChromaDB**: base de datos vectorial. Guarda los summaries de cada categoría como embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
- **RAG**: recuperamos contexto (mismo mes año anterior + 2 meses recientes) y se lo damos al LLM para que prediga multiplicadores por categoría.
- **Cascada de 8 modelos LLM** (`mlops/utils/llm_client.py`): si Llama 70B se cae, prueba Qwen 80B, luego GPT-OSS 120B, etc. Solo si los 8 fallan caemos a heurística.

### 7. CI/CD — GitHub Actions

- `ci.yml`: lint (ruff/bandit/pip-audit/eslint), typecheck (tsc), tests (pytest), build Docker. Jobs condicionales por path.
- `release.yml`: al taggear `v*`, build + push a GHCR.

### 8. Docker / Docker Compose

- Dockerfile **multi-stage** del backend: stage 1 compila TS, stage 2 base Python 3.12 + Node 24, venv en `/opt/venv`, mlops instalado editable (`pip install -e mlops/`).
- Frontend usa **standalone output** de Next.js (no necesita `node_modules` en runtime).
- `docker-compose.yml` (dev) + `docker-compose.prod.yml` (overlay con límites de CPU/RAM, sin build, pull desde GHCR).

---

## F. Lo que has hecho tú — 13 commits explicados

> Orden cronológico de más antiguo a más nuevo, en `dev` y `prod`.

### `863533c` — *prueba-inicial-simulación* (2026-02-18)

- **Qué**: subiste el primer prototipo de simulación en `codigo-base-prueba/` (Python con `simulation.py`, `sales_simulation.csv/json`).
- **Por qué**: era la fase exploratoria del proyecto, antes de tener nada del pipeline real.
- **Defensa**: "Fue el código semilla con el que validamos que el problema de optimización tenía sentido formularse como ML. No forma parte del producto final, está en `archive/` ahora."

### `129416d` — *Simulación comparativa* (2026-02-18)

- **Qué**: segundo prototipo en `codigo-base-simulacion-comparativa/` (`multi_simulation.py` y datos comparativos).
- **Por qué**: evolución del primer prototipo para comparar varias estrategias de layout simultáneamente.
- **Defensa**: "Iteración del prototipo inicial: comparamos varios layouts a la vez para tener idea del rango de mejora posible. También código exploratorio en `archive/`."

### `b73f9af` — *Create BACKLOG_JIRA.md* (2026-03-04)

- **Qué**: añadiste `BACKLOG_JIRA.md` (623 líneas) con la planificación completa por épicas e historias.
- **Por qué**: organizar el trabajo del sprint inicial siguiendo la estructura Jira.
- **Defensa**: "Aporté la planificación del proyecto en formato de épicas/historias. Sirvió de guía para repartir trabajo entre los 5 miembros."

### `bbb7e2c` — *Merge branch 'main'* (2026-03-04)

- **Qué**: merge mecánico. Trajo `auditorias/segunda.md` al main.
- **Defensa**: trivial; no lo destaques.

### `8c1acc9` — *Create final.md* (2026-04-19)

- **Qué**: creaste `auditorias/final.md` (173 líneas) con notas de la auditoría final.
- **Por qué**: documentar lo que la auditoría A3 nos señaló como crítico (los 3 fallos en demo: Docker, concurrencia, fallback LLM).
- **Defensa**: "Documenté formalmente las conclusiones de la auditoría final para que el equipo pudiera atacar los hallazgos críticos antes de la entrega."

### `7fcb24c` — *Reorganización Auditorias* (2026-04-19)

- **Qué**: reorganizaste la carpeta `auditorias/` en dos subcarpetas: `Auditados/` (lo que recibimos) y `Auditores/` (lo que entregamos). Añadiste `AuditoriaFinal.pdf` en `Auditados/`.
- **Por qué**: limpieza estructural antes de entrega.
- **Defensa**: trivial pero útil. "Higiene de carpetas: separar lo que nuestros auditores nos dieron vs. lo que nosotros entregamos a los proyectos que auditamos."

### `f96f7d2` — *Reorganización, arreglos código y modificación de web* (2026-04-19) ⭐ **EL GORDO**

> Este es el commit más importante. 31 archivos, **3 140 líneas añadidas, 331 borradas**. Cierra el sprint de mejoras post-auditoría.

**Qué metió** (resumen agrupado):

- **`documentacion/`** — los 8 docs teóricos de defensa (justificación hiperparámetros Transformer, análisis PPO, MSE con unidades, test set independiente, error RAG, drift, perfiles cliente, README índice).
- **`mlops/utils/`** — 4 archivos nuevos:
  - `llm_client.py` — cascada de 8 modelos OpenRouter con failover.
  - `csv_schema.py` — validador del esquema de los CSVs mensuales.
  - `explainability.py` — explicaciones rule-based ("por qué se movió este producto").
  - `model_persistence.py` — guardado versionado con hash + timestamp.
- **`mlops/02_train_models.py`** — split train/val/test + rack-holdout con hash SHA-256.
- **`mlops/04_ingest.py`** — pre-flight con `csv_schema.validate_directory()`.
- **`mlops/05_predict.py`** — fallback LLM → heurística automático, `_source` en el JSON.
- **`web/backend/src/`**:
  - `routes/optimize.ts` (NUEVO) — endpoint unificado `/run` (SSE chain ingest+predict) + `/results` con KPIs/Sankey/racks.
  - `services/pythonRunner.ts` — detección de venv POSIX y fallback a Windows.
- **`web/frontend/app/`**:
  - `page.tsx` — refactorizado al flujo unificado de 3 fases con Sankey.
  - `predict/page.tsx`, `evaluate/page.tsx`, `ingest/page.tsx`, `train/page.tsx` — adaptaciones.
- **`web/frontend/components/`**:
  - `ShelfSankey.tsx` (NUEVO) — Sankey 7→7 en SVG puro.
  - `TopNav.tsx` — dropdown "Avanzado".
- **`web/frontend/lib/`**:
  - `progressParser.ts` (NUEVO) — convierte logs Python en mensajes amigables.
  - `metrics.ts` (NUEVO) — fuente única de KPIs canónicos (+16,2 %).

**Por qué**: este commit cerró la fase de respuesta a la auditoría A3. Cada cambio responde a un hallazgo concreto.

**Defensa — cómo presentarlo**: *"En este commit consolido la fase de mejoras post-auditoría. Resuelve los principales hallazgos (fallback LLM automático, persistencia versionada, explicabilidad, split independiente, validación de CSVs) y a la vez introduce el flujo unificado de la home con el componente Sankey. Documento todo en `documentacion/` con referencias línea a línea al código."*

**Si te preguntan por algún archivo concreto** (es probable):
- `optimize.ts` → "Encadena `04_ingest.py` y `05_predict.py` en un solo SSE. La función `aggregate()` lee el CSV optimizado y devuelve KPIs + matriz 7×7 de movimientos para el Sankey."
- `ShelfSankey.tsx` → "Renderiza 7 baldas antes + 7 baldas después con cintas de cubic Bezier proporcionales al número de productos que cambian de balda. SVG puro porque las librerías Sankey arrastran 250 KB."
- `progressParser.ts` → "Traduce ~20 patrones de log (`Epoch 40/80`, `Querying LLM`, `Step 5:` …) en frases en español que el usuario manager puede entender."
- `llm_client.py` → "Cascada de failover. Intenta el modelo 1, si falla pasa al 2, etc. Solo cuando todos fallan devuelve `None` y el caller cae a heurística."
- `csv_schema.py` → "Validador no destructivo. Define columnas requeridas/opcionales, tipos y rangos. Devuelve un `ValidationResult` con errores y warnings."

### `ea9d462` — *traducir LiveLog y páginas Upload/Ingest a español* (2026-05-02)

- **Qué**: traducción consistente a español de 3 archivos (`LiveLog.tsx`, `upload/page.tsx`, `ingest/page.tsx`).
- **Por qué**: la auditoría señaló "UX para perfil IT, no para usuario final" (hallazgo A3-M03). El user objetivo es un manager de supermercado.
- **Defensa**: "Atacamos el hallazgo de UX para usuario no técnico. Mantuvimos terminología técnica solo en tooltips."

### `f76d4b2` — *paleta verde unificada y legend descriptiva* (2026-05-02)

- **Qué**: refactor de `components/ShelfMap.tsx`. Sustituí la paleta "semáforo" (rojo/naranja/amarillo/verde) por 4 tonos verdes graduados; función `profitColor` → `profitTier` (devuelve color + label). Legend nueva con rangos numéricos en €.
- **Por qué**: la paleta antigua chocaba con el resto del frontend (verde primary). Coherencia visual.
- **Defensa**: "Cambié el estilo 'semáforo' por un gradiente verde coherente con la theme primary. Ahora la leyenda muestra rangos numéricos explícitos (`<80 €`, `80-200 €`, `200-500 €`, `>500 €`) en vez de etiquetas vagas como 'High'/'Low'."

### `f9866b9` — *formato europeo en gráficas y traducción a español* (2026-05-02)

- **Qué**: en `app/train/page.tsx` añadí `tickFormatter` con `.toLocaleString("es-ES")` y unidades `€` / `€²` en tooltips de Recharts. Tradujimos h1, botones, headers de tabla.
- **Defensa**: "Mejora cosmética pero importante para el evaluador: los números en `€12.345` y no `12345`. Recharts permite custom formatters en `<Tooltip>` y `<YAxis tickFormatter>`."

### `af43feb` — *key global y dropdown enriquecido con categoría* (2026-05-02)

- **Qué**: en `app/predict/page.tsx`:
  1. Añadí el `ShelfSankey` (vista global) reutilizando `/api/optimize/results` — sin duplicar lógica en backend.
  2. Sustituí el dropdown numérico de rack ("0", "1", …) por `"Rack 47 — Aves y jamón cocido"` precalculado con un `useMemo`.
  3. Tabla "Profit Summary" ahora muestra nombre de categoría legible.
  4. Color rojo/verde condicional del lift.
- **Por qué**: era el "fallo tonto" que el profesor señaló — la lista de racks parecía interminable porque mostraba solo ids numéricos sin contexto.
- **Defensa**: *"Reutilicé el endpoint `/api/optimize/results` (que ya devuelve `movements`) para tener el Sankey global también en /predict sin tocar backend. El dropdown enriquecido se monta con un `useMemo` que recorre `results.products` una sola vez y construye un `Map<id, label>`."*

### `f926e6b` — *modal lightbox para ampliar gráficas* (2026-05-02)

- **Qué**: en `app/evaluate/page.tsx` añadí cards clicables con affordance (cursor-zoom-in, hover ring, texto "Pulsa para ampliar") y un modal *lightbox* con cierre por Escape, click fuera y botón X.
- **Cómo funciona**:
  - State `expanded` con `{url, title} | null`.
  - `useEffect` que registra `keydown` para Escape y bloquea `document.body.style.overflow = "hidden"` mientras el modal está montado (scroll-lock del fondo).
  - Click en el backdrop cierra; `e.stopPropagation()` en el contenido para que el click interno no cierre.
  - `role="dialog" aria-modal="true"` para accesibilidad.
- **Por qué no shadcn Dialog**: no estaba instalado y añadir el primitivo (radix) sería 30 KB extra para un solo uso. 50 LOC propias resuelven el 100 % del caso.
- **Defensa**: *"Modal accesible (`role=dialog`, `aria-modal`), cierra por las 3 vías estándar (X, Escape, click fuera), scroll-lock del fondo, con `e.stopPropagation()` para evitar que un click en el modal lo cierre. Sin dependencias nuevas."*

### `4103dda` — *Presentaciones finales* (2026-05-06)

- **Qué**: añadiste los `.pptx` finales en `documentacion/` (boceto + final).
- **Defensa**: trivial pero útil. La presentación final está rediseñada con la misma paleta que la web (`presentacion_redisenada_estilo_web.pptx` si la tienes generada).

---

## G. Preguntas tipo del tribunal y respuestas listas

### Sobre el proyecto en general

| Pregunta | Respuesta corta |
|----------|-----------------|
| "¿Por qué 4 modelos diferentes?" | Cada uno representa una familia distinta: MLP (feedforward), LSTM (sequential), Transformer (attention), PPO (RL). Querer comparar nos da rigor experimental para defender la elección final. |
| "¿Por qué el ensemble y no solo el Transformer?" | Paradoja MSE vs lift: Transformer mejor MSE pero solo +€651 lift; MLP peor MSE pero +€1 648 lift. Ensemble: MLP propone 5 candidatos con ruido, Transformer puntúa, gana el mejor. Resultado: +€67 794. |
| "¿Cómo de fiable es el dataset?" | Es sintético, generado por `01_generate_monthly_sales.py` (heurístico o vía LLM). Honestidad: el test set ahora tiene 3 niveles (swaps disjuntos, rack-holdout, temporal) precisamente para no inflar MSE. Hash SHA-256 del split en `results/test_split_hash.json`. |
| "¿Qué pasa si OpenRouter se cae?" | Cascada de 8 modelos con retries. Si fallan los 8, el código cae automáticamente a `heuristic_forecast()` y persiste `_source: "heuristic"` en el JSON. Esto resuelve el hallazgo A3-I03. |
| "¿Cómo aseguráis que no se pierde un modelo entrenado?" | `model_persistence.save_model()` guarda dos copias: una canónica `<name>.pth` y otra archivada `<name>_<timestamp>_<hash>.pth` en `results/models/` con un `manifest.json` que documenta cada versión. |

### Sobre el frontend (tu zona principal)

| Pregunta | Respuesta corta |
|----------|-----------------|
| "¿Cómo se comunica el frontend con el backend durante una optimización larga?" | Server-Sent Events. El frontend abre un `EventSource` al endpoint `/api/optimize/run`. El backend mantiene la conexión HTTP abierta y emite eventos `step`/`log`/`done`. El navegador los recibe con `es.addEventListener("log", …)`. |
| "¿Por qué Next.js?" | App Router (file-system routing), client components con SSE nativo, optimización de imágenes, Turbopack para dev. |
| "¿Por qué los gráficos en SVG y no con Recharts?" | El Sankey 7→7 no está bien soportado en Recharts. d3-sankey arrastra 250 KB. 100 LOC de SVG resuelven el 80 % del problema sin dependencias. Para gráficas estándar (barras) sí usamos Recharts (`/train`). |
| "¿Cómo funciona el modal de `/evaluate`?" | State `expanded` con `{url, title}`. `useEffect` añade `keydown` para Escape y bloquea scroll del body mientras está montado. `e.stopPropagation()` en el contenido evita que el click interno cierre. `role="dialog" aria-modal="true"`. |
| "¿Por qué el dropdown de racks mostraba 'Rack 47' antes y ahora 'Rack 47 — Aves y jamón cocido'?" | Era el "fallo tonto" del feedback del profesor. Lo arreglé con un `useMemo` que recorre productos una vez y construye `Map<rack_id, "Rack X — Categoría">`. |
| "¿Cómo está el sistema de tipografía?" | Inter para body, DM Sans para headings (`font-heading` con `extrabold uppercase tracking-tight`), Geist Mono para code. Definido en `app/layout.tsx`. |
| "¿Qué es Tailwind?" | Framework CSS utility-first. Las clases (`bg-primary rounded-lg px-4`) se traducen a CSS en build. Las CSS variables del tema están en `globals.css` (`--primary`, etc.). |
| "¿Qué es shadcn?" | NO es una librería instalada — es un patrón de **componentes copiados al repo**. Vivimos en `components/ui/`. Ventaja: control total, sin dependencia opaca. |

### Sobre el commit gordo (`f96f7d2`)

Si te preguntan "¿qué hiciste tú en ese commit que tiene 3 140 líneas?":

> *"Fue el commit que cerró el sprint de mejoras post-auditoría. Consolidaba en una sola entrega: (1) la documentación teórica de los 8 hallazgos persistentes, (2) las utilidades nuevas del lado Python (cascada LLM, validador CSV, explicabilidad, persistencia versionada), (3) el endpoint backend unificado `/api/optimize`, y (4) la home rediseñada con el flujo de 3 fases y el Sankey. Mi peso principal en este commit fue el frontend completo (home, Sankey, parser de progreso, metrics) y la documentación; las utilidades Python las trabajamos en paralelo con Víctor que tenía más profundidad en el pipeline."*

### Sobre los problemas que reportarás en P3 del cuestionario

#### Problema 1 — dropdown de racks
*"El backend devolvía `rack_id` como string crudo (`'0'`, `'1'`, … `'148'`). El select del frontend renderizaba esos números sin contexto y resultaba imposible identificar qué rack era cuál. Lo arreglé enriqueciendo cada opción con la categoría: `'Rack 47 — Aves y jamón cocido'`. Usé un `useMemo` para recorrer `results.products` una vez y montar un `Map<id, label>` ordenado numéricamente."*

#### Problema 2 — mezcla inglés/español
*"El boilerplate de shadcn arrancó en inglés. Conforme metíamos features en español quedaron strings huérfanas que rompían el flujo del usuario manager (que el profesor remarcó). Tradujimos sistemáticamente los 5 archivos del modo Avanzado y el `LiveLog`, manteniendo terminología técnica (MSE, RMSE) solo en tooltips. Los tooltips de Recharts ahora formatean con `.toLocaleString('es-ES')` y unidades `€`/`€²`."*

---

## H. Si el tribunal te pregunta por algo del pipeline Python que no dominas

Estrategias honestas:

1. **Sobre los hiperparámetros del Transformer**: remite al documento `documentacion/01_justificacion_hiperparametros_transformer.md`. Tres claves: BatchNorm en input, Pre-LN, 4 layers d_model=128.

2. **Sobre PPO**: el documento `documentacion/02_analisis_ppo_underperformance.md` lo explica. Resumen: PPO falla porque el problema tiene reward inmediato y calculable; RL brilla con reward retardado (ajedrez, Go).

3. **Sobre el test set**: 3 niveles (swaps independientes, rack-holdout, temporal). Hash SHA-256 del split en `results/test_split_hash.json` para reproducibilidad.

4. **Sobre la RAG**: ChromaDB guarda summaries por categoría (sentence-transformers multilingual-MiniLM). Se recupera el mismo mes año anterior + 2 meses recientes y se le pasa al LLM como contexto.

5. **Si te preguntan algo de mlops/utils que no recuerdas bien**: *"Esa parte la trabajó principalmente Víctor, mi rol fue frontend y la coordinación entre capas. Pero el docs/05_error_rag_cuantificado.md explica el comportamiento esperado."*

---

## I. Checklist final para mañana por la mañana

- [ ] Repasa los 4 puntos de la sección A ("el proyecto en 60 segundos") hasta poder soltarlos sin mirar.
- [ ] Dibuja la arquitectura de la sección B en un papel — si la entiendes a mano, la entiendes.
- [ ] Estudia las preguntas de la sección G — cada una es una pregunta tipo de tribunal.
- [ ] Para los 5 commits del 2 de mayo (refactor visual), ten claro qué archivo tocaste en cada uno: `LiveLog/upload/ingest` (ea9d462), `ShelfMap` (f76d4b2), `train` (f9866b9), `predict` con Sankey (af43feb), `evaluate` con modal (f926e6b).
- [ ] Para el commit gordo `f96f7d2`, ten clara la narrativa: "consolidé las mejoras post-auditoría: 8 docs + 4 utils Python + endpoint optimize unificado + home rediseñada con Sankey".
- [ ] Lleva el repositorio abierto en el portátil por si el tribunal te pide enseñar un archivo. Branches importantes: `prod` (la default) y `dev`.
- [ ] Si te preguntan por testing: hay 13 archivos `.py` en `mlops/tests/` (`pytest`), tests `.test.ts` para las 6 rutas backend, tests `.test.tsx` para los 5 componentes frontend. CI en `.github/workflows/ci.yml` los corre con jobs condicionales por path.
- [ ] **Honestidad si no sabes algo**: di "esa parte la trabajó principalmente [Víctor/Raúl/Luis/Samu] y yo coordiné desde [tu zona]". Nunca inventes detalles técnicos.

¡Suerte mañana!
