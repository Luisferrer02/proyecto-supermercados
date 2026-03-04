# BACKLOG DE PRODUCTO -- Supermarket Digital Twin (MLOps)

> Generado a partir del analisis del contenido de `mlops/`.
> Formato compatible con importacion a Jira.

---

## INDICE DE EPICAS

| ID     | Epica                                          | Prioridad |
|--------|------------------------------------------------|-----------|
| EP-01  | Pipeline de Generacion de Datos Sinteticos     | Alta      |
| EP-02  | Entrenamiento y Comparativa de Modelos ML      | Alta      |
| EP-03  | Sistema RAG e Ingestion de Conocimiento        | Alta      |
| EP-04  | Prediccion y Optimizacion de Lineales          | Alta      |
| EP-05  | Evaluacion, Metricas y Visualizacion           | Media     |
| EP-06  | Infraestructura, Reproducibilidad y DevOps     | Media     |
| EP-07  | Calidad de Datos y Validacion                  | Media     |

---

## EP-01: PIPELINE DE GENERACION DE DATOS SINTETICOS

**Descripcion**: Cubrir la generacion de datos mensuales de ventas a partir
del catalogo base de Mercadona (`products_macro.csv`), incluyendo el modo
heuristico local y el modo LLM via OpenRouter.

**Archivos implicados**: `01_generate_monthly_sales.py`, `01_augment_data.py`,
`products_macro.csv`

---

### HU-01.1: Generacion heuristica de ventas mensuales

**Titulo**: Como data engineer, quiero generar 12 CSVs de ventas mensuales
con multiplicadores estacionales deterministicos, para tener datos de
entrenamiento sin depender de APIs externas.

**Descripcion**: Ejecutar `01_generate_monthly_sales.py` sin `--use-llm`
debe producir 12 ficheros en `data/monthly/` con columnas de producto,
precio numerico, ventas estimadas, margen, ancho, rack_id y shelf_level.

**Criterios de aceptacion**:
- [ ] Se generan exactamente 12 CSVs (enero-diciembre 2025)
- [ ] Cada CSV contiene todas las columnas requeridas: Category, name, subtitle, price_numeric, estimated_monthly_sales, profit_margin_percentage, product_width_cm, rack_id, shelf_level
- [ ] Los precios en formato "0,36 EUR" se parsean correctamente a float
- [ ] Los multiplicadores estacionales se aplican por categoria y mes (ej: helados altos en verano)
- [ ] Ningun shelf supera los 300cm de ancho total
- [ ] El script es idempotente: ejecutar dos veces produce el mismo resultado con la misma seed

**Tareas tecnicas**:
- TT-01.1.1: Implementar parsing robusto de precios EUR espanoles (comas, simbolos)
- TT-01.1.2: Definir tabla de multiplicadores estacionales por categoria (12 meses)
- TT-01.1.3: Implementar asignacion de productos a racks con restriccion de 300cm
- TT-01.1.4: Crear directorio `data/monthly/` si no existe, con logging de progreso

**Prioridad**: Alta

---

### HU-01.2: Augmentacion de datos via LLM (OpenRouter)

**Titulo**: Como data engineer, quiero enriquecer el catalogo base usando
un LLM via OpenRouter, para obtener estimaciones de ventas y margenes mas
realistas cuando haya API disponible.

**Descripcion**: `01_augment_data.py` y el modo `--use-llm` de
`01_generate_monthly_sales.py` envian batches de productos a OpenRouter
(modelo arcee-ai/trinity-large-preview) y parsean la respuesta JSON.

**Criterios de aceptacion**:
- [ ] El modo `--dry-run` funciona sin API key y produce datos mock deterministicos
- [ ] Los batches se envian de 10-30 productos con retry exponencial
- [ ] Si el modelo primario falla, se usa el fallback (step-3.5-flash)
- [ ] Se hace checkpoint cada 50 batches para no perder progreso
- [ ] El rate-limit (HTTP 429) se maneja con espera minima de 5 segundos
- [ ] La salida `data/products_augmented.csv` tiene el mismo schema que el modo heuristico

**Tareas tecnicas**:
- TT-01.2.1: Implementar cliente OpenRouter con retry, backoff y modelo fallback
- TT-01.2.2: Implementar checkpointing de progreso a disco cada N batches
- TT-01.2.3: Validar que las respuestas LLM producen valores numericos validos (no NaN, no negativos)
- TT-01.2.4: Documentar variable de entorno OPENROUTER_API_KEY en .env.example

**Prioridad**: Media

---

### HU-01.3: Validacion del catalogo base

**Titulo**: Como data engineer, quiero validar que `products_macro.csv`
tiene el formato esperado antes de iniciar la generacion, para detectar
errores de datos antes de que corrompan el pipeline.

**Criterios de aceptacion**:
- [ ] Se valida que existen las columnas: Category, name, subtitle, price
- [ ] Se detectan y reportan filas con precio no parseable
- [ ] Se informa del numero total de productos y categorias encontradas
- [ ] Si el fichero no existe, el error es claro y no un traceback de pandas

**Tareas tecnicas**:
- TT-01.3.1: Crear funcion de validacion de schema CSV al inicio de cada script de generacion
- TT-01.3.2: Agregar log resumen: "Cargadas N categorias, M productos, K precios invalidos"

**Prioridad**: Media

---

## EP-02: ENTRENAMIENTO Y COMPARATIVA DE MODELOS ML

**Descripcion**: Entrenamiento de los 4 modelos (MLP, LSTM, Transformer, PPO)
sobre los datos sinteticos, evaluacion comparativa por MSE, y optimizacion
greedy de layouts usando los modelos entrenados.

**Archivos implicados**: `02_train_models.py`, `models/mlp.py`,
`models/lstm_model.py`, `models/transformer_model.py`, `models/ppo_agent.py`

---

### HU-02.1: Entrenamiento de modelos supervisados (MLP, LSTM, Transformer)

**Titulo**: Como ML engineer, quiero entrenar los tres modelos supervisados
sobre datos sinteticos de reasignacion de estanterias, para comparar cual
predice mejor el profit_lift.

**Descripcion**: `02_train_models.py` genera ~20,000 muestras sinteticas
(reasignaciones aleatorias con calculo de profit antes/despues), entrena
MLP (256-128-64-1), LSTM (2 capas, hidden=64), y Transformer (4 capas,
4 heads), y guarda los pesos en `results/`.

**Criterios de aceptacion**:
- [ ] Los 3 modelos se entrenan durante 80 epocas con Adam y ReduceLROnPlateau
- [ ] Se usa un split train/test (80/20) con seed fija
- [ ] Se reporta test MSE para cada modelo al finalizar
- [ ] Se guardan los pesos en results/mlp.pth, results/lstm.pth, results/transformer.pth
- [ ] El vector de entrada tiene 10 features: price, margin, sales, width, original_shelf, new_shelf, shelf_counts, n_shelves_used, rack_size
- [ ] Los datos sinteticos se generan usando `utils/retail_physics.py` con efectos de crowding, spread y price-tier

**Tareas tecnicas**:
- TT-02.1.1: Implementar generacion de datos sinteticos con reasignaciones aleatorias y calculo de profit diferencial
- TT-02.1.2: Implementar arquitectura MLP (Linear 10->256->128->64->1 con ReLU y Dropout)
- TT-02.1.3: Implementar arquitectura LSTM (2 capas, hidden=64, FC 64->32->1)
- TT-02.1.4: Implementar arquitectura Transformer (BatchNorm, Projection 10->128, 4 encoder layers, 4 heads)
- TT-02.1.5: Crear loop de entrenamiento comun con logging de loss por epoca

**Prioridad**: Alta

---

### HU-02.2: Entrenamiento del agente PPO

**Titulo**: Como ML engineer, quiero entrenar un agente PPO que aprenda
a intercambiar productos entre estanterias para maximizar el profit del
rack, como alternativa al enfoque supervisado.

**Descripcion**: `models/ppo_agent.py` define un entorno RackEnv (estado =
propiedades + shelf actual, accion = swap de dos productos, reward = delta
profit) y un ActorCritic con PPO clipped objective.

**Criterios de aceptacion**:
- [ ] El entorno soporta hasta 50 pasos por episodio
- [ ] El agente usa gamma=0.99, eps_clip=0.2, k_epochs=4
- [ ] Se reporta reward acumulado por episodio durante entrenamiento
- [ ] Se guarda el modelo entrenado en results/ppo.pth
- [ ] El layout final del PPO se exporta como CSV para comparacion

**Tareas tecnicas**:
- TT-02.2.1: Implementar RackEnv con estado (n_products * 5 features), accion (swap top-2), reward (delta profit)
- TT-02.2.2: Implementar ActorCritic con backbone compartido (128-128), actor head y critic head
- TT-02.2.3: Implementar PPOTrainer con clipped objective y advantage normalization

**Prioridad**: Media

---

### HU-02.3: Optimizacion greedy con modelos entrenados

**Titulo**: Como ML engineer, quiero usar cada modelo entrenado para
asignar productos a estanterias de forma greedy, para medir la ganancia
real de cada modelo frente al layout original.

**Criterios de aceptacion**:
- [ ] Para cada modelo, se itera sobre productos y se asigna cada uno al shelf con mayor profit_lift predicho
- [ ] Se respeta la restriccion de 300cm por shelf
- [ ] Se exporta un CSV de layout por modelo: results/rack_layout_{model}.csv
- [ ] Se calcula el profit total (EUR) original vs optimizado por modelo
- [ ] Se incluye un baseline greedy (sin ML) usando `optimize_rack_greedy()` de retail_physics.py

**Tareas tecnicas**:
- TT-02.3.1: Implementar bucle de asignacion greedy con prediccion por modelo
- TT-02.3.2: Exportar layouts y resultados numericos a results/training_results.json

**Prioridad**: Alta

---

## EP-03: SISTEMA RAG E INGESTION DE CONOCIMIENTO

**Descripcion**: Pipeline de ingestion de datos historicos en ChromaDB
y sistema de recuperacion de contexto temporal para alimentar al LLM
en la fase de prediccion.

**Archivos implicados**: `04_ingest.py`, `utils/knowledge_base.py`

---

### HU-03.1: Ingestion de CSVs mensuales en ChromaDB

**Titulo**: Como ML engineer, quiero ingestar los CSVs mensuales en una
base de conocimiento vectorial (ChromaDB), para que el sistema de
prediccion tenga contexto historico de ventas.

**Descripcion**: `utils/knowledge_base.py` agrupa productos por categoria,
genera resumenes en lenguaje natural (n_productos, ventas totales, top-5
vendedores), los embebe con sentence-transformers
(paraphrase-multilingual-MiniLM-L12-v2) y los almacena en ChromaDB
con metadata de ano, mes, categoria.

**Criterios de aceptacion**:
- [ ] Se ingesan todos los CSVs de data/monthly/ sin duplicados
- [ ] Cada chunk contiene: resumen de categoria, metadata (year, month, month_key, category, total_sales, avg_price, avg_margin, n_products)
- [ ] El embedding model es paraphrase-multilingual-MiniLM-L12-v2 (soporte espanol)
- [ ] La base se persiste en data/knowledge_base/ y sobrevive entre ejecuciones
- [ ] stats() retorna total de chunks, meses y categorias almacenadas

**Tareas tecnicas**:
- TT-03.1.1: Implementar ingest_csv() con agrupacion por categoria y generacion de summary
- TT-03.1.2: Implementar ingest_directory() con callback de progreso
- TT-03.1.3: Configurar ChromaDB con coleccion "monthly_sales" y distancia coseno
- TT-03.1.4: Implementar deteccion de duplicados por month_key + category

**Prioridad**: Alta

---

### HU-03.2: Recuperacion de contexto temporal (RAG retrieval)

**Titulo**: Como ML engineer, quiero recuperar contexto de los ultimos
N meses y del mismo mes del ano anterior, para que el LLM tenga
informacion de tendencias y estacionalidad al hacer predicciones.

**Criterios de aceptacion**:
- [ ] retrieve_context(target_month, n_months_back) retorna los N meses anteriores + mismo mes ano anterior
- [ ] Se puede filtrar por categoria
- [ ] El resultado incluye documentos y metadatas de ChromaDB
- [ ] Si no hay datos para un mes solicitado, se omite sin error

**Tareas tecnicas**:
- TT-03.2.1: Implementar retrieve_context() con queries por month_key
- TT-03.2.2: Implementar get_latest_month_data() y get_all_months_data()

**Prioridad**: Alta

---

### HU-03.3: Ingestion paralela (embeddings + entrenamiento)

**Titulo**: Como ML engineer, quiero ejecutar la ingestion de embeddings
y el entrenamiento de modelos en paralelo, para reducir el tiempo total
del pipeline.

**Descripcion**: `04_ingest.py` lanza dos threads: uno para ChromaDB
embeddings y otro para entrenar MLP + Transformer simultaneamente.

**Criterios de aceptacion**:
- [ ] Modo paralelo (por defecto) ejecuta ambos threads concurrentemente
- [ ] Modo `--sequential` ejecuta uno tras otro (para debugging)
- [ ] ProgressTracker con locks muestra estado en tiempo real (<< inicio, OK exito, !! error)
- [ ] Si un thread falla, el otro continua y se reporta el error al final

**Tareas tecnicas**:
- TT-03.3.1: Implementar ProgressTracker thread-safe con locks y elapsed time
- TT-03.3.2: Implementar threading con manejo de excepciones por thread
- TT-03.3.3: Agregar flag --sequential via argparse

**Prioridad**: Media

---

## EP-04: PREDICCION Y OPTIMIZACION DE LINEALES

**Descripcion**: Pipeline completo de prediccion mensual que combina RAG,
LLM forecasting, y optimizacion ensemble (MLP propone, Transformer valida)
para generar layouts optimizados.

**Archivos implicados**: `05_predict.py`, `utils/retail_physics.py`

---

### HU-04.1: Forecast de demanda con LLM + RAG

**Titulo**: Como usuario del sistema, quiero obtener multiplicadores de
demanda por categoria para un mes futuro, usando contexto historico RAG
y un LLM, para ajustar las ventas antes de optimizar el layout.

**Descripcion**: `05_predict.py` pasos 1-4: recupera contexto de ChromaDB,
construye prompt con historial + lista de productos, envia a OpenRouter,
parsea multiplicadores {categoria: float}, y los aplica a
estimated_monthly_sales.

**Criterios de aceptacion**:
- [ ] Se requiere parametro --month YYYY-MM
- [ ] Opcionalmente se filtra por --category
- [ ] El LLM retorna un dict JSON con multiplicadores por categoria (ej: {"Fruta": 1.2, "Marisco": 0.8})
- [ ] Si el LLM falla, se usan multiplicadores heuristicos estacionales como fallback
- [ ] --dry-run omite la llamada LLM y usa heuristicos directamente
- [ ] Las ventas ajustadas se clipean a minimo 1 (no negativos ni ceros)

**Tareas tecnicas**:
- TT-04.1.1: Implementar construccion de prompt con contexto RAG + productos actuales
- TT-04.1.2: Implementar parsing de respuesta LLM a dict de multiplicadores
- TT-04.1.3: Implementar fallback heuristico si LLM no responde
- TT-04.1.4: Aplicar multiplicadores y clipear ventas

**Prioridad**: Alta

---

### HU-04.2: Optimizacion ensemble (MLP propone, Transformer valida)

**Titulo**: Como usuario del sistema, quiero que el MLP genere N candidatos
de layout por rack y el Transformer seleccione el mejor, para obtener
layouts mas robustos que con un solo modelo.

**Descripcion**: `05_predict.py` paso 5: para cada rack, MLP genera
N layouts candidatos (greedy + variaciones con ruido), Transformer
puntua cada uno, y se selecciona el de mayor score.

**Criterios de aceptacion**:
- [ ] Se generan --n-candidates (default 5) layouts por rack
- [ ] El MLP usa asignacion greedy + variaciones con ruido aleatorio
- [ ] El Transformer puntua cada candidato (mayor score = mejor layout)
- [ ] Se selecciona el candidato con mayor score del Transformer
- [ ] Se respeta la restriccion de 300cm por shelf en todos los candidatos

**Tareas tecnicas**:
- TT-04.2.1: Implementar generacion de N candidatos con MLP (greedy + noise)
- TT-04.2.2: Implementar scoring de candidatos con Transformer
- TT-04.2.3: Implementar seleccion del mejor candidato y export a CSV

**Prioridad**: Alta

---

### HU-04.3: Reporte de resultados de optimizacion

**Titulo**: Como usuario del sistema, quiero ver una tabla comparativa
de profit original vs optimizado por rack, con total y porcentaje de
mejora, para evaluar el impacto de la optimizacion.

**Criterios de aceptacion**:
- [ ] Se imprime tabla por consola: Category | Original EUR | Optimized EUR | Lift EUR
- [ ] Se muestra fila TOTAL con porcentaje de mejora
- [ ] Se exporta el layout optimizado a results/optimized_YYYY_MM_*.csv
- [ ] Se exportan los multiplicadores de forecast a results/forecast_YYYY_MM.json

**Tareas tecnicas**:
- TT-04.3.1: Implementar formateo de tabla de resultados con alineacion
- TT-04.3.2: Implementar export de CSV optimizado y JSON de forecast

**Prioridad**: Media

---

## EP-05: EVALUACION, METRICAS Y VISUALIZACION

**Descripcion**: Generacion de reportes visuales comparativos entre
modelos y entre layouts originales y optimizados.

**Archivos implicados**: `03_evaluate.py`

---

### HU-05.1: Graficos comparativos de modelos

**Titulo**: Como profesor/evaluador, quiero ver graficos que comparen
el MSE de cada modelo y el profit obtenido por cada uno, para entender
visualmente que modelo funciona mejor.

**Criterios de aceptacion**:
- [ ] Grafico de barras de MSE por modelo (MLP, LSTM, Transformer, PPO)
- [ ] Grafico de barras de profit original vs optimizado por modelo
- [ ] Se identifica y resalta el mejor modelo por MSE y por profit
- [ ] Las graficas se guardan como PNG en results/
- [ ] Funciona en modo headless (backend Agg de matplotlib)

**Tareas tecnicas**:
- TT-05.1.1: Implementar carga de results/training_results.json
- TT-05.1.2: Implementar grafico de barras MSE comparativo
- TT-05.1.3: Implementar grafico de barras profit comparativo

**Prioridad**: Media

---

### HU-05.2: Visualizacion de racks (original vs optimizado)

**Titulo**: Como profesor/evaluador, quiero ver una visualizacion
side-by-side de la distribucion de productos en estanterias antes y
despues de la optimizacion, con colores por nivel.

**Criterios de aceptacion**:
- [ ] Barras horizontales side-by-side: layout original (izq) vs optimizado (der)
- [ ] Codificacion por color: dorado = eye level (3-5), azul = bottom (1-2), teal = top (6-7)
- [ ] Se muestra profit por shelf
- [ ] Se guarda como PNG en results/

**Tareas tecnicas**:
- TT-05.2.1: Implementar visualizacion de rack con barras por shelf y colores por nivel

**Prioridad**: Media

---

### HU-05.3: Diagrama alluvial de movimientos de productos

**Titulo**: Como profesor/evaluador, quiero un diagrama alluvial que
muestre como se mueven los productos entre estanterias despues de la
optimizacion, para entender el patron de reasignaciones.

**Criterios de aceptacion**:
- [ ] Columna izquierda = shelves originales, columna derecha = shelves optimizados
- [ ] Los flujos muestran productos que se mueven, con grosor proporcional a cantidad
- [ ] Cada flujo muestra nombres de productos individuales
- [ ] Labels de shelf incluyen conteo de productos
- [ ] Se guarda como PNG en results/

**Tareas tecnicas**:
- TT-05.3.1: Implementar diagrama alluvial con matplotlib (sin librerias externas de Sankey)

**Prioridad**: Baja

---

## EP-06: INFRAESTRUCTURA, REPRODUCIBILIDAD Y DEVOPS

**Descripcion**: Aspectos de infraestructura detectados como ausentes
o incompletos en el proyecto actual. Estos items se deducen de gaps
observados en los archivos existentes.

---

### HU-06.1: Fichero de dependencias y entorno

**Titulo**: Como desarrollador, quiero un requirements.txt completo
y un .env.example, para poder instalar y configurar el proyecto
desde cero sin errores.

**Descripcion**: Los scripts importan torch, pandas, numpy, scikit-learn,
chromadb, sentence-transformers, dotenv, matplotlib, y requests.
No existe un requirements.txt en la carpeta mlops.

**Criterios de aceptacion**:
- [ ] requirements.txt lista todas las dependencias con versiones pinneadas
- [ ] .env.example documenta OPENROUTER_API_KEY con valor placeholder
- [ ] pip install -r requirements.txt en un venv limpio no produce errores
- [ ] Se documenta la version minima de Python (>=3.10)

**Tareas tecnicas**:
- TT-06.1.1: Crear requirements.txt con: torch, pandas, numpy, scikit-learn, chromadb, sentence-transformers, python-dotenv, matplotlib, requests
- TT-06.1.2: Crear .env.example con OPENROUTER_API_KEY=your_key_here
- TT-06.1.3: Verificar instalacion limpia en venv nuevo

**Prioridad**: Alta

---

### HU-06.2: Script de ejecucion end-to-end

**Titulo**: Como desarrollador, quiero un script unico que ejecute
todo el pipeline en orden (generar datos, ingestar, entrenar, evaluar,
predecir), para poder reproducir el resultado completo con un solo
comando.

**Descripcion**: Actualmente hay que ejecutar 5 scripts en secuencia
manual. No existe un orquestador.

**Criterios de aceptacion**:
- [ ] Un unico `run_pipeline.py` o `Makefile` ejecuta los 5 pasos en orden
- [ ] Cada paso reporta exito/fallo antes de continuar
- [ ] Si un paso falla, se detiene el pipeline con mensaje claro
- [ ] Soporta flag --dry-run que se propaga a los scripts con LLM

**Tareas tecnicas**:
- TT-06.2.1: Crear run_pipeline.py que ejecute los 5 pasos secuencialmente
- TT-06.2.2: Agregar manejo de errores con exit codes por paso

**Prioridad**: Media

---

### HU-06.3: Seeds y reproducibilidad

**Titulo**: Como ML engineer, quiero que todas las operaciones aleatorias
usen seeds fijas y documentadas, para que los resultados sean
reproducibles entre ejecuciones.

**Criterios de aceptacion**:
- [ ] Cada script que usa numpy/torch/random fija una seed al inicio
- [ ] Las seeds se documentan en una constante visible al inicio del fichero
- [ ] Ejecutar el pipeline dos veces produce los mismos resultados (modo heuristico)

**Tareas tecnicas**:
- TT-06.3.1: Auditar todos los scripts y asegurar seed fija en numpy, torch y random
- TT-06.3.2: Centralizar constante SEED en un config comun si es posible

**Prioridad**: Media

---

### HU-06.4: Estructura de directorios automatica

**Titulo**: Como desarrollador, quiero que los scripts creen
automaticamente los directorios de salida (data/monthly, results,
data/knowledge_base) si no existen, para evitar errores de FileNotFoundError.

**Criterios de aceptacion**:
- [ ] Cada script crea sus directorios de salida con os.makedirs(exist_ok=True)
- [ ] No se producen errores si se ejecuta desde un clone limpio

**Tareas tecnicas**:
- TT-06.4.1: Agregar os.makedirs() al inicio de cada script para sus directorios de salida

**Prioridad**: Alta

---

## EP-07: CALIDAD DE DATOS Y VALIDACION

**Descripcion**: Validaciones y controles de calidad deducidos de la
logica de negocio implementada en `utils/retail_physics.py`.

---

### HU-07.1: Validacion de restricciones fisicas de estanteria

**Titulo**: Como ML engineer, quiero validar que ningun layout generado
por los modelos viole la restriccion de 300cm por shelf, para asegurar
que las recomendaciones son fisicamente realizables.

**Descripcion**: `retail_physics.py` contiene `validate_all_shelves()`
y `check_shelf_width()`, pero no se aplican consistentemente despues
de cada optimizacion.

**Criterios de aceptacion**:
- [ ] Despues de cada optimizacion (greedy, MLP, ensemble), se ejecuta validate_all_shelves()
- [ ] Si hay violaciones, se aplica enforce_shelf_constraint() automaticamente
- [ ] Se reporta el numero de violaciones corregidas
- [ ] El CSV final exportado pasa validacion sin violaciones

**Tareas tecnicas**:
- TT-07.1.1: Integrar validacion post-optimizacion en 02_train_models.py
- TT-07.1.2: Integrar validacion post-optimizacion en 05_predict.py
- TT-07.1.3: Agregar log de violaciones detectadas y corregidas

**Prioridad**: Alta

---

### HU-07.2: Tests unitarios para retail_physics

**Titulo**: Como ML engineer, quiero tests unitarios para las funciones
de calculo de profit, crowding penalty y shelf constraint, para asegurar
que los calculos de negocio son correctos.

**Descripcion**: `retail_physics.py` contiene logica compleja (crowding
penalty >5 productos, spread bonus +15%, diminishing returns en eye-level,
price-tier affinity) que actualmente no tiene tests.

**Criterios de aceptacion**:
- [ ] Test de compute_product_profit() con casos conocidos
- [ ] Test de crowding penalty: >5 productos en un shelf reduce multiplicador
- [ ] Test de spread bonus: usar mas shelves da +15%
- [ ] Test de price-tier affinity: premium en eye-level da +10%, premium en bottom da -15%
- [ ] Test de enforce_shelf_constraint(): productos se mueven si shelf > 300cm
- [ ] Todos los tests pasan con pytest

**Tareas tecnicas**:
- TT-07.2.1: Crear tests/test_retail_physics.py con pytest
- TT-07.2.2: Cubrir los 5 escenarios de calculo descritos en criterios
- TT-07.2.3: Agregar caso edge: rack vacio, un solo producto, shelf lleno

**Prioridad**: Media

---

## RESUMEN DE PRIORIZACION

### Sprint 1 (Alta prioridad -- Fundamentos)

| ID       | Historia                                          |
|----------|---------------------------------------------------|
| HU-01.1  | Generacion heuristica de ventas mensuales         |
| HU-02.1  | Entrenamiento de modelos supervisados             |
| HU-02.3  | Optimizacion greedy con modelos entrenados        |
| HU-03.1  | Ingestion de CSVs en ChromaDB                     |
| HU-03.2  | Recuperacion de contexto temporal (RAG)           |
| HU-04.1  | Forecast de demanda con LLM + RAG                 |
| HU-04.2  | Optimizacion ensemble (MLP + Transformer)         |
| HU-06.1  | Fichero de dependencias y entorno                 |
| HU-06.4  | Estructura de directorios automatica              |
| HU-07.1  | Validacion de restricciones fisicas               |

### Sprint 2 (Media prioridad -- Robustez)

| ID       | Historia                                          |
|----------|---------------------------------------------------|
| HU-01.2  | Augmentacion via LLM (OpenRouter)                 |
| HU-01.3  | Validacion del catalogo base                      |
| HU-02.2  | Entrenamiento del agente PPO                      |
| HU-03.3  | Ingestion paralela (threads)                      |
| HU-04.3  | Reporte de resultados de optimizacion             |
| HU-05.1  | Graficos comparativos de modelos                  |
| HU-05.2  | Visualizacion de racks                            |
| HU-06.2  | Script de ejecucion end-to-end                    |
| HU-06.3  | Seeds y reproducibilidad                          |
| HU-07.2  | Tests unitarios para retail_physics               |

### Sprint 3 (Baja prioridad -- Polish)

| ID       | Historia                                          |
|----------|---------------------------------------------------|
| HU-05.3  | Diagrama alluvial de movimientos                  |
