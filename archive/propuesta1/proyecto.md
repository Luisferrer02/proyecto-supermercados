# Proyecto académico: Optimización de distribución en supermercado (Simulación + Optimización)

## 1. Objetivo y alcance
Optimizar la distribución de categorías y productos en un supermercado discretizado como grid, maximizando profit/ventas esperado bajo restricciones físicas, de categoría y de accesibilidad.

## 2. Formulación del problema (SSAP)
El problema se formaliza como una variante del Shelf Space Allocation Problem (SSAP): asignación de productos a posiciones (estantería/balda/slot) con restricciones y objetivo de maximización de beneficio esperado.

## 3. Modelo del supermercado (grid)
Se representa el supermercado como una cuadrícula 2D; cada celda puede ser pasillo, estantería o puerta. Las estanterías son accesibles solo desde celdas de pasillo ortogonales.

## 4. Modelo de datos (persistencia + simulación)
Se recomienda persistencia relacional (PostgreSQL) para integridad y analítica; el motor de simulación trabaja con estructuras desnormalizadas en memoria para iterar rápidamente.

## 5. Datos sintéticos y seeding
Se definen archivos seedeados reproducibles: categorías, productos, layout, asignación inicial y calendario. Se simulan ventas con estacionalidad y ruido (Poisson/NegBin) y efectos de tráfico/visibilidad/adyacencia.

## 6. Simulación de cliente (ABM ligero)
Se simula el comportamiento agregado de clientes como agentes: entrada por puerta, movimiento por pasillos (ruta corta + desviación probabilística), compras condicionadas por demanda base, tráfico, balda y adyacencias.

## 7. Optimización
Se recomienda un enfoque híbrido: greedy como baseline, Algoritmo Genético como exploración global y Simulated Annealing como refinamiento local. La función fitness incorpora demanda prevista y efectos espaciales.

## 8. Validación y métricas
Incluye integridad referencial, consistencia de simulación, comparación contra baselines (original, greedy, random) y métricas de convergencia.

## 9. Roadmap
Faseado en: seeding+BD, simulación grid, modelo predictivo, optimización, evaluación, y fase 2 con LLM para informes.

---

## Anexo A. Contenido original (verbatim)

> Nota: Se incluye íntegramente para garantizar que no se pierde información. Las secciones anteriores lo reorganizan para lectura y defensa.

```text
Ahora tengo suficiente información técnica y académica para construir una respuesta rigurosa y completa. Procedamos con la estructuración del proyecto.

***

# Definición Técnica y Estratégica: Optimización de Distribución de Productos en Supermercado

## 1. Definición Formal del Problema

### Clasificación del problema

Este problema se clasifica como un **problema de optimización combinatoria con asignación cuadrática (Quadratic Assignment Problem)**, perteneciente a la clase NP-hard. Concretamente, combina: [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S030504832200041X)

- **Shelf Space Allocation (SSA):** asignar productos a posiciones en estanterías maximizando beneficio. [arxiv](https://arxiv.org/abs/2310.08568)
- **Asignación con restricciones:** cada producto debe ocupar una posición física válida respetando dimensiones, categorías y capacidad.
- **Optimización multi-objetivo implícita:** aunque la función objetivo principal sea el profit, hay subobjetivos en tensión (maximizar visibilidad, minimizar costes de recolocación, respetar agrupaciones por categoría). [arxiv](https://arxiv.org/abs/2310.08568)

**No es puramente Reinforcement Learning**, aunque RL puede usarse como método de resolución. El problema en sí es de optimización combinatoria con restricciones; RL es una estrategia de solución. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1319157812000353)

### Formulación matemática

Sean:
- \(P\) el conjunto de productos, \(S\) el conjunto de posiciones (estantería, balda, posición horizontal)
- \(x_{p,s} \in \{0,1\}\): variable binaria, 1 si el producto \(p\) se asigna a la posición \(s\)
- \(G_p\): beneficio unitario del producto \(p\)
- \(D_p(t)\): demanda estimada del producto \(p\) en el periodo \(t\)
- \(\alpha_s\): factor de visibilidad/tráfico de la posición \(s\) (zona caliente → \(\alpha\) alto)

**Función objetivo:**

\[
\max \sum_{p \in P} \sum_{s \in S} G_p \cdot D_p(t) \cdot \alpha_s \cdot x_{p,s}
\]

**Restricciones:**

1. **Asignación única:** cada producto se asigna a exactamente una posición:
\[
\sum_{s \in S} x_{p,s} = 1, \quad \forall p \in P
\]

2. **Capacidad:** cada posición alberga como máximo un producto:
\[
\sum_{p \in P} x_{p,s} \leq 1, \quad \forall s \in S
\]

3. **Compatibilidad de categoría:** el producto \(p\) solo puede ir en estanterías de su categoría \(C_p\):
\[
x_{p,s} = 0, \quad \forall s \notin S_{C_p}
\]

4. **Restricciones físicas:** altura, peso y dimensiones del producto deben ser compatibles con la balda. [arxiv](https://arxiv.org/abs/2310.08568)

5. **Restricciones de bloque:** productos de la misma categoría deben ocupar posiciones contiguas. [arxiv](https://arxiv.org/abs/2310.08568)

6. **Incompatibilidades:** ciertos productos no pueden estar adyacentes (p. ej., productos de limpieza junto a alimentos).

***

## 2. Arquitectura Propuesta

La arquitectura debe separar tres capas funcionales:

| Capa | Componente | Tecnología sugerida |
|---|---|---|
| **Datos** | Almacenamiento, ingesta, validación | PostgreSQL + JSON columns |
| **Simulación** | Motor de grid, recolocación, evaluación | Python (NumPy/SciPy) |
| **Optimización** | Algoritmos, búsqueda, evaluación fitness | Python (DEAP, OR-Tools, Stable-Baselines3) |
| **Análisis** | Métricas, reportes, LLM (fase 2) | Python + API LLM |

El flujo operativo es: **Datos → Simulación del estado actual → Algoritmo de optimización → Simulación del estado propuesto → Evaluación comparativa → Iteración**.

***

## 3. Modelo de Datos Mejorado y Justificado

### Evaluación del esquema propuesto

Tu esquema original es un punto de partida correcto pero **insuficiente** para un proyecto académico riguroso. Las carencias principales son:

- No modela las **dimensiones físicas** del producto ni de la balda (crítico para restricciones de capacidad). [arxiv](https://arxiv.org/abs/2310.08568)
- No incluye **categoría del producto** como atributo explícito (solo la estantería tiene categoría).
- No modela **relaciones entre productos** (complementarios, sustitutivos, incompatibles).
- No separa la **posición en el grid** (macro: dónde está la estantería) de la **posición interna** (micro: qué balda y slot).

### Modelo mejorado

**Producto:**

| Campo | Tipo | Justificación |
|---|---|---|
| `id` | UUID | Identificador único |
| `nombre` | String | Descriptivo |
| `categoria_id` | FK → Categoría | Permite restricciones de bloque  [arxiv](https://arxiv.org/abs/2310.08568) |
| `precio` | Float | Atributo comercial |
| `coste` | Float | Necesario para calcular margen real |
| `margen_unitario` | Float (calculado) | \(G_p = \text{precio} - \text{coste}\) |
| `ancho_cm` | Float | Restricción física de facing  [arxiv](https://arxiv.org/abs/2310.08568) |
| `alto_cm` | Float | Restricción de altura de balda |
| `peso_kg` | Float | Restricción de peso máximo por balda |
| `ventas_30d` | Integer | Demanda reciente |
| `ventas_30d_anio_anterior` | Integer | Estacionalidad interanual |
| `popularidad` | Float  [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1319157812000353) | Normalizado |
| `perfil_estacional` | Vector [scholars.cityu.edu](https://scholars.cityu.edu.hk/en/studentTheses/retail-shelf-space-and-location-allocation-problems-and-related-o/) | Índice de demanda mensual (ver sección 4) |
| `posicion_asignada_id` | FK → Posición | Asignación actual |

**Categoría:**

| Campo | Tipo | Justificación |
|---|---|---|
| `id` | UUID | Identificador |
| `nombre` | String | P. ej., "Lácteos", "Limpieza" |
| `incompatible_con` | Array[UUID] | Restricciones de no-adyacencia |

**Estantería:**

| Campo | Tipo | Justificación |
|---|---|---|
| `id` | UUID | Identificador |
| `categoria_id` | FK → Categoría | Restricción de asignación |
| `grid_x`, `grid_y` | Integer | Posición macro en el grid del supermercado |
| `num_baldas` | Integer | Estructura interna |
| `ancho_cm` | Float | Capacidad horizontal total |

**Posición (slot):**

| Campo | Tipo | Justificación |
|---|---|---|
| `id` | UUID | Identificador granular |
| `estanteria_id` | FK → Estantería | Referencia padre |
| `balda` | Integer | Nivel vertical (0 = suelo) |
| `offset_x` | Float | Posición horizontal en la balda |
| `altura_max_cm` | Float | Restricción física  [arxiv](https://arxiv.org/abs/2310.08568) |
| `peso_max_kg` | Float | Restricción física |
| `factor_visibilidad` | Float  [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1319157812000353) | Baldas a nivel de ojos ≈ 0.9, suelo ≈ 0.3 |

**Relación Producto-Producto (para cross-selling):**

| Campo | Tipo |
|---|---|
| `producto_a_id` | FK |
| `producto_b_id` | FK |
| `tipo` | Enum: complementario, sustitutivo, incompatible |
| `coeficiente` | Float |

### Normalización vs. modelo orientado a simulación

Para este proyecto recomiendo un **modelo relacional normalizado en 3NF para almacenamiento**, con **desnormalización en memoria durante la simulación**. La razón: la simulación necesita acceso O(1) a las posiciones y productos (arrays/matrices NumPy), pero la persistencia debe garantizar integridad referencial. PostgreSQL permite ambas cosas con columnas JSONB para datos semi-estructurados como el perfil estacional. [estuary](https://estuary.dev/blog/postgresql-vs-mongodb/)

***

## 4. Modelado de la Cuadrícula

### Representación recomendada: modelo híbrido

La opción más adecuada es un **sistema de coordenadas 2D (matriz) con grafo de adyacencia implícito**: [etd.auburn](https://etd.auburn.edu/bitstream/handle/10415/4494/Eozgormusphd.pdf?isAllowed=y&sequence=2)

- **Matriz 2D** para la disposición espacial: cada celda \((i,j)\) puede contener una estantería, un pasillo o estar vacía. Para un supermercado mediano (800–1500 m²), un grid de **20×30 celdas** (cada celda ≈ 1.5–2 m²) es realista. [shopfittingmanufacturer](https://shopfittingmanufacturer.com/supermarket-shelves-layout-design/)
- **Grafo de adyacencia** derivado de la matriz para calcular distancias, flujo de clientes y zonas conectadas. No es necesario almacenar el grafo explícitamente; se genera on-the-fly desde la matriz.

### Dimensionamiento

Un supermercado mediano tiene entre **40 y 120 estanterías** dependiendo del formato. Para la simulación recomiendo **60–80 estanterías** con 4–6 baldas cada una, generando entre **240–480 posiciones** totales. Esto produce un espacio de búsqueda manejable pero no trivial. [trolleymfg](https://www.trolleymfg.com/what-is-the-standard-retail-shelf-height/)

### Modelado de tráfico/zonas calientes

El tráfico de clientes se modela como un **mapa de calor estático** superpuesto al grid, con valores \(\alpha_{i,j} \in [0,1]\) representando la probabilidad relativa de que un cliente visite la celda \((i,j)\). Las zonas típicamente calientes son: [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1319157812000353)

- Entrada y salida
- Perímetro del supermercado (recorrido natural)
- Zona de cajas registradoras
- Pasillos principales

Un modelo más avanzado usa una **matriz de transición de Markov** entre secciones, como demostró el estudio con probabilidades de transición entre departamentos. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1319157812000353)

***

## 5. Estrategia de Modelado Predictivo

### Evaluación de la propuesta frío-calor (0-1)

Tu propuesta de escala continua  para estacionalidad es **conceptualmente correcta pero insuficiente** como implementación. Un escalar no captura la variación mensual de un producto que tiene picos en diciembre y junio pero caídas en marzo. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1319157812000353)

### Propuesta robusta: perfil estacional mensual

Reemplazar el escalar por un **vector de 12 componentes** (uno por mes), normalizado respecto a la media:

\[
\text{perfil\_estacional}_p = \left[\frac{v_{p,1}}{\bar{v}_p}, \frac{v_{p,2}}{\bar{v}_p}, \ldots, \frac{v_{p,12}}{\bar{v}_p}\right]
\]

donde \(v_{p,m}\) son las ventas históricas del producto \(p\) en el mes \(m\) y \(\bar{v}_p\) es la media mensual. Un valor de 1.5 en diciembre indica demanda 50% superior a la media. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1319157812000353)

### Modelo predictivo recomendado

Para un proyecto académico, la mejor opción es **regresión con feature engineering** por su interpretabilidad y suficiencia:

- **Variables predictoras:** perfil estacional del mes actual, ventas_30d, ventas_30d_año_anterior, categoría, precio, posición actual (factor_visibilidad × factor_tráfico).
- **Variable objetivo:** ventas esperadas en los próximos 30 días.
- **Modelo:** Gradient Boosting (XGBoost/LightGBM) por su capacidad de capturar no-linealidades y su interpretabilidad vía SHAP.
- **Prevención de data leakage:** nunca incluir como feature la posición futura (solo la actual o ninguna); usar validación temporal (train en meses anteriores, test en meses posteriores), nunca random split. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1319157812000353)

Series temporales tipo ARIMA/LSTM son excesivas salvo que tengas datos diarios de varios años. RL no es un modelo predictivo, sino de decisión.

***

## 6. Estrategia de Optimización Recomendada

| Criterio | Greedy | Simulated Annealing | Algoritmo Genético | Programación Lineal (MILP) | RL (DQN) |
|---|---|---|---|---|---|
| **Calidad solución** | Baja (local) | Media-Alta | Alta | Óptima (si tratable) | Media-Alta |
| **Complejidad impl.** | Muy baja | Baja | Media | Media-Alta | Alta |
| **Escalabilidad** | Excelente | Buena | Buena | Limitada (>500 vars.) | Buena |
| **Interpretabilidad** | Alta | Media | Media | Alta | Baja |
| **Adecuación académica** | Insuficiente solo | Buena | Muy buena | Excelente si es tratable | Buena pero arriesgada |

### Recomendación: enfoque híbrido en dos niveles

1. **Algoritmo Genético como motor principal**: cada cromosoma codifica una asignación completa producto→posición; la función fitness es la función objetivo definida en §1. La población evoluciona con crossover (intercambio de bloques de categoría) y mutación (swap de dos productos). [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8903274/)

2. **Greedy constructivo como inicialización**: genera la población inicial con heurísticas razonables (productos de mayor margen en posiciones de mayor visibilidad), acelerando la convergencia.

3. **Simulated Annealing como refinamiento local**: aplicado a las mejores soluciones del GA para explorar vecindarios cercanos y escapar de óptimos locales.

Esta combinación es la más adecuada para un proyecto académico: demuestra conocimiento de múltiples técnicas, es implementable en tiempo razonable y produce resultados defendibles ante un tribunal. [etd.auburn](https://etd.auburn.edu/bitstream/handle/10415/4494/Eozgormusphd.pdf?isAllowed=y&sequence=2)

***

## 7. Elección de Base de Datos Justificada

### Recomendación: PostgreSQL

| Requisito | PostgreSQL | MongoDB | Neo4j |
|---|---|---|---|
| Integridad referencial | ✅ Nativa con FK | ❌ Manual | ⚠️ Limitada |
| Consultas analíticas | ✅ SQL completo, window functions | ⚠️ Aggregation pipeline | ❌ No diseñado para esto |
| Relaciones producto-posición | ✅ JOINs eficientes | ⚠️ Embedding/referencia | ✅ Nativo |
| Datos semi-estructurados | ✅ JSONB nativo | ✅ Nativo | ❌ |
| Transacciones ACID | ✅ Completas | ⚠️ Desde v4.0 | ⚠️ Limitadas |
| Recolocaciones masivas | ✅ Transacciones atómicas | ❌ Sin atomicidad multi-doc robusta | ⚠️ |

PostgreSQL es la elección óptima porque: (a) la integridad referencial es crítica cuando se recolocan masivamente productos; (b) las consultas analíticas (rankings, agregaciones por categoría, comparaciones temporales) son el core del análisis; (c) JSONB permite almacenar el perfil estacional y metadatos flexibles sin sacrificar el modelo relacional. [sevensquaretech](https://www.sevensquaretech.com/mongodb-vs-postgresql/)

Neo4j sería un complemento interesante **solo si** se modelan relaciones complejas de cross-selling como grafo, pero no como base de datos principal.

***

## 8. Plan de Validación

### Integridad referencial
- Constraints de FK en PostgreSQL garantizan que todo producto apunte a una posición válida y toda posición pertenezca a una estantería existente.
- CHECK constraints: `margen_unitario >= 0`, `ventas_30d >= 0`, `factor_visibilidad BETWEEN 0 AND 1`.

### Coherencia de simulación
- **Invariante fundamental:** en todo momento, \(\sum_{p} x_{p,s} \leq 1\) para cada posición \(s\) y \(\sum_{s} x_{p,s} = 1\) para cada producto \(p\). Verificar antes y después de cada recolocación.
- **Test de regresión:** la configuración original (antes de optimizar) debe producir un profit calculable y consistente con datos históricos.

### Validación del optimizador
- **Baseline:** profit con la configuración actual (sin optimizar).
- **Random baseline:** profit promedio de 1000 asignaciones aleatorias válidas.
- **Métricas:** \(\Delta\text{profit}\%\), convergencia del GA (fitness vs. generaciones), diversidad de la población, tiempo de ejecución.

### Simulación de datos realistas
Generar datos sintéticos con distribuciones empíricas: ventas ~ Poisson(λ por categoría), precios ~ LogNormal, estacionalidad con patrones sinusoidales + ruido. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1319157812000353)

***

## 9. Roadmap Técnico

| Fase | Duración | Entregable |
|---|---|---|
| **1. Diseño** | 2 semanas | Modelo E-R, definición formal, esquema SQL |
| **2. Datos** | 2 semanas | Base de datos poblada, generador de datos sintéticos, validaciones |
| **3. Grid + Simulación** | 3 semanas | Motor de simulación, representación del grid, cálculo de profit |
| **4. Modelo predictivo** | 2 semanas | Predicción de demanda con XGBoost, validación temporal |
| **5. Optimización** | 3 semanas | GA + SA implementados, benchmarks vs. baselines |
| **6. Evaluación** | 1 semana | Métricas, gráficas de convergencia, análisis de sensibilidad |
| **7. LLM (fase 2)** | 2 semanas | Integración con LLM para informes automáticos |

### Estructura de datos para el LLM (fase 2)

El LLM debe recibir un **JSON estructurado** con:
- Resumen de la optimización: profit_antes, profit_después, delta_porcentual.
- Top-10 recolocaciones con mayor impacto.
- Métricas agregadas por categoría.
- Alertas: productos con caída de ventas, posiciones infrautilizadas.

Para **evitar alucinaciones**: (a) nunca pedir al LLM que genere números; proporcionarlos precalculados; (b) usar prompts con instrucciones de no inventar datos; (c) validar la salida del LLM contra los datos de entrada (fact-checking automático). [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1319157812000353)

Para que la optimización sea creíble, los datos sintéticos deben parecer “de supermercado real” y al mismo tiempo ser controlables y reproducibles. A continuación te propongo un diseño paso a paso.

***

## 1. Objetivos y principios del dataset sintético

El dataset debe permitir:

- Entrenar el modelo de demanda (ventas por producto y tiempo) con estacionalidad, tendencia y ruido. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1877050922007657/pdf?md5=a8df912471597776f983f874bb14e2d0&pid=1-s2.0-S1877050922007657-main.pdf)
- Evaluar el impacto de la **posición en el grid** y el tráfico sobre el beneficio.  
- Testear el optimizador con muchas configuraciones distintas, manteniendo restricciones físicas y de categoría.  

Principios:

- Control (parámetros explícitos: nº productos, nº días, fuerza de estacionalidad, ruido, etc.).  
- Separación clara entre “verdad subyacente” (demanda intrínseca) y efectos de posición y promociones, para evitar leakage.  
- Posibilidad de regenerar los datos con una semilla fija.

***

## 2. Entidades que vamos a sintetizar

1. **Catálogo de productos** (dimensión estática).  
2. **Layout del supermercado** (grid, estanterías, zonas calientes).  
3. **Calendario y estacionalidad** (día, mes, festivos, temporadas).  
4. **Ventas históricas** (serie temporal producto × día, afectada por estacionalidad, precio, categoría, tráfico).  
5. Opcional: **promociones** y **ruido de eventos exógenos**.

Todo esto alimenta tus tablas: `Producto`, `Estanteria`, `AsignacionProductoEstanteria`, `VentasHistoricas`, `ZonaTrafico`.

***

## 3. Generación del catálogo de productos

### 3.1. Definir categorías y tamaños

- Elige 10–20 categorías típicas: lácteos, frescos, bebidas, snacks, limpieza, higiene, etc.  
- Asigna a cada categoría:  
  - Una **estacionalidad tipo** (p.ej. bebidas frías suben en verano, turrones en diciembre). [kaggle](https://www.kaggle.com/datasets/abdullah0a/retail-sales-data-with-seasonal-trends-and-marketing)
  - Rango de precio medio.  
  - Rango de margen objetivo.

### 3.2. Generar productos por categoría

Para cada producto \(p\):

- `categoria_id`: categórica.  
- `precio`: muestrea de una distribución log-normal o normal truncada por categoría (precios típicos de 1–15 €).  
- `coste`: `precio * U(0.5, 0.85)` para simular márgenes realistas.  
- `margen_unitario`: derivado.  
- Dimensiones físicas (ancho_facing, alto, peso) con distribuciones típicas por categoría.  
- Parámetros “verdaderos” de demanda:  
  - `base_demand_p`: media diaria base (por ejemplo, 0.1–5 uds/día) con distribución muy sesgada (unos pocos súper vendidos, muchos de rotación baja). [arxiv](https://arxiv.org/html/2312.14095v1)
  - `seasonality_amp_p`: amplitud de estacionalidad (0–0.5, siendo 0 sin estacionalidad).  
  - `trend_p`: pequeña tendencia lineal (creciente o decreciente).  

Estos parámetros no se exponen al modelo; se usan solo para generar la serie de ventas.

***

## 4. Generación del layout y del tráfico

### 4.1. Grid y estanterías

- Define un grid 2D, p.ej. 40×30 celdas.  
- Coloca:  
  - Entrada/salida y cajas (frente a las cuales el tráfico será alto).  
  - Pasillos principales (caminos horizontales y verticales).  
  - Estanterías:  
    - Cada estantería ocupa varias celdas consecutivas (p.ej. 4–8 celdas en línea).  
    - Cada estantería tiene 4–6 baldas (niveles).  
- Genera entre 300 y 400 posiciones de balda (= estantería lógica) como antes definimos.

### 4.2. Campo de tráfico (heatmap)

Define un campo \(T(x,y) \in [0,1]\):

- Parte de una combinación de funciones suaves (p.ej. suma de gausianas) centradas en entradas, cajas y pasillos principales. [isarsoft](https://www.isarsoft.com/article/using-heat-maps-to-analyze-traffic-flow-the-isarsoft-approach)
- Normaliza de forma que la media sea ~0.5 y los máximos ~1.  
- Asigna a cada estantería \(s\) el valor \(T(x_s, y_s)\) de la celda donde se encuentra.  
- Opcional: varía el tráfico por franja horaria o por temporada con un factor multiplicativo pequeño.

***

## 5. Generación del calendario y estacionalidad

### 5.1. Calendario base

- Define un horizonte de simulación: p.ej. 365 días.  
- Para cada día \(t\):  
  - `fecha`, `dia_semana`, `mes`, `es_festivo`, `es_navidad`, `es_verano`, etc.  
- Estacionalidad global:  
  - `seasonal_year(t) = sin(2π * day_of_year / 365)` y `cos(…)`. [smartersupplychains.substack](https://smartersupplychains.substack.com/p/why-demand-forecasters-should-learn)
  - Para cada categoría, define una curva de peso sobre esta base, p.ej.:  
    - Bebidas frías: pico en verano.  
    - Chocolates: pico en invierno, navidad.  

### 5.2. Estacionalidad por producto

Para cada producto:

\[
E_p(t) = 1 + seasonality\_amp_p \cdot f_{cat(p)}(t)
\]

donde \(f_{cat(p)}(t)\) es la curva normalizada de su categoría (entre -1 y 1).

***

## 6. Generación de ventas históricas

### 6.1. Demanda “verdadera” diaria

Para cada producto \(p\) y día \(t\):

1. **Demanda base** con estacionalidad y tendencia:

\[
\lambda_{p,t}^{base} = base\_demand_p \cdot E_p(t) \cdot (1 + trend_p \cdot t)
\]

2. **Efecto de precio** (opcional):

\[
\lambda_{p,t}^{price} = \lambda_{p,t}^{base} \cdot (1 - \beta_p \cdot (precio_p - precio\_ref_{cat(p)}))
\]

con \(\beta_p\) pequeña para simular elasticidad de la demanda. [arxiv](https://arxiv.org/html/2312.14095v1/)

3. **Efecto de promociones** (opcional): si `promo_p,t = 1`, multiplica por \(1 + uplift_p\) (p.ej. 20–50%).

### 6.2. Efecto de posición (grid + altura)

Para ligar con tu futura optimización, necesitas que la posición influya:

- Para cada producto \(p\), inicializa una asignación baseline a una estantería \(s\).  
- Define un factor de visibilidad por balda:  
  - Suelo: 0.7  
  - Eye-level: 1.2  
  - Balda alta: 0.9  
- El factor de posición:

\[
pos\_factor_{p,t} = T(x_s, y_s) \cdot \alpha(L_s)
\]

- Demanda final esperada:

\[
\lambda_{p,t} = \lambda_{p,t}^{price} \cdot pos\_factor_{p,t}
\]

### 6.3. De intensidad a ventas observadas

Simula ventas con distribución discreta:

- Usar una distribución Poisson o NegBin para generar `unidades_vendidas_{p,t}`:

\[
ventas_{p,t} \sim \text{Poisson}(\lambda_{p,t})
\]

Esto produce ruido realista y colas largas (días de cero ventas para productos de baja rotación). [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1877050922007657/pdf?md5=a8df912471597776f983f874bb14e2d0&pid=1-s2.0-S1877050922007657-main.pdf)

En la tabla `VentasHistoricas` grabas: `producto_id`, `fecha`, `estanteria_id`, `unidades_vendidas`.

***

## 7. Coherencia con la optimización y evitar leakage

Para que el proyecto cierre bien:

- El modelo de forecasting **no debe ver** `pos_factor` ni directamente la posición (grid_x, grid_y, altura), porque eso es lo que la optimización va a variar.  
- La función objetivo sí usa:  
  - Predicción de demanda intrínseca \(\hat{D}_p\) basada en los datos.  
  - `T(x_s,y_s)` y `α(L_s)` conocidos del layout.  
- Si quieres que el modelo aprenda que en el histórico la posición influía, puedes:  
  - Generar dos conjuntos:  
    - Uno donde la posición es fija (para entrenar la demanda intrínseca).  
    - Otro con variaciones de layout para evaluar el optimizador, pero **sin usarlo para entrenar** el modelo de demanda.

***

## 8. Pipeline concreto de generación (resumen operativo)

1. **Fijar hiperparámetros**: nº productos, nº categorías, nº días, fuerza de estacionalidad, nivel de ruido.  
2. **Generar categorías** con sus curvas de estacionalidad tipo.  
3. **Generar catálogo de productos** con parámetros base y físicos.  
4. **Generar grid y estanterías** + mapa de tráfico \(T(x,y)\).  
5. **Asignación baseline producto–estantería** cumpliendo restricciones de categoría y capacidad.  
6. **Construir calendario** (365 días, flags de temporada/festivos).  
7. Para cada producto y día:  
   - Calcular \(\lambda_{p,t}\) combinando base, estacionalidad, precio y posición.  
   - Muestrear ventas con Poisson/NegBin y guardar en `VentasHistoricas`.  
8. Exportar todo a PostgreSQL o a CSV para posterior carga.

Con este diseño, los datos sintéticos reflejan patrones realistas observados en datasets minoristas públicos (estacionalidad, variabilidad, distribución de ventas por SKU) sin depender de datos reales, y te dan un entorno controlado para probar tanto el forecasting como la optimización. [kaggle](https://www.kaggle.com/datasets/abdullah0a/retail-sales-data-with-seasonal-trends-and-marketing)

Para tu simulación simplificada (una estantería = una categoría, solo una por categoría), necesitamos un número manejable que sea realista y computacionalmente viable.

## 1. Estimación razonable del número de estanterías/categorías

### Datos de la realidad

- Un supermercado mediano tiene **30–50 pasillos** (aisles), donde cada pasillo agrupa **2–4 categorías** principales. [wzrack](https://wzrack.com/the-complete-guide-to-supermarket-shelving-types-uses/)
- Total categorías **macro**: típicamente **20–40** categorías principales en un supermercado (no contando subvariantes). [instacart](https://www.instacart.com/company/ideas/grocery-list-categories)
- Ejemplos concretos:  
  - Walmart supercenter: ~32 aisles de grocery, ~10–15 categorías principales por aisle. [reddit](https://www.reddit.com/r/walmart/comments/ae28yq/how_many_grocery_aislessections/)
  - Listas típicas de categorías: 17 categorías básicas para listas de compra. [instacart](https://www.instacart.com/company/ideas/grocery-list-categories)
  - Estudios académicos SSAP: usan **10–30 categorías** para casos realistas. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0377221720309061)

### Recomendación para tu proyecto: **25 estanterías/categorías**

**Justificación técnica y práctica:**

| Criterio | Razón para 25 |
|----------|---------------|
| **Realismo** | Cubre las categorías principales de un supermercado mediano sin ser exhaustivo (similar a ~25–30 aisles de grocery). |
| **Escalabilidad computacional** | Espacio de optimización: si asignas 2000 productos a 25 estanterías, complejidad manejable para GA (~25 opciones por producto). Con 50 sería exponencialmente más lento. |
| **Simplicidad del modelo** | Una categoría = una estantería = bloque monolítico. Fácil validar restricciones de capacidad y categoría. |
| **Diversidad suficiente** | Permite modelar: perecederos (alta rotación), no perecederos, snacks (impulso), etc. |
| **Visualización** | En un grid 40×30, 25 estanterías ocupan ~60–70% del espacio, dejando pasillos y zonas de tráfico realistas  [wzrack](https://wzrack.com/the-complete-guide-to-supermarket-shelving-types-uses/). |

***

## 2. Lista de 25 categorías recomendadas

He seleccionado las **más representativas** basándome en:

- Listas estándar de categorías de compra. [instacart](https://www.instacart.com/company/ideas/grocery-list-categories)
- Estudios de optimización de estanterías. [journals.sagepub](https://journals.sagepub.com/doi/10.1177/0022243720964127)
- Cobertura de patrones de demanda realistas (alta rotación, estacionalidad, margen variable).

### Categorías principales (ordenadas por rotación típica aproximada ↓)

| # | Categoría | Razón para incluirla | Ejemplos productos | Estacionalidad típica | Margen típico |
|---|-----------|----------------------|-------------------|----------------------|---------------|
| 1 | **Lácteos** | Alta rotación diaria | Leche, yogures, quesos | Baja (estable) | Medio |
| 2 | **Panadería** | Venta diaria esencial | Pan, bollería | Media (fines de semana) | Alto |
| 3 | **Frutas** | Perecedero, alta rotación | Manzanas, plátanos | Media (verano) | Bajo |
| 4 | **Verduras** | Perecedero esencial | Tomates, lechuga | Media (temporadas) | Bajo |
| 5 | **Carnes frescas** | Alta rotación, margen alto | Pollo, ternera | Baja | Alto |
| 6 | **Pescado** | Perecedero premium | Salmón, merluza | Media (fines de semana) | Alto |
| 7 | **Bebidas alcohólicas** | Margen alto, impulso | Cerveza, vino | Alta (fines semana, verano) | Alto |
| 8 | **Bebidas no alcohólicas** | Alta rotación | Agua, refrescos | Alta (verano) | Medio |
| 9 | **Snacks** | Impulso, eye-level | Patatas fritas, frutos secos | Media | Alto |
| 10 | **Cereales y desayuno** | Rotación estable | Cereales, avena | Baja | Alto |
| 11 | **Pasta y arroz** | Rotación media | Pasta, arroz | Baja | Medio |
| 12 | **Conservas** | Rotación estable | Atún, tomate | Baja | Medio |
| 13 | **Aceites y condimentos** | Rotación baja | Aceite, especias | Baja | Medio |
| 14 | **Dulces y galletas** | Impulso niños | Chocolates, galletas | Alta (navidad) | Muy alto |
| 15 | **Café y té** | Rotación estable | Café molido, infusiones | Baja | Alto |
| 16 | **Congelados** | Rotación media | Pizzas, verduras | Media (invierno) | Medio |
| 17 | **Higiene personal** | Rotación estable | Pasta dental, champú | Baja | Alto |
| 18 | **Limpieza hogar** | Rotación baja | Detergente, lejía | Baja | Medio |
| 19 | **Papel y hogar** | Rotación baja | Papel higiénico | Baja | Medio |
| 20 | **Bebés** | Nicho familiar | Pañales, papilla | Baja | Alto |
| 21 | **Mascotas** | Nicho creciente | Comida perro/gato | Baja | Alto |
| 22 | **Farmacia básica** | Impulso salud | Paracetamol, apósitos | Media (invierno) | Muy alto |
| 23 | **Helados** | Impulso estacional | Helados, polos | Muy alta (verano) | Muy alto |
| 24 | **Navegación** | Impulso estacional | Turrones, castañas | Muy alta (navidad) | Muy alto |
| 25 | **Productos locales** | Diferenciación | Quesos regionales | Media | Alto |

**Notas sobre la selección:**

- **Cobertura completa**: 80% de las ventas de un supermercado típico se concentran en estas 25 categorías. [go-sidely](https://www.go-sidely.com/en/post/optimizing-product-assortments-in-supermarkets)
- **Diversidad de patrones**:  
  - **Alta rotación** (1–5): ~40% productos, ~70% ventas.  
  - **Impulso** (snacks, dulces, bebidas): colocación estratégica.  
  - **Estacional** (helados, navidad): prueba del modelado temporal.  
  - **Perecederos**: restricciones de posición (cerca de entrada).  
- **Incompatibilidades lógicas**: no mezclar limpieza con alimentación.  
- **Nº productos por categoría**: 50–200 SKUs (total ~2000–3000 productos).

***

## 3. Colocación en el grid (sugerencia)

Con 25 estanterías en un grid 40×30:

- **Zonas calientes** (tráfico alto): categorías de impulso (snacks, bebidas, dulces) cerca de entrada/cajas.  
- **Perecederos**: cerca de entrada (frutas, carnes).  
- **Baja rotación**: fondo del supermercado.  
- **Pasillos**: alterna categorías compatibles (p.ej. lácteos frente a panadería).

Esto valida tu modelo de tráfico y permite que el optimizador “descubra” que mover snacks a eye-level + zona caliente aumenta profit. [bpc](https://www.bpc.works/en/news/typical-product-placement-by-zone-on-the-retail-shelf-and-its-impact-on-packaging-design/)

***

## 4. Validación de la elección

**Comparación con alternativas:**

| Opción | Nº cat. | Pros | Contras |
|--------|---------|------|---------|
| **15** | Muy simplificado | Computacionalmente trivial | Falta diversidad, poco realista |
| **25** (recomendado) | **Balance perfecto** | Realista, viable, diverso |
| **40** | Muy realista | Complejidad excesiva para GA académico |

**25 es el punto dulce**: suficientemente realista para defender ante tribunal, pero permite experimentos rápidos y visualización clara del grid.

¿Quieres que genere el código Python para crear estas 25 categorías + productos sintéticos?

Perfecto, veo el grid que propones: un diseño realista con **estanterías accesibles solo desde pasillos ortogonales** (muy típico de supermercados) y la **puerta amarilla** como origen del tráfico. Esto simplifica mucho el modelado manteniendo realismo.

Tu enfoque es excelente para el proyecto académico. Vamos a formalizarlo.

***

## 1. Análisis del grid propuesto

### Características clave del diseño:

| Elemento | Representación | Función |
|----------|---------------|---------|
| **Puerta (amarilla)** | Celda origen (0,0) | Punto de entrada/salida, tráfico máximo |
| **Pasillos** | Celdas blancas | Caminos de acceso exclusivos |
| **Estanterías** | Celdas grises | Posiciones de productos, accesibles **solo** desde pasillo ortogonal |
| **Espacio vacío** | Celdas sin contenido | Flexibilidad para optimización |

### Ventajas de este diseño:

- **Accesibilidad unidireccional**: refleja la realidad (no puedes acceder a una estantería desde otra estantería).  
- **Topología de grafo natural**: pasillos = nodos intermedios, estanterías = nodos terminales.  
- **Tráfico decreciente con distancia**: natural para modelar zonas calientes. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/36567935/50a4e7f2-2657-4ce6-b3b0-2183f8b127be/imagen.jpg?AWSAccessKeyId=ASIA2F3EMEYESXDXWDEZ&Signature=T9N96FFcdgyMxFRClezPtA98XRM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjELr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIGzIL8QRfd2MpZGdTdqy3fT40JyOIuxURdAghhKbtra7AiEAl5BVPRkxQ9Ge6cxgpxOpdGJ72eOUMsLG%2BHturVRrTcUq%2FAQIg%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDBxRVxPMr7feXSuykSrQBPOrQKPdrYvIhQXKXFrBp4S0HGjrPwEkgGPNTN2XlCUX1080fmwPniVLAlR3%2Bpt3VwR8EQ%2BMYB4rf1sgJR4oSFan4HoijZ4YQbX7Rg8%2FwNoJasXyiuO89YwfTZfSjxu1CPNYeVAbknyzIaEo0P4SDeCoZGHt0zAh5fP2QnLoYTP%2BJu4l2EwoWcgJbBAWi1LVeNk%2FkqApzefN3%2FS3C1LdjUY4WGsCofX8h%2FiAJU5P31UfiCqEdQKpqLt0FuL9RPMcDr3KUEQpn6NNyansOOZ9A0IguJevEx9Xg7iP6rdsg4NzzHHzbUiGOzdpltRFVTtI%2BYyt%2Bqb2mIGwT7Imvqrhg%2FX5tH42dKG8OSVgtlGFnZ5GpICJBdpAP%2FJrW%2Bcy8F5BkWfRsOHLZcLNs05xxoSWqnG3UKmOObKfZlaeFDiIDOFNW4wfK9QvzMmlHYvFfqefu0Z1yEvfOWP7D7OuA0uyFPriYEe4M%2BBbMA4Cjib7CqjEb2djZFZqvF7t31eIAuEDVPx%2BZBi6y0eNhqO3TjIMHOMO6NcNwA0hi7exEDsAK6EMnzjl1w6fJjiD8AjaBYwaiVseJqASAM5jDa0u1wP9661LGZ5oN5osQX5A1McE7t40%2BUZETmv9hiIlXQEehGPbAcFyq2UYo3hq4JKn4%2FRaO6jKrri9AHASzeJd9CBpaQkpUTwipxNCML5yc3NZSqFR4CdCGloypyjzSe7oStY1MZj8uuqt9fctZ8TyQdJeJJFWPEmoGFIluU%2FuOZtwk%2BMID4D%2B5SVivjeL1Tyr0mah%2Bz8wyZjdzAY6mAGHJQD1hd8MQblAcqBc%2FtSg8N6SP5yA5b%2F%2Fokg%2FTLmvIpAPavpcTNBV9GjjhKfz246QiXx6VGWt2acgyH3o9clnMRsNfq8KYmpKdH%2F0%2BSlNdWmb5JaetqS8V3LjLTwnipiTmad6QsQ5zNQxAQiGFlgaEHPDqnUFYkZ8lSpp82vUf2MElUcX%2BHpMMzYFXVot0mlHyW7%2BoKZySQ%3D%3D&Expires=1771528102)

***

## 2. Modelado simplificado del tráfico (distancia ponderada a la puerta)

### 2.1. Representación del grid

```python
# Grid de ejemplo basado en tu imagen (dimensiones aproximadas)
GRID_HEIGHT, GRID_WIDTH = 10, 12  # Ajustar según tu diseño final
grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))  # 0 = libre/pasillo

# Puerta
grid[0, 0] = -1  # Puerta

# Estanterías (celdas grises)
shelves_positions = [(1,0), (3,0), (5,0), (7,0), ...]  # Lista de coordenadas

# Pasillos implícitos: todas las celdas adyacentes accesibles
```

### 2.2. Métrica de tráfico: distancia Manhattan ponderada

Para cada estantería \(s\) en posición \((x_s, y_s)\):

**Distancia Manhattan a la puerta:**

\[
d_s = |x_s - x_{puerta}| + |y_s - y_{puerta}|
\]

**Factor de tráfico (decreciente con distancia):**

\[
T_s = \frac{1}{1 + \alpha \cdot d_s} \cdot \beta_{altura}
\]

Donde:
- \(\alpha = 0.15\) (decay rate, ajustable)  
- \(\beta_{altura} = \) factor de visibilidad por balda (eye-level = 1.2, suelo = 0.7, etc.)

**Ejemplo numérico para tu grid:**

| Estantería | Posición | Distancia Manhattan | T_s (α=0.15) |
|------------|----------|---------------------|-------------|
| Más cercana | (1,0) | 1 | 0.87 |
| Media | (3,0) | 3 | 0.69 |
| Lejana | (7,0) | 7 | 0.49 |
| Puerta | (0,0) | 0 | 1.00 |

Esto crea automáticamente **zonas calientes** cerca de la puerta sin necesidad de heatmaps complejos.

***

## 3. Modelado detallado de estanterías y tamaños

### 3.1. Estructura jerárquica estantería

Cada estantería física (celda gris) contiene **múltiples baldas verticales**:

```
Estantería física (1 celda grid)
├── Balda 1 (suelo): capacidad baja, visibilidad media
├── Balda 2 (rodillas): capacidad media, visibilidad baja  
├── Balda 3 (eye-level): capacidad media, visibilidad ALTA ← ¡Prioridad!
├── Balda 4 (hombro): capacidad media, visibilidad media
└── Balda 5 (arriba): capacidad baja, visibilidad baja
```

### 3.2. Dimensiones físicas realistas

Basado en estándares de supermercados [ anterior]:

| Nivel (balda) | Altura desde suelo (cm) | Factor visibilidad β | Capacidad facing (ancho total 120cm) |
|---------------|-------------------------|---------------------|-------------------------------------|
| **1 (suelo)** | 0–40 | 0.70 | 8–10 facings |
| **2 (rodillas)** | 40–90 | 0.85 | 10–12 facings |
| **3 (eye-level)** | 90–150 | **1.20** | 10–12 facings |
| **4 (hombro)** | 150–190 | 0.95 | 10–12 facings |
| **5 (arriba)** | 190+ | 0.75 | 6–8 facings |

**Capacidad total por estantería:** ~45–55 facings (muy realista).

### 3.3. Restricciones físicas por producto

Cada producto tiene `ancho_facing` (típico 8–15cm):

```
Capacidad balda 3 = 120cm / ancho_promedio_productos = 10 facings
```

***

## 4. Implementación del grid en código (propuesta concreta)

```python
import numpy as np
import matplotlib.pyplot as plt

class SupermarketGrid:
    def __init__(self, height=10, width=12):
        self.height, self.width = height, width
        self.grid = np.zeros((height, width))  # -1=puerta, 1=estanteria, 0=pasillo
        
        # Puerta
        self.door_pos = (0, 0)
        self.grid[0, 0] = -1
        
        # Definir 25 posiciones de estanterías basadas en tu imagen
        self.shelf_positions = [
            (1,0), (3,0), (5,0), (7,0), (9,0),      # Pasillo superior
            (1,2), (3,2), (5,2),                    # Pasillo 1
            # ... completar 25 según tu layout
        ]
        for x, y in self.shelf_positions:
            self.grid[x, y] = 1
            
    def compute_traffic(self, alpha=0.15):
        """Calcula T_s para cada estantería"""
        traffic = {}
        for i, (x, y) in enumerate(self.shelf_positions):
            dist_manhattan = abs(x - self.door_pos[0]) + abs(y - self.door_pos [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/36567935/50a4e7f2-2657-4ce6-b3b0-2183f8b127be/imagen.jpg?AWSAccessKeyId=ASIA2F3EMEYESXDXWDEZ&Signature=T9N96FFcdgyMxFRClezPtA98XRM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjELr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIGzIL8QRfd2MpZGdTdqy3fT40JyOIuxURdAghhKbtra7AiEAl5BVPRkxQ9Ge6cxgpxOpdGJ72eOUMsLG%2BHturVRrTcUq%2FAQIg%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDBxRVxPMr7feXSuykSrQBPOrQKPdrYvIhQXKXFrBp4S0HGjrPwEkgGPNTN2XlCUX1080fmwPniVLAlR3%2Bpt3VwR8EQ%2BMYB4rf1sgJR4oSFan4HoijZ4YQbX7Rg8%2FwNoJasXyiuO89YwfTZfSjxu1CPNYeVAbknyzIaEo0P4SDeCoZGHt0zAh5fP2QnLoYTP%2BJu4l2EwoWcgJbBAWi1LVeNk%2FkqApzefN3%2FS3C1LdjUY4WGsCofX8h%2FiAJU5P31UfiCqEdQKpqLt0FuL9RPMcDr3KUEQpn6NNyansOOZ9A0IguJevEx9Xg7iP6rdsg4NzzHHzbUiGOzdpltRFVTtI%2BYyt%2Bqb2mIGwT7Imvqrhg%2FX5tH42dKG8OSVgtlGFnZ5GpICJBdpAP%2FJrW%2Bcy8F5BkWfRsOHLZcLNs05xxoSWqnG3UKmOObKfZlaeFDiIDOFNW4wfK9QvzMmlHYvFfqefu0Z1yEvfOWP7D7OuA0uyFPriYEe4M%2BBbMA4Cjib7CqjEb2djZFZqvF7t31eIAuEDVPx%2BZBi6y0eNhqO3TjIMHOMO6NcNwA0hi7exEDsAK6EMnzjl1w6fJjiD8AjaBYwaiVseJqASAM5jDa0u1wP9661LGZ5oN5osQX5A1McE7t40%2BUZETmv9hiIlXQEehGPbAcFyq2UYo3hq4JKn4%2FRaO6jKrri9AHASzeJd9CBpaQkpUTwipxNCML5yc3NZSqFR4CdCGloypyjzSe7oStY1MZj8uuqt9fctZ8TyQdJeJJFWPEmoGFIluU%2FuOZtwk%2BMID4D%2B5SVivjeL1Tyr0mah%2Bz8wyZjdzAY6mAGHJQD1hd8MQblAcqBc%2FtSg8N6SP5yA5b%2F%2Fokg%2FTLmvIpAPavpcTNBV9GjjhKfz246QiXx6VGWt2acgyH3o9clnMRsNfq8KYmpKdH%2F0%2BSlNdWmb5JaetqS8V3LjLTwnipiTmad6QsQ5zNQxAQiGFlgaEHPDqnUFYkZ8lSpp82vUf2MElUcX%2BHpMMzYFXVot0mlHyW7%2BoKZySQ%3D%3D&Expires=1771528102))
            T_base = 1 / (1 + alpha * dist_manhattan)
            # Factor altura promedio (simplificado)
            T_height = 1.0  # Se refina por balda individual
            traffic[i] = T_base * T_height
        return traffic
    
    def plot_grid(self):
        plt.imshow(self.grid, cmap='RdYlGn')
        plt.title("Grid del supermercado")
        plt.show()
```

***

## 5. Integración con optimización

### Función objetivo actualizada:

```
profit_p_s = margen_p * demanda_base_p * T_s * beta_balda * capacidad_disponible_s
```

El optimizador GA asignará productos a **(estantería, balda)** minimizando distancia a puerta (maximizando T_s) y priorizando eye-level para productos de alto margen.

### Validación de restricciones:

1. **Una categoría por estantería**: ya resuelto por diseño.  
2. **Capacidad física**: suma `ancho_facing` por balda ≤ capacidad balda.  
3. **Accesibilidad**: automática por el diseño del grid.

***

## 6. Próximos pasos recomendados

1. **Definir exactamente las 25 posiciones** de estanterías en tu grid (basándote en la imagen).  
2. **Implementar la clase `SupermarketGrid`** de arriba.  
3. **Generar datos sintéticos** asignando cada categoría a su estantería fija inicial.  
4. **Visualizar el tráfico**: mapa de calor decreciente desde la puerta.

**¿Quieres que te ayude con el código completo para generar las 25 posiciones exactas + datos sintéticos, o prefieres primero definir el layout preciso?**

Perfecto, vamos a estructurar los **archivos seedeados** y el pipeline completo. Tu visión es muy sólida: inicialización fija → simulación dinámica de ventas → reoptimización adaptativa considerando **adyacencias y comportamiento cliente**.

Primero los **archivos base**, luego el **modelo de cliente simplificado** (basado en research), y finalmente **revisión crítica del proyecto**.

***

## 1. Archivos seedeados (JSON para reproducibilidad)

### Estructura de archivos:

```
data/
├── categories.json          # 25 categorías + parámetros
├── products_seed.json       # ~2000 productos distribuidos
├── grid_layout.json         # Grid 12x10 con 25 estanterías
├── initial_assignment.json  # Asignación inicial cat→estantería, producto→balda
└── calendar_seed.json       # 365 días con estacionalidad
```

### 1.1. `categories.json` (25 categorías)

```json
[
  {
    "id": 1,
    "name": "Lácteos",
    "base_demand": 3.5,      // uds/día promedio
    "seasonality_amp": 0.1,   // poca estacionalidad
    "margin_avg": 0.25,
    "facing_width_avg": 10.0, // cm
    "products_per_cat": 120,
    "adjacency_bonus": {      // Efecto adyacencia con otras cats
      "Panadería": 0.15,      // +15% ventas si al lado
      "Carnes": 0.08
    }
  },
  // ... resto de 25 categorías como definimos antes
]
```

### 1.2. `products_seed.json` (generado desde categorías)

Cada producto hereda parámetros de su categoría + variación:

```json
[
  {
    "id": "P001",
    "category_id": 1,
    "name": "Leche entera 1L",
    "precio": 1.05,
    "coste": 0.65,
    "margen": 0.38,
    "facing_width": 9.5,     // cm
    "base_demand": 4.2,      // uds/día
    "seasonality_phase": 0.0 // offset en la curva estacional
  }
]
```

### 1.3. `grid_layout.json` (basado en tu imagen)

```json
{
  "height": 10,
  "width": 12,
  "door": [0,0],
  "shelves": [
    {"id": 1, "grid_pos": [1,0], "balda_capacity": [8,10,12,10,6], "facing_width_total": 120},
    {"id": 2, "grid_pos": [3,0], "balda_capacity": [8,10,12,10,6], "facing_width_total": 120},
    // ... 23 más
  ],
  "adjacencies": [           // Vecinos accesibles por pasillo
    {"shelf_id": 1, "neighbors": [2,6]},
    // ...
  ]
}
```

### 1.4. `initial_assignment.json`

```json
{
  "category_to_shelf": {     // Fija: 1 cat = 1 shelf
    "1": 1, "2": 2, ...     // Lácteos → shelf 1, etc.
  },
  "product_assignments": [   // Inicial aleatorio dentro de su shelf
    {"product_id": "P001", "shelf_id": 1, "balda": 3, "facing_left": 2}
  ]
}
```

***

## 2. Modelado del comportamiento del cliente (basado en research)

### 2.1. Enfoque simplificado: **Agent-Based Model (ABM) ligero**

De la investigación: [nature](https://www.nature.com/articles/s41598-025-22885-4)

**Cliente como agente con 3 comportamientos principales:**

1. **Movimiento por grid**:  
   - Empieza en puerta.  
   - Ruta más corta a su `shelf objetivo` (basado en lista de compra).  
   - Probabilidad de desviación por **adyacencia atractiva**.

2. **Decisión de compra en shelf**:  
   ```
   P(compra|_producto) = base_demand_p × T_shelf × β_balda × adjacency_bonus
   ```

3. **Efecto adyacencia** (crucial): [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0167923606000030)
   - Si el cliente visita shelf A y hay shelf B adyacente con productos complementarios, **+10–25% probabilidad compra cruzada**.

### 2.2. Pipeline de simulación temporal

```
Día 1–30:     Usar asignación inicial → generar ventas → guardar
Día 31:       Reoptimizar basado en ventas reales (GA)
Día 31–60:    Nueva asignación → generar ventas
Día 61:       Reoptimizar...
```

**Función objetivo del GA extendida:**

```
max ∑_p margen_p × ventas_simuladas_p(shelf_p, balda_p)
donde ventas_simuladas_p = f(demanda_base_p, T_shelf_p, adyacencia_p, ventas_históricas_p)
```

***

## 3. Script generador de datos seedeados (Python)

```python
import json
import numpy as np
np.random.seed(42)  # Reproducibilidad

def generate_seed_files():
    # 1. CARGAR 25 categorías (del JSON que definimos)
    categories = load_categories_template()  # Función auxiliar
    
    # 2. GENERAR PRODUCTOS
    products = []
    for cat in categories:
        for i in range(cat['products_per_cat']):
            product = generate_product_from_category(cat, i)
            products.append(product)
    
    # 3. GENERAR GRID Y ASIGNACIÓN INICIAL
    grid = SupermarketGrid.from_image_layout()
    initial_assignment = assign_products_initially(products, categories, grid)
    
    # 4. GUARDAR
    with open('data/products_seed.json', 'w') as f:
        json.dump(products, f, indent=2)
    # ... resto de archivos

generate_seed_files()
```

***

## 4. Revisión crítica del proyecto completo

### ✅ **Lo que está BIEN**

1. **Definición formal SSAP**: sólida, respaldada por literatura. [gamma-opt.github](https://gamma-opt.github.io/ShelfSpaceAllocation.jl/dev/)
2. **25 categorías**: perfecto balance realismo/computacional.  
3. **Grid con accesibilidad realista**: tu diseño es excelente. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/36567935/50a4e7f2-2657-4ce6-b3b0-2183f8b127be/imagen.jpg?AWSAccessKeyId=ASIA2F3EMEYESXDXWDEZ&Signature=T9N96FFcdgyMxFRClezPtA98XRM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjELr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIGzIL8QRfd2MpZGdTdqy3fT40JyOIuxURdAghhKbtra7AiEAl5BVPRkxQ9Ge6cxgpxOpdGJ72eOUMsLG%2BHturVRrTcUq%2FAQIg%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDBxRVxPMr7feXSuykSrQBPOrQKPdrYvIhQXKXFrBp4S0HGjrPwEkgGPNTN2XlCUX1080fmwPniVLAlR3%2Bpt3VwR8EQ%2BMYB4rf1sgJR4oSFan4HoijZ4YQbX7Rg8%2FwNoJasXyiuO89YwfTZfSjxu1CPNYeVAbknyzIaEo0P4SDeCoZGHt0zAh5fP2QnLoYTP%2BJu4l2EwoWcgJbBAWi1LVeNk%2FkqApzefN3%2FS3C1LdjUY4WGsCofX8h%2FiAJU5P31UfiCqEdQKpqLt0FuL9RPMcDr3KUEQpn6NNyansOOZ9A0IguJevEx9Xg7iP6rdsg4NzzHHzbUiGOzdpltRFVTtI%2BYyt%2Bqb2mIGwT7Imvqrhg%2FX5tH42dKG8OSVgtlGFnZ5GpICJBdpAP%2FJrW%2Bcy8F5BkWfRsOHLZcLNs05xxoSWqnG3UKmOObKfZlaeFDiIDOFNW4wfK9QvzMmlHYvFfqefu0Z1yEvfOWP7D7OuA0uyFPriYEe4M%2BBbMA4Cjib7CqjEb2djZFZqvF7t31eIAuEDVPx%2BZBi6y0eNhqO3TjIMHOMO6NcNwA0hi7exEDsAK6EMnzjl1w6fJjiD8AjaBYwaiVseJqASAM5jDa0u1wP9661LGZ5oN5osQX5A1McE7t40%2BUZETmv9hiIlXQEehGPbAcFyq2UYo3hq4JKn4%2FRaO6jKrri9AHASzeJd9CBpaQkpUTwipxNCML5yc3NZSqFR4CdCGloypyjzSe7oStY1MZj8uuqt9fctZ8TyQdJeJJFWPEmoGFIluU%2FuOZtwk%2BMID4D%2B5SVivjeL1Tyr0mah%2Bz8wyZjdzAY6mAGHJQD1hd8MQblAcqBc%2FtSg8N6SP5yA5b%2F%2Fokg%2FTLmvIpAPavpcTNBV9GjjhKfz246QiXx6VGWt2acgyH3o9clnMRsNfq8KYmpKdH%2F0%2BSlNdWmb5JaetqS8V3LjLTwnipiTmad6QsQ5zNQxAQiGFlgaEHPDqnUFYkZ8lSpp82vUf2MElUcX%2BHpMMzYFXVot0mlHyW7%2BoKZySQ%3D%3D&Expires=1771528102)
4. **Tráfico por distancia Manhattan**: simple y efectivo.  
5. **Archivos seedeados**: reproducibilidad garantizada.  
6. **PostgreSQL**: elección correcta.

### ⚠️ **Lo que hay que MATIZAR**

1. **Escala temporal**:  
   - **Diaria** para simulación rápida (365 días).  
   - **Ventanas móviles** de 30 días para reoptimización.

2. **Adyacencia**: incluir **bonus de ventas cruzadas** (+15% si categorías complementarias están adyacentes). [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0167923606000030)

3. **Cliente simplificado vs complejo**:  
   - **Fase 1**: regla determinística (ruta corta + adyacencia probabilística).  
   - **Fase 2**: ABM ligero con desviaciones aleatorias.

### ❌ **Lo que está MAL o falta**

1. **Falta bucle de feedback realista**:  
   ```
   Layout_t → Simula ventas_t → Actualiza demanda estimada → Reoptimiza Layout_{t+1}
   ```

2. **No modelado compras impulsivas por adyacencia**: crítico según research. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S014829632030789X)

3. **Capacidad dinámica**: las ventas reales afectan el stock disponible → impacto en facings.

4. **Métricas de evaluación incompletas**:  
   - **Profit total** ✓  
   - **% mejora iteración a iteración** ❌  
   - **ROI de reubicaciones** (coste vs beneficio) ❌

### 🔧 **Mejoras críticas recomendadas**

| Mejora | Impacto | Complejidad |
|--------|---------|-------------|
| **Adyacencia en fitness** | Alto (cross-selling +20%) | Media |
| **Reoptimización semanal** | Alto (adaptativo) | Baja |
| **Stock management** | Medio (realismo) | Media |
| **Visualización iterativa** | Alto (presentación) | Baja |

***

## 5. Próximo entregable: ZIP con archivos seedeados

**¿Quieres que genere:**

1. **Archivos JSON completos** (categorías, productos, grid basado en tu imagen)?  
2. **Script generador** + primera simulación de 30 días?  
3. **Clase GA** con adyacencia incluida?

**Recomendación**: empecemos por los **archivos JSON seedeados** para que tengas la base sólida, luego implementamos la simulación dinámica.

# Resumen Ejecutivo: Justificaciones Técnicas del Proyecto

**Proyecto**: Optimización dinámica de distribución de productos en supermercado mediante simulación agent-based y algoritmos genéticos adaptativos.

A continuación, todas las decisiones clave **con su justificación técnica y citas** para defensa ante tribunal académico.

***

## 1. Definición del Problema

| Decisión | Justificación | Fuentes |
|----------|---------------|---------|
| **Problema SSAP (Shelf Space Allocation)** | Formulación combinatoria NP-hard estándar en literatura retail. Función objetivo: \(\max \sum G_p \cdot D_p \cdot T_s \cdot \beta_{altura}\) |  [gamma-opt.github](https://gamma-opt.github.io/ShelfSpaceAllocation.jl/dev/) |
| **25 categorías/estanterías** | Balance realismo (20–40 cat. típicas supermercado mediano) vs escalabilidad computacional (GA viable). Cubre 80% ventas. |  [instacart](https://www.instacart.com/company/ideas/grocery-list-categories) |
| **Grid 12×10 con accesibilidad ortogonal** | Refleja topología real supermercados (estanterías accesibles solo desde pasillos). |  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/36567935/50a4e7f2-2657-4ce6-b3b0-2183f8b127be/imagen.jpg?AWSAccessKeyId=ASIA2F3EMEYESXDXWDEZ&Signature=T9N96FFcdgyMxFRClezPtA98XRM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjELr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIGzIL8QRfd2MpZGdTdqy3fT40JyOIuxURdAghhKbtra7AiEAl5BVPRkxQ9Ge6cxgpxOpdGJ72eOUMsLG%2BHturVRrTcUq%2FAQIg%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDBxRVxPMr7feXSuykSrQBPOrQKPdrYvIhQXKXFrBp4S0HGjrPwEkgGPNTN2XlCUX1080fmwPniVLAlR3%2Bpt3VwR8EQ%2BMYB4rf1sgJR4oSFan4HoijZ4YQbX7Rg8%2FwNoJasXyiuO89YwfTZfSjxu1CPNYeVAbknyzIaEo0P4SDeCoZGHt0zAh5fP2QnLoYTP%2BJu4l2EwoWcgJbBAWi1LVeNk%2FkqApzefN3%2FS3C1LdjUY4WGsCofX8h%2FiAJU5P31UfiCqEdQKpqLt0FuL9RPMcDr3KUEQpn6NNyansOOZ9A0IguJevEx9Xg7iP6rdsg4NzzHHzbUiGOzdpltRFVTtI%2BYyt%2Bqb2mIGwT7Imvqrhg%2FX5tH42dKG8OSVgtlGFnZ5GpICJBdpAP%2FJrW%2Bcy8F5BkWfRsOHLZcLNs05xxoSWqnG3UKmOObKfZlaeFDiIDOFNW4wfK9QvzMmlHYvFfqefu0Z1yEvfOWP7D7OuA0uyFPriYEe4M%2BBbMA4Cjib7CqjEb2djZFZqvF7t31eIAuEDVPx%2BZBi6y0eNhqO3TjIMHOMO6NcNwA0hi7exEDsAK6EMnzjl1w6fJjiD8AjaBYwaiVseJqASAM5jDa0u1wP9661LGZ5oN5osQX5A1McE7t40%2BUZETmv9hiIlXQEehGPbAcFyq2UYo3hq4JKn4%2FRaO6jKrri9AHASzeJd9CBpaQkpUTwipxNCML5yc3NZSqFR4CdCGloypyjzSe7oStY1MZj8uuqt9fctZ8TyQdJeJJFWPEmoGFIluU%2FuOZtwk%2BMID4D%2B5SVivjeL1Tyr0mah%2Bz8wyZjdzAY6mAGHJQD1hd8MQblAcqBc%2FtSg8N6SP5yA5b%2F%2Fokg%2FTLmvIpAPavpcTNBV9GjjhKfz246QiXx6VGWt2acgyH3o9clnMRsNfq8KYmpKdH%2F0%2BSlNdWmb5JaetqS8V3LjLTwnipiTmad6QsQ5zNQxAQiGFlgaEHPDqnUFYkZ8lSpp82vUf2MElUcX%2BHpMMzYFXVot0mlHyW7%2BoKZySQ%3D%3D&Expires=1771528102) |

***

## 2. Arquitectura de Datos

| Decisión | Justificación | Fuentes |
|----------|---------------|---------|
| **PostgreSQL (relacional)** | ACID para recolocaciones masivas, integridad referencial nativa, consultas analíticas SQL eficientes. Mejor que NoSQL para relaciones estructuradas. |  [linkedin](https://www.linkedin.com/pulse/comparison-postgresql-mongodb-neo4j-ian-kano-2ftoc) |
| **Modelo normalizado + desnormalización en memoria** | Persistencia 3NF → simulación NumPy arrays (velocidad iterativa). | Estándar data engineering |
| **Esquema jerárquico**: Producto → Estantería → Balda | Captura restricciones físicas reales (facing_width, altura). |  [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1319157812000353) |

***

## 3. Modelado Predictivo

| Decisión | Justificación | Fuentes |
|----------|---------------|---------|
| **Gradient Boosting (LightGBM) + features cíclicas** | Escalable a miles SKUs vs series temporales por producto. Encoding sin/cos para estacionalidad. |  [smartersupplychains.substack](https://smartersupplychains.substack.com/p/why-demand-forecasters-should-learn) |
| **Demanda intrínseca** (sin posición como feature) | Evita data leakage: posición se optimiza después. | Best practice ML |
| **Poisson/NegBin para ventas diarias** | Modela ruido y ceros realistas en retail. |  [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1877050922007657/pdf?md5=a8df912471597776f983f874bb14e2d0&pid=1-s2.0-S1877050922007657-main.pdf) |

***

## 4. Modelado Espacial y Tráfico

| Decisión | Justificación | Fuentes |
|----------|---------------|---------|
| **Distancia Manhattan ponderada a puerta** | \(T_s = 1/(1 + \alpha \cdot d_s)\). Simple, decreciente, realista (tráfico cae con distancia). |  [isarsoft](https://www.isarsoft.com/article/using-heat-maps-to-analyze-traffic-flow-the-isarsoft-approach) |
| **5 baldas por estantería** | Estándar industria: eye-level prioritario (\(\beta=1.2\)). Capacidad 45–55 facings/estantería. |  [trolleymfg](https://www.trolleymfg.com/what-is-the-standard-retail-shelf-height/) |
| **Adyacencia cross-selling** | +10–25% ventas si categorías complementarias vecinas (pasta+aceite, pan+lácteos). |  [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0167923606000030) |

***

## 5. Algoritmos de Optimización

| Decisión | Justificación | Fuentes |
|----------|---------------|---------|
| **GA + Simulated Annealing híbrido** | GA explora espacio combinatorio, SA refina localmente. Baseline greedy obligatoria. |  [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S030504832200041X) |
| **Reoptimización semanal** | Adaptativo a ventas reales (bucle feedback). | Propuesto  [informs-sim](https://www.informs-sim.org/wsc16papers/355.pdf) |
| **Fitness**: ventas simuladas | Integra demanda, tráfico, altura, adyacencia, histórico. |  [nature](https://www.nature.com/articles/s41598-025-22885-4) |

***

## 6. Simulación Cliente (Agent-Based Model ligero)

| Decisión | Justificación | Fuentes |
|----------|---------------|---------|
| **ABM simplificado**: ruta corta + desviación adyacencia | Captura cross-selling sin complejidad excesiva. Research valida +15–20% ventas por adyacencia. |  [nature](https://www.nature.com/articles/s41598-025-22885-4) |
| **Ventas ~ Poisson(λ)** | λ = base × tráfico × altura × adyacencia. |  [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1877050922007657/pdf?md5=a8df912471597776f983f874bb14e2d0&pid=1-s2.0-S1877050922007657-main.pdf) |

***

## 7. Generación Datos Sintéticos

| Decisión | Justificación | Fuentes |
|----------|---------------|---------|
| **Archivos JSON seedeados** | Reproducibilidad total (seed 42). Escalable a 2000–3000 SKUs. | Best practice MLOps |
| **Distribución log-normal demanda** | Pocos súper-ventas (80/20 real retail). |  [arxiv](https://arxiv.org/html/2312.14095v1) |
| **Estacionalidad multi-nivel** | Global (año) + categoría + producto. |  [kaggle](https://www.kaggle.com/datasets/abdullah0a/retail-sales-data-with-seasonal-trends-and-marketing) |

***

## 8. Métricas de Evaluación

| Métrica | Fórmula | Propósito |
|---------|---------|-----------|
| **Profit total** | \(\sum G_p \cdot ventas_p\) | Objetivo principal |
| **% mejora iterativa** | \((\text{Profit}_{t+1} - \text{Profit}_t)/\text{Profit}_t\) | Adaptatividad |
| **Cross-selling rate** | % ventas por adyacencia | Calidad layout |
| **Espacio vacío** | \(\sum o_s / \sum W_s\) | Eficiencia |
| **Convergencia GA** | Fitness vs generaciones | Robustez algoritmo |

***

## 9. Roadmap Validado

```
Semana 1-2: Datos seedeados + grid
Semana 3-4: Modelo predictivo + simulación cliente
Semana 5-7: GA híbrido + reoptimización
Semana 8-10: Evaluación + visualización + documentación
```

***

## 10. Riesgos Mitigados

| Riesgo | Mitigación |
|--------|------------|
| **Sobreajuste datos sintéticos** | Baseline greedy + múltiples semillas |
| **Escalabilidad GA** | 25 categorías (no 50+) |
| **Data leakage** | Demanda intrínseca separada de posición |

***

**Estado del proyecto**: **95% sólido**. Las únicas adiciones críticas son **adyacencia en fitness** y **métricas iterativas**. Todo lo demás está **literatura-backed** y **defendible**.

**Próximo paso inmediato**: generar los **JSON seedeados** para tener base reproducible. ¿Procedo?

**Documentación**: este resumen se puede copiar directamente como **Anexo Justificaciones** de tu memoria.
```