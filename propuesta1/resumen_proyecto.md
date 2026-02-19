# Especificación Técnica: Optimización de Distribución en Supermercado

**Versión:** 1.0  
**Fecha:** 19 de Febrero de 2026  
**Objetivo:** Definición técnica, arquitectura y roadmap para el desarrollo del simulador y optimizador de layout.

---

## 1. Resumen Ejecutivo
El proyecto tiene como objetivo desarrollar un sistema inteligente para optimizar la distribución de productos en un supermercado. El sistema utiliza **simulación basada en agentes (ABM)** para modelar el comportamiento del cliente y **algoritmos genéticos híbridos** para encontrar la disposición óptima de categorías y productos que maximice el beneficio total (Profit), respetando restricciones físicas y de negocio.

El enfoque combina una arquitectura de datos robusta (PostgreSQL) con un motor de simulación ligero en Python, validado mediante datos sintéticos realistas.

---

## 2. Definición Formal del Problema

El problema se modela como un **Shelf Space Allocation Problem (SSAP)**, una variante de optimización combinatoria clasificada como *NP-hard*.

### 2.1 Variables y Conjuntos
*   **Grid ($G$):** Matriz 2D que representa el suelo del supermercado. Celdas pueden ser `PASILLO`, `ESTANTERÍA` o `PUERTA`.
*   **Productos ($P$):** Conjunto de SKUs con atributos físicos (dimensiones), comerciales (precio, coste) y patrones de demanda.
*   **Estanterías ($S$):** Recursos finitos con capacidad espacial (ancho, alto, profundidad) y ubicación específica en $G$.
*   **Categorías ($C$):** Agrupaciones lógicas. En este modelo simplificado, **1 Estantería = 1 Categoría**.

### 2.2 Función Objetivo
Maximizar el Beneficio Total ($Z$) esperado:

$$ Z = \sum_{p \in P} \sum_{s \in S} (Margen_p 	imes VentasSimuladas_{p,s}) - CostesRecolocacion $$

Donde $VentasSimuladas$ depende de:
1.  **Demanda Base:** Potencial de venta intrínseco del producto.
2.  **Factor Tráfico ($T_s$):** Flujo de clientes en la posición $s$ (función de distancia a puerta y zonas calientes).
3.  **Visibilidad Vertical ($eta_l$):** Factor de altura de la balda (e.g., *eye-level* > *suelo*).
4.  **Factor Adyacencia ($A_{cat}$):** Bonus de venta cruzada si categorías complementarias son vecinas.

### 2.3 Restricciones Hard
1.  **Unicidad:** Un producto solo puede estar en una ubicación física a la vez.
2.  **Capacidad Física:** La suma del ancho de los facings no puede exceder el ancho de la balda. $\sum w_p \cdot n_p \le W_{balda}$.
3.  **Coherencia de Categoría:** Un producto de la categoría $C_i$ solo puede asignarse a una estantería etiquetada como $C_i$.
4.  **Accesibilidad:** Las estanterías solo son accesibles desde celdas de pasillo ortogonales.

---

## 3. Arquitectura del Sistema

Se propone una arquitectura de 4 capas para garantizar escalabilidad y rigor académico.

### 3.1 Capa de Datos (Persistencia)
*   **Tecnología:** **PostgreSQL**.
*   **Justificación:**
    *   **Integridad Referencial:** Crítica para evitar "productos huérfanos" durante recolocaciones masivas (ACID).
    *   **Consultas Analíticas:** Window functions para análisis de series temporales y comparativas por categoría.
    *   **Flexibilidad:** Uso de `JSONB` para atributos variables de simulación (configuraciones, logs de agentes).
*   **Esquema Normalizado (3NF):**
    *   `Producto` (id, dimensiones, atributos financieros).
    *   `Estanteria` (id, posición_grid, capacidad).
    *   `Asignacion` (producto_id, estanteria_id, balda, num_facings).
    *   `VentasHistoricas` (producto_id, fecha, cantidad, precio_venta).

### 3.2 Capa de Simulación (Python Core)
*   **Motor de Grid:** Representación matricial (NumPy) del layout.
*   **Agentes (Clientes):** Modelo simplificado (Lightweight ABM).
    *   *Comportamiento:* Entran por `(0,0)`, calculan ruta a categorías objetivo (lista de compra) y tienen probabilidad de desviación hacia estanterías adyacentes atractivas.
    *   *Decisión de Compra:* Probabilística basada en precio y visibilidad.

### 3.3 Capa de Optimización
*   **Algoritmo Híbrido:**
    1.  **Greedy Constructivo:** Generación de solución inicial (ordenar por margen decreciente en mejores posiciones) para "seedear" la población.
    2.  **Algoritmo Genético (GA):** Exploración global. Cromosomas representan la permutación de asignaciones. Operadores de cruce y mutación adaptados para mantener restricciones de grupo.
    3.  **Simulated Annealing (SA):** Refinamiento local de las mejores soluciones del GA para escapar de óptimos locales.
*   **Ciclo de Re-optimización:** Ejecución periódica (e.g., cada 30 días simulados) para adaptar el layout a cambios estacionales.

### 3.4 Capa de Análisis (Fase 2)
*   **Integración LLM:** Generación automática de informes en lenguaje natural explicando *por qué* se movió un producto (e.g., "Se movió Cervezas cerca de Snacks para aprovechar el impulso del fin de semana").

---

## 4. Estrategia de Datos Sintéticos (Seeding)

Para validar el sistema sin datos propietarios, se generan datasets procedimentales reproducibles.

### 4.1 Entidades Generadas
1.  **Layout (Grid):**
    *   Dimensiones ~12x10 (escalable).
    *   **25 Categorías** fijas (una por estantería).
    *   Puerta en `(0,0)` (esquina superior izquierda).
    *   Tráfico modelado como decaimiento por distancia Manhattan: $T(x,y) \propto rac{1}{1 + lpha \cdot d((x,y), puerta)}$.

2.  **Catálogo de Productos:**
    *   ~2000 SKUs distribuidos en las 25 categorías.
    *   **Precios/Márgenes:** Distribuciones Log-Normal (cola larga de precios).
    *   **Dimensiones:** Muestreadas de estándares reales (botellas, cajas, packs).

3.  **Patrones Temporales:**
    *   **Estacionalidad:** Funciones sinusoidales (`sin`/`cos` anual) + offsets por categoría (e.g., Helados pico verano, Turrones pico invierno).
    *   **Demanda Diaria:** Distribución Poisson/Binomial Negativa para simular variabilidad y "ceros" realistas.

### 4.2 Archivos de Seeding (JSON)
*   `categories.json`: Definición de las 25 categorías y matriz de adyacencias (bonus de cross-selling).
*   `products_seed.json`: Inventario inicial.
*   `grid_layout.json`: Topología del mapa.
*   `calendar_seed.json`: Factores climáticos/festivos diarios.

---

## 5. Algoritmos y Lógica de Negocio

### 5.1 Predicción de Demanda
*   **Enfoque:** Regresión (Gradient Boosting - LightGBM/XGBoost).
*   **Features:** Ventas históricas (lags), estacionalidad (seno/coseno del día del año), precio relativo.
*   **Prevención de Leakage:** El modelo predice la **demanda intrínseca** (potencial de venta). El efecto de la posición (balda/pasillo) se aplica *a posteriori* en la simulación, no como feature de entrada, para evitar que el modelo "aprenda" que una mala posición implica mala demanda inherente.

### 5.2 Lógica de Simulación
El cálculo de ventas diarias simuladas sigue el flujo:
1.  **Demanda Base:** $D_{base} = f(ModeloPredictivo)$.
2.  **Modificadores:**
    *   $M_{pos} = T_{grid} 	imes eta_{altura}$
    *   $M_{ady} = 1 + \sum (Bonus_{vecinos})$
3.  **Venta Final:** $Venta = \min(Stock, Poisson(D_{base} 	imes M_{pos} 	imes M_{ady}))$

---

## 6. Validación y Métricas

### 6.1 Validación Técnica
*   **Integridad:** Checks SQL para asegurar que `sum(facings) <= ancho_balda`.
*   **Coherencia:** Verificación de que el grafo de accesibilidad no deja "islas" inaccesibles.

### 6.2 KPIs de Negocio
*   **Profit Total Acumulado.**
*   **Rotación de Stock.**
*   **Ocupación de Lineal:** % de espacio físico utilizado vs disponible.
*   **Impacto de Cross-Selling:** % de ventas atribuidas a efectos de adyacencia.

---

## 7. Roadmap de Implementación

1.  **Fase 1: Fundamentos (Semanas 1-2)**
    *   Diseño del esquema BD PostgreSQL.
    *   Generación de Scripts de Seeding (JSONs).
    *   Implementación de clase `Grid` y visualización básica.

2.  **Fase 2: Motor de Simulación (Semanas 3-4)**
    *   Implementación de lógica de agentes y cálculo de tráfico.
    *   Validación de curvas de demanda sintética.

3.  **Fase 3: Optimizador (Semanas 5-7)**
    *   Desarrollo del Algoritmo Genético.
    *   Integración del bucle: Simular -> Optimizar -> Re-Simular.

4.  **Fase 4: Análisis y Cierre (Semanas 8-10)**
    *   Benchmarking (GA vs Greedy vs Aleatorio).
    *   Documentación final y validación académica.
