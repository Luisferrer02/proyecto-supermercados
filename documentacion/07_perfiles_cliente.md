# Perfiles de cliente: por qué quedan fuera del alcance

**Hallazgo auditoría**: F-04 ("Sin perfiles de cliente") — PERSISTE en todas las auditorías.

## 1. Qué pedía la crítica

Los auditores señalan que el modelo optimiza estanterías asumiendo un **cliente medio homogéneo**. No distingue entre:

- Clientes con poder adquisitivo alto vs. bajo.
- Segmentos etarios (jóvenes, mayores).
- Horarios (early-birds, compradores de tarde).
- Tickets grandes vs. pequeños.
- Patrones geográficos (urbano vs. rural).

Un supermercado real segmenta su layout por perfil (p.ej. los productos premium cerca de la entrada en zonas ABC1, las marcas blancas en zonas populares).

## 2. Por qué no lo abordamos

### 2.1 Los datos no lo permiten

**Dataset base**: [`products_macro.csv`](../mlops/products_macro.csv) (Mercadona, Kaggle) contiene solo:
- Nombre del producto, subtítulo, categoría
- Precio y precio con descuento
- Marca

**No contiene** ni una sola variable del lado del **cliente**:
- No hay tickets de compra.
- No hay IDs de cliente.
- No hay fechas/horas de transacción.
- No hay datos demográficos.
- No hay identificadores de tienda.

Añadir perfiles de cliente requeriría un dataset completamente distinto: **datos de panel de consumidores** (p.ej. tipo Kantar, Nielsen, o los datos internos de tarjetas de fidelización). Estos no son públicos ni gratuitos.

### 2.2 Los CSVs generados tampoco lo tienen

Los CSVs mensuales sintéticos (`sales_YYYY_MM_monthname.csv`, generados por `01_generate_monthly_sales.py`) extienden el dataset base con:
- `estimated_monthly_sales`, `profit_margin_percentage`, `product_width_cm`, `rack_id`, `shelf_level`.

Todas estas son propiedades del **producto**, no del cliente. No hay forma de derivar perfiles de cliente a partir de estos CSVs porque el proceso generativo (`retail_physics.py`) no los modela.

### 2.3 El enunciado del proyecto no lo pedía

El alcance acordado fue: *"generar datos mensuales realistas y optimizar el layout de estanterías maximizando el beneficio total del supermercado"*. Segmentación por cliente habría sido un proyecto distinto — más cercano a marketing o CRM que a optimización logística.

## 3. Cómo sería la extensión si tuviéramos los datos

Para dejar constancia de que el **diseño es consciente**, aquí la extensión natural:

### 3.1 Modelo de dos niveles

```
                ┌─────────────────┐
                │ Segmentación    │   K-means / LatentDirichlet sobre
                │ de clientes     │   histórico de tickets → K clusters
                └─────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ profit(cluster, │   Retrainemos el ensemble MLP+Tr
                │ shelf_layout)   │   condicionado al cluster
                └─────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Optimización    │   Maximizar profit esperado
                │ ponderada       │   ponderado por Pr(cluster | tienda)
                └─────────────────┘
```

### 3.2 Features adicionales necesarias

- `shelf_multiplier` pasaría a ser `shelf_multiplier[cluster]` (array, no escalar).
- Input features incluirían `store_id`, `avg_ticket_store`, `weekday`, `hour_band`.
- El target seguiría siendo `profit_lift` pero calculado por cluster y agregado con los pesos de presencia de cada cluster en esa tienda.

### 3.3 Datos que necesitaríamos

- Transacciones a nivel ticket (mínimo 6 meses, varias tiendas).
- Metadatos de tienda (ubicación, radio socioeconómico).
- Opcional: datos de tarjetas de fidelización (permite ir del ticket anónimo al cliente).

## 4. Resumen ejecutivo para defensa

> **¿Por qué no modelan perfiles de cliente?**
>
> Porque los datos disponibles no lo permiten: ni el catálogo base de Mercadona (Kaggle) ni los CSVs mensuales generados contienen información del lado del cliente (sin tickets, sin IDs, sin demografía). Añadir perfiles requeriría un dataset de panel de consumidores que no tenemos acceso. El diseño es consciente: el modelo optimiza para el "cliente medio" de cada categoría, que es la unidad de decisión real cuando se planifica un planograma de supermercado. La extensión a segmentación por cluster es la evolución natural del sistema y está esbozada en la documentación, pero queda fuera del scope del proyecto por falta de datos, no por desconocimiento.

## 5. Lo que sí captura el modelo actual

Aunque no hay perfiles de cliente explícitos, el pipeline **incorpora indirectamente** variabilidad de demanda:

- **Multiplicadores estacionales** (`01_generate_monthly_sales.py`) modelan cambios macro del comportamiento agregado del cliente (helados en julio, turrón en diciembre).
- **El forecast RAG+LLM** (`05_predict.py`) ajusta las ventas esperadas por categoría para el mes objetivo, capturando tendencias de consumo.
- **La física de estantería** (`retail_physics.py`) asume que el efecto shelf-level es universal (eye-level vende más), que es empíricamente robusto entre segmentos.

Esto no es segmentación pero sí reconoce que la demanda **no es estática**.
