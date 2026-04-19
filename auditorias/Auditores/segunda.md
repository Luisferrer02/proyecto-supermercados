Informe de Auditoría Técnica: Proyecto Stockit – Simulación y Optimización de Retail

1. Introducción y Alcance de la Auditoría

El proyecto Stockit ha sido auditado con el fin de evaluar su viabilidad técnica y estratégica como solución para la simulación y optimización de la disposición física (layout) en supermercados. En el contexto actual de retail, la capacidad de modelar el comportamiento del consumidor para maximizar el margen operativo no es solo una ventaja competitiva, sino un imperativo de negocio. Stockit se posiciona como una herramienta que busca cerrar la brecha entre la logística tradicional y el análisis predictivo.

El objetivo central del proyecto es la generación de simulaciones estocásticas que, partiendo de una combinación de datos reales y sintéticos, permitan identificar la disposición de estanterías más rentable. Esta auditoría determina que el éxito de la solución no depende únicamente del algoritmo de optimización, sino de la robustez de su arquitectura de datos y de la fidelidad del modelado del comportamiento humano.

2. Arquitectura de Datos: Evaluación de ArangoDB vs. Soluciones Híbridas

Para sistemas de alta complejidad donde la relación entre entidades (grafos) es tan crítica como los atributos de los objetos (documentos), la arquitectura de datos define la escalabilidad del producto. La propuesta inicial, basada en una fragmentación de servicios, presentaba riesgos operativos inaceptables para un entorno de producción.

Evaluación de Diferenciadores Técnicos

A continuación, se detalla la comparativa entre la arquitectura descartada y la implementación actual:

Característica	Solución Inicial (MongoDB + Neo4j + ETLs)	Solución Adoptada (ArangoDB)
Paradigma de Datos	Documentos y Grafos en silos separados.	Multi-modelo nativo (Documentos + Grafos).
Complejidad de Integración	Alta (Requiere procesos ETL y/o PuPyGraph).	Mínima (Unificación en un solo motor).
Rendimiento (Latencia)	Elevada por transferencia entre servicios.	Optimizada; consultas AQL en memoria compartida.
DevOps Overhead	Crítico; gestión de 3+ contenedores inestables.	Bajo; un solo servicio simplificado.
Curva de Aprendizaje	Requiere dominio de múltiples lenguajes.	AQL (Similar a SQL, facilitando la transición).

Justificación Técnica

La transición a ArangoDB fue un movimiento estratégico acertado. El equipo técnico identificó que un entorno basado en múltiples contenedores ("Docker Compose") presentaba fallos de orquestación recurrentes. Al unificar la persistencia en ArangoDB, se eliminan las latencias de las capas ETL y se simplifica el despliegue. El uso de AQL (ArangoDB Query Language) permite realizar consultas complejas sobre la estructura del supermercado y el inventario de productos con una sintaxis familiar, reduciendo los tiempos de desarrollo.

3. Modelado de Grafos y Logística de Distribución

La teoría de grafos es el núcleo que permite transformar un plano estático en un entorno de decisión dinámico. Sin embargo, en su estado actual, el modelo se asemeja más a una "tienda de conveniencia" que a un supermercado a gran escala debido a las limitaciones de espacio en el prototipo.

Desglose del Modelo

* Vértices: Representan estanterías categorizadas (frío, seco, congelado) y productos.
* Aristas: Modelan la conectividad física y la distancia euclidiana entre nodos.

Análisis Estratégico de Conectividad

El sistema implementa una lógica de conexiones forzadas. Para acceder a nodos de alta necesidad (ej. lácteos), el grafo obliga al cliente a transitar por pasillos secundarios. Este diseño busca explotar el potencial de compra impulsiva, un pilar del retail moderno. No obstante, se observa que la topología actual es limitada; el escalado del modelo requerirá una representación más densa para evitar que la navegación sea excesivamente lineal.

4. Pipeline de Ingesta y Metodología Estocástica

La integridad de la simulación depende de la procedencia y el tratamiento del dato. Es imperativo aclarar que el equipo no realizó webscraping directo, sino que utilizó un dataset pre-existente del catálogo de Supermercados Día obtenido de un tercero.

Procesamiento y Definición de Datos

El pipeline realiza una limpieza profunda: se consolidan productos de distintas marcas para establecer un precio medio y se filtran atributos críticos. Un hallazgo clave de esta auditoría es la distinción entre:

1. Semilla (Seed): Datos base de productos y precios que garantizan la reproducibilidad de las simulaciones.
2. Contexto: Variables dinámicas (stock, promociones temporales, fechas de entrada) que se inyectan en cada ejecución.

Modelado Dinámico (Distribución Beta)

Para evitar que valores atípicos (outliers) rompan el realismo, se emplea la Distribución Beta (NumPy). Esto permite generar variaciones de precios y promociones dentro de parámetros controlados. Se ha integrado un hándicap de frescura: los productos próximos a su caducidad reciben descuentos agresivos automáticos, simulando la urgencia de salida de inventario real.

5. Simulación del Comportamiento del Consumidor

Modelar la intencionalidad humana es el desafío técnico más complejo de Stockit. El sistema intenta superar el simple movimiento aleatorio mediante el perfilado de clientes.

Perfilado y Perspectiva Visual

La simulación introduce una variable de profundidad excepcional: la perspectiva según edad y altura.

* Adultos: Su campo visual prioritario alcanza la estantería 5, mientras que su capacidad de alcance físico óptimo es la estantería 4.
* Niños: Su perspectiva visual y de interacción se limita a la estantería 3. Este detalle es fundamental para la optimización de productos de impulso (chuches/snacks) en niveles inferiores.

Algoritmo de Navegación (Pathfinding)

Actualmente, el cliente sigue un orden estricto según su lista de la compra, lo que provoca desplazamientos ineficientes de "ida y vuelta". Aunque el sistema usa minimización de distancia para localizar productos, el objetivo estratégico es el "peor caso": si el supermercado es rentable para un cliente fiel que busca el camino más corto, el beneficio está asegurado. Un acierto notable es la inclusión de la "Estantería Especial" (chocolatinas, chicles, caramelos) ubicada obligatoriamente antes de la caja para incentivar la conversión de último minuto.

6. Infraestructura, Despliegue y Seguridad

La infraestructura actual presenta el mayor riesgo para la viabilidad del proyecto a largo plazo.

* Diagnóstico de Render: El uso de contenedores Docker en Render sin volúmenes persistentes provoca la pérdida total de datos en cada reinicio o periodo de inactividad del servicio.
* Estado de Desarrollo: Actualmente, el equipo depende de entornos locales para mantener la persistencia, lo que dificulta la validación cruzada.
* Evolución Arquitectónica: Existe la intención declarada de dockerizar los scripts de Python para transicionar hacia una arquitectura de microservicios, un paso necesario para desacoplar la simulación de la lógica de optimización.

7. Evaluación Crítica: Aciertos y Sugerencias de Mejora

Aciertos Estratégicos

1. Unificación Multi-modelo: El abandono de Neo4j/MongoDB en favor de ArangoDB reduce drásticamente el riesgo de fallos en el despliegue.
2. Realismo Estocástico: El uso de la Distribución Beta aporta una base estadística sólida que evita simulaciones con datos inverosímiles.
3. Modelado de Perspectiva: La inclusión de la altura visual del cliente eleva el proyecto por encima de un simple simulador logístico.

Recomendaciones de Mejora

* Garantizar Persistencia: Es obligatorio implementar una solución de almacenamiento externo (Cloud Volumes) para evitar la pérdida de la "Semilla" y los resultados de las simulaciones en la nube.
* Optimización del Pathfinding: El algoritmo debe evolucionar de un seguimiento de lista lineal a una navegación basada en la "Memoria del Cliente" (familiaridad con el entorno) para evitar recorridos erráticos de ida y vuelta.
* Estandarización de Semilla: Automatizar el script de inicialización de la base de datos para asegurar la sincronización total entre los entornos de desarrollo de todo el equipo.
* De Prototipo a Motor de Optimización: Pasar de realizar 10,000 simulaciones para observar resultados, a utilizar esos resultados como input para un algoritmo que reubique productos automáticamente basándose en el beneficio máximo alcanzado.

8. Hoja de Ruta (Roadmap) y Conclusiones

La siguiente fase debe priorizar la estabilidad de la infraestructura y la seguridad de los secretos (API keys y credenciales de despliegue). Se recomienda avanzar en el desarrollo de una interfaz web que permita a un gestor de supermercado interactuar con la API del simulador de forma transparente.

Conclusión: Stockit presenta una arquitectura conceptualmente potente. La elección de ArangoDB y el rigor estadístico en la generación de datos sintéticos compensan las carencias actuales en el despliegue. Si se resuelven los problemas de persistencia y se refina la lógica de navegación del cliente, el sistema tiene un alto potencial para convertirse en una herramienta de toma de decisiones crítica para el sector retail.
