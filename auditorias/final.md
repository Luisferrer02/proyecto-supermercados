# Informe de Auditoría Final: Proyecto Stockit – Cierre del Ciclo de Evaluación

### 1. Resumen Ejecutivo

La presente auditoría constituye el cierre del ciclo de evaluaciones técnicas sobre el proyecto **Stockit**, un sistema de simulación y optimización de layouts retail basado en IA. A lo largo de las sesiones anteriores se validaron la arquitectura de datos (ArangoDB), el modelado estocástico (Distribución Beta) y el perfilado de clientes por perspectiva visual. En esta sesión final se evalúan tres avances críticos para el MVP: la **migración de monolítico a microservicios**, la implementación de un **dashboard web con streaming en tiempo real** y la adopción de un **esquema de fine-tuning evolutivo** que reduce drásticamente los tiempos de entrenamiento.

El equipo ha conseguido transformar el prototipo en una plataforma operativa con pipeline de CI/CD automatizado, tres entornos de simulación de complejidad creciente (9, 50 y 109 nodos) y una interfaz de demostración utilizable ante el cliente. No obstante, la migración arquitectónica ha introducido una deuda técnica puntual — el proceso se encuentra a mitad de ruta, con rollback parcial del repositorio — y persisten problemas de calibración en la simulación de comportamiento (impulsividad desbordada). El veredicto final es favorable, con reservas acotadas sobre estabilización pre-demo.

---

### 2. Migración Arquitectónica: De Monolítico a Microservicios

El equipo ha abordado en esta iteración uno de los saltos estructurales más complejos del ciclo de vida del software: la descomposición de un núcleo monolítico en tres servicios independientes.

**Justificación técnica.** La simulación, el frontend y el modelo de IA son ciclos de trabajo fundamentalmente desacoplados. Entrenar la IA no requiere el frontend; enseñar el frontend al cliente no requiere el backend de entrenamiento. La fragmentación en tres contenedores permite que cada equipo trabaje de forma aislada y que cada servicio escale de manera independiente.

| Característica | Arquitectura Monolítica (previa) | Arquitectura de Microservicios (actual) |
| --- | --- | --- |
| Contenedores | 1 único | 3 (frontend, backend, modelo) |
| Acoplamiento de despliegue | Total | Independiente por servicio |
| Reutilización de imágenes | N/A | Docker Hub centralizado |
| Pipeline CI/CD | Único build | Build selectivo por rama afectada |
| Experiencia formativa | Baja | Alta — curva de aprendizaje DevOps |

**Observación del auditor.** Se introduce un contrapunto relevante: grandes actores como AWS han realizado el movimiento inverso (microservicios → monolítico) por razones de simplicidad arquitectónica en componentes concretos. La lección para Stockit es que la migración solo tiene sentido cuando el dominio del problema lo exige, y en su caso sí lo exige por la naturaleza independiente de los tres ciclos de trabajo. La complejidad de la transición se amortiza en las iteraciones posteriores.

**Estado actual.** Los contenedores están construidos y publicados en Docker Hub. Queda pendiente la resolución del build de PyTorch en el contenedor del modelo (fallo de importación durante el build de imagen) y la gestión de timeouts en el pull de la imagen por su tamaño elevado.

---

### 3. Dashboard Web y Monitorización en Tiempo Real

Stockit ha pasado de ser una simulación sin interfaz a disponer de un **dashboard web funcional** que sirve tanto al entrenamiento como a la demostración de producto final.

**Capacidades implementadas.**

- **API REST** con endpoints GET para sesiones, productos y resultados de simulación.
- **Streaming SSE (Server-Sent Events)** para visualización de clientes recorriendo el supermercado estantería a estantería en tiempo real.
- **Gestión de sesiones** con persistencia local (limitación actual: ausencia de servidor común).
- **Mapa interactivo de estanterías** que permite al usuario inspeccionar el inventario de cada posición.
- **Comparativa antes/después de IA** por estantería: unidades vendidas, ingresos y producto óptimo sugerido.
- **Dashboard de métricas** con ingresos totales, gasto medio por cliente, unidades vendidas, ventas por estantería, productos más vendidos y estadísticas individuales por cliente.

**Limitación actual.** La simulación no es estrictamente en tiempo real: la ejecución tarda un periodo en completarse por la naturaleza secuencial del recorrido y las consultas unitarias a ArangoDB. Una vez generados los datos, el visualizador los reproduce en tiempo real.

**Optimización en curso.** El equipo está migrando la ejecución secuencial a **multithreading con queries en batch** (clientes procesados en grupos de 10, consultas agrupadas por recorrido). El cuello de botella identificado es la latencia de ArangoDB por volumen de queries unitarias por estantería.

---

### 4. Optimización del Modelo: Fine-Tuning Evolutivo

El problema del tiempo de entrenamiento (horas por simulación completa) se ha resuelto mediante un cambio de paradigma: de **entrenamiento from scratch** a **fine-tuning dirigido**.

**Estrategia implementada.**

- Se parte de un modelo pre-entrenado cargado en memoria.
- Se **congelan todas las capas intermedias** del Transformer.
- Se modifican únicamente las **cabezas de salida** mediante un recorrido por nombre de capa que descarta las capas no coincidentes.
- Se aplica una **temperatura** controlada y un learning rate diferencial para que el proceso converja en pocas épocas.

**Evidencia cuantitativa aportada por el equipo.**

| Estrategia | Beneficio simulado |
| --- | --- |
| Entrenamiento desde cero (from scratch) | ≈ 4.400 |
| Fine-tuning sobre modelo pre-entrenado | ≈ 5.000 |

La mejora no es solo de velocidad sino también de convergencia: el fine-tuning parte de una representación ya adaptada al dominio, por lo que converge con menor ruido y alcanza mejores beneficios acumulados.

**Recomendación del auditor.** Documentar de forma sistemática la comparativa from-scratch vs fine-tuning en las próximas iteraciones: estos datos son un activo de auditoría valioso para justificar decisiones técnicas ante stakeholders.

---

### 5. Nuevos Entornos de Simulación: Tres Tamaños de Supermercado

Para validar la robustez del modelo y su capacidad de generalización, el equipo ha construido tres entornos de simulación de complejidad creciente.

| Entorno | Nodos | Referencia real | Estado |
| --- | --- | --- | --- |
| **Pequeño** | 9 | Tienda de barrio | Funcional |
| **Mediano** | 50 | Supermercado urbano de conveniencia (tipo DIA) | Funcional, con ajustes pendientes en el layout |
| **Grande** | 109 | Réplica de Supercor cercano a la universidad | Funcional, con estanterías pendientes de reubicación precisa |

**Observación sobre el entorno grande.** La distribución espacial actual no refleja con precisión la ubicación real de las estanterías en el Supercor. El equipo reconoce la desviación y tiene planificado reubicar las estanterías conforme al trabajo de campo ya documentado en auditorías previas (entrevistas con Leclerc y Eroski). Las estanterías de frío, sin embargo, ya están correctamente posicionadas.

**Recomendación.** Añadir un fondo visual tipo "vista aérea" al mapa del supermercado grande para mejorar la legibilidad del demo final.

---

### 6. Calibración del Comportamiento del Consumidor

Uno de los hallazgos más claros de la demo es la **descalibración del parámetro de impulsividad**. El sistema actual ejecuta un chequeo de compra impulsiva **en cada estantería** que el cliente atraviesa, independientemente del tipo de producto.

**Evidencia del problema.**

- En la simulación demostrada con 20 clientes: **23 de 23 compras fueron impulsivas**.
- El cliente tipo funciona como un "ludópata de supermercado": ignora la lista de la compra.
- Productos de limpieza, frío y alimentación de consumo regular activan el mismo disparador que chuches de caja.

**Recomendación técnica del auditor.** Restringir la llamada a la función de impulsividad a un **subconjunto de estanterías especiales** (caja, chocolatinas, chicles, caramelos, snacks de impulso) y no ejecutarla globalmente. El equipo ha aceptado esta recomendación y la incorporará a la siguiente iteración. La función de impulsividad debe ser una decisión de diseño, no un efecto secundario global.

---

### 7. Infraestructura, CI/CD y Despliegue

El proyecto ha alcanzado un nivel de madurez operativa notable en el ámbito DevOps.

**Pipeline GitHub Actions automatizado.**

- Detecta automáticamente qué parte del código ha cambiado (frontend, backend o modelo) por análisis de rutas.
- Actualiza **únicamente el contenedor afectado**, evitando reconstrucciones innecesarias.
- Publica las imágenes resultantes en Docker Hub.
- Dispone de flag para forzar rebuild completo durante pruebas de entorno.

**Composición final prevista.** Un `docker-compose.yml` en el repositorio principal permitirá levantar los tres servicios directamente desde Docker Hub. Se prevé un segundo `docker-compose` para levantar los servicios desde contenedores locales durante el desarrollo.

**Problemas abiertos.**

1. **Build de PyTorch falla en el contenedor del modelo** al construir la imagen. Causa en investigación.
2. **Timeouts en el pull de imagen** por tamaño elevado. Se están ampliando los `timeout` del cliente Docker.
3. **Persistencia del modelo no resuelta**. Actualmente el modelo vive en el entorno local de cada desarrollador. La recomendación es almacenarlo en un **volumen Docker montado** (aunque el volumen esté en local como paso intermedio) o en un bucket cloud (S3, One Drive, Google Drive).
4. **Artefactos de entrenamiento en GitHub**. Los datos de entrenamiento residen en ordenadores locales por no saturar el repositorio. Se recomienda migrar a un bucket externo para sincronización entre miembros del equipo.

---

### 8. Estado de Hallazgos Previos (Evolución Acumulada)

La trazabilidad de los hallazgos a lo largo de las tres sesiones permite dimensionar con precisión el progreso real del equipo y diferenciar el trabajo completado del trabajo pendiente.

| Hallazgo | Sesión 1 | Sesión 2 | Sesión Final |
| --- | --- | --- | --- |
| Unificación multi-modelo (ArangoDB) | Detectado | Solucionado | Se mantiene |
| Realismo estocástico (Distribución Beta) | Detectado | Solucionado | Se mantiene |
| Perspectiva visual por edad | Detectado | Solucionado | Se mantiene |
| Dockerización de microservicios | Pendiente | En proceso | **Solucionado** (con deuda en build de PyTorch) |
| Persistencia del modelo / pérdida en Render | Crítico | Sin cambios | **Parcial** — volumen Docker local previsto como paso intermedio |
| Lógica de memoria del cliente (bucles de paciencia) | N/A | Nuevo hallazgo | **Pendiente** — el multithreading resuelve latencia de queries, no la memoria de los agentes |
| Correlación paciencia vs longitud de lista | N/A | Nuevo hallazgo | Evolucionó hacia el problema de impulsividad (ver §6) |
| Algoritmo evolutivo elitista (Transformer) | N/A | Nuevo hallazgo | Evolucionó → fine-tuning de cabezas de salida (ver §4) |
| Aleatoriedad controlada en pathfinding | N/A | Nuevo hallazgo | Se mantiene — mitigación frente a overfitting |
| Calibración de la impulsividad | N/A | N/A | **Bloqueante nuevo** — fix acordado pre-demo |

De los diez hallazgos acumulados a lo largo del ciclo, **cinco están plenamente resueltos y consolidados**, **dos han evolucionado** hacia refinamientos técnicos concretos (fine-tuning, calibración de impulsividad), **uno está parcialmente resuelto** (persistencia en volumen Docker local) y **dos permanecen abiertos** (memoria del cliente e impulsividad), ambos con plan de resolución declarado antes de la presentación final.

---

### 9. Aciertos Estratégicos de la Sesión Final

1. **Migración a microservicios documentada y justificada** con trade-offs explícitos respecto al caso AWS. El equipo ha comprendido la complejidad del movimiento y sus condiciones de aplicabilidad.
2. **Pipeline CI/CD selectivo por rama** — madurez operativa superior a la media de proyectos académicos de este nivel.
3. **Dashboard web con streaming SSE** — visualización en tiempo real de la simulación, asset de alto valor para la presentación final.
4. **Fine-tuning evolutivo con mejora cuantificada** (4.400 → 5.000) — decisión técnica acertada que resuelve el problema de latencia de entrenamiento.
5. **Tres entornos de simulación escalonados** — validación de generalización del modelo en complejidades crecientes.

---

### 10. Recomendaciones Finales para la Presentación

1. **Restringir la impulsividad a estanterías especiales** antes de la demo final. El ratio 23/23 invalida la credibilidad del simulador.
2. **Preparar el modo demo con fine-tuning de una época** para mostrar el antes/después de IA en vivo. Si el tiempo de ejecución es elevado, mantener el proceso corriendo y transicionar a otra pestaña mientras se completa.
3. **Dataset de clientes fijo para la demo** — misma cohorte de clientes antes y después del fine-tuning para que la comparativa sea visualmente clara.
4. **Resolver la persistencia del modelo** mediante un volumen Docker al menos a nivel local antes del cierre.
5. **Añadir fondo visual al mapa del supermercado grande** y ordenar correctamente las estanterías del layout Supercor.
6. **Preparar retrospectiva arquitectónica** (V0 monolítico inicial → V3 microservicios actual) como material narrativo para la presentación.
7. **Monitorizar recursos durante la ejecución** (CPU, memoria, latencia de ArangoDB) para identificar cuellos de botella antes de escalar a 500 clientes.

---

### 11. Conclusión Técnica — Veredicto de Cierre

El proyecto Stockit culmina el ciclo de auditorías con un **veredicto favorable**. La decisión arquitectónica (ArangoDB + microservicios), el rigor estadístico (Distribución Beta), la estrategia de entrenamiento (fine-tuning evolutivo) y la capa de presentación (dashboard con streaming) conforman un MVP coherente y defendible ante cliente. La deuda técnica restante es acotada, identificada y con plan de resolución declarado.

Si el equipo logra cerrar la calibración de la impulsividad, estabilizar el build del contenedor del modelo y asegurar el dataset de demo antes de la presentación, Stockit se presenta como un prototipo de analítica prescriptiva retail con potencial real de evolución a producto comercial. El proyecto ha recorrido el camino completo desde un prototipo conceptual hasta una plataforma operativa con pipeline automatizado. Se cierra el ciclo de evaluación con reconocimiento al trabajo realizado.
