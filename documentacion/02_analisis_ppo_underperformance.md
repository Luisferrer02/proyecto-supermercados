# Análisis del underperformance de PPO

**Hallazgo auditoría**: A2-M02 ("PPO underperformance sin análisis").

## 1. Resultado observado

| Modelo | Profit lift sobre rack de test (Rack 2, 40 productos) |
|--------|-------|
| MLP | **+€1,648** 🥇 |
| Transformer | +€651 |
| LSTM | +€390 |
| **PPO** | **+€230** |
| Greedy (baseline) | −€2,477 |

PPO queda por debajo de todos los modelos supervisados aun siendo el más complejo conceptualmente. A continuación se justifica **por qué era esperable** y **qué se aprende de este resultado**.

## 2. Configuración de PPO en el proyecto

Referencia: [`mlops/models/ppo_agent.py`](../mlops/models/ppo_agent.py)

| Elemento | Valor | Línea |
|----------|-------|-------|
| Actor-Critic backbone compartido | `Flatten → 128 → ReLU → 128 → ReLU` | 109-138 |
| Actor head | `Linear(128, n_products)` (logits) | 126 |
| Critic head | `Linear(128, 1)` | 129 |
| Espacio de acción | Discreto: selección de **2 productos** para swap | 161-175 |
| Reward | `Δ profit = profit_rack_tras_swap − profit_rack_antes` | 72 |
| Clip ratio (ε) | 0.2 | 149 |
| γ (descuento) | 0.99 | 152 |
| LR | 3×10⁻⁴ | 148 |
| Episodios | 200–500 | 310 (`02_train_models.py`) |
| k épocas por rollout | 4 | 245 |
| Entropy coef | **Ausente** | — |

## 3. Razones técnicas del underperformance

### 3.1 Problema mal ajustado a RL: recompensa densa y calculable

RL brilla en problemas con **recompensas retardadas y composicionales** (ajedrez, Go, robótica). Nuestro problema tiene:

- **Recompensa inmediata y determinista**: mover un producto de la estantería A a la B genera un `Δ profit` calculable exactamente con `retail_physics.compute_rack_profit_advanced()` (`utils/retail_physics.py:57-119`). No hay nada que "descubrir" mediante prueba y error — la recompensa ya *es* la función objetivo.
- **Etiquetas disponibles**: puesto que podemos evaluar cualquier swap con la función de profit, tenemos infinitos pares (entrada, etiqueta) para entrenar un modelo supervisado. PPO se priva voluntariamente de esta señal.

**Consecuencia**: el supervisado aprende *directamente* la función objetivo con N ejemplos; PPO la *aproxima indirectamente* vía el baseline del crítico + muestreo estocástico de la política. Es intrínsecamente menos eficiente en datos.

### 3.2 Espacio de acción cuadrático

Cada paso PPO elige **dos productos** que se intercambian. Con N = 40 productos, hay `40·39/2 = 780` acciones posibles por paso. El actor tiene que asignar probabilidad a cada una. Esto significa:
- La política necesita **muchas más trayectorias** para cubrir el espacio.
- El muestreo estocástico descarta acciones prometedoras durante la mayor parte del entrenamiento.
- La señal de ventaja (advantage) se diluye.

Con 500 episodios × 50 pasos = 25.000 acciones, cada acción se ejecuta de media ~32 veces. Es poco para converger bien.

### 3.3 Ausencia de regularización por entropía

En nuestra implementación (`ppo_agent.py:245`), la loss PPO es solo `−min(ratio·adv, clip(ratio,1±ε)·adv) + MSE(critic)`. **No añadimos término de entropía** `−β · H(π)`.

Sin entropía explícita, la política puede colapsar prematuramente sobre un subconjunto de swaps aparentemente buenos, quedándose atrapada en óptimos locales. Esto es un "failure mode" documentado del PPO básico.

### 3.4 Entrena sobre **un solo rack** (Rack 2)

El agente PPO se entrena sobre **Rack 2 con 40 productos** y ese mismo rack se usa como test (`02_train_models.py:310`). Resultado:
- **No hay generalización**: si cambiamos de rack, el modelo entrenado no sirve (los productos son distintos, `actor → Linear(128, n_products)` tiene forma fija).
- El "modelo" final es efectivamente una **asignación de estanterías para ese rack específico**, no un modelo reutilizable.
- En producción, `05_predict.py` no carga PPO — precisamente por esto (`05_predict.py:98-260`). Solo se usa en `02_train_models.py` como punto de comparación.

### 3.5 Reward shaping ausente

El reward es el `Δ profit` crudo. No hay shaping que premie estados intermedios "buenos" (p.ej. estanterías premium ocupadas por productos caros incluso si el swap concreto es neutral). Esto hace que los primeros 50–100 episodios sean casi ruido aleatorio.

## 4. ¿Por qué entonces se incluyó PPO en el estudio?

1. **Completitud del estudio comparativo**: el enunciado del proyecto pedía comparar al menos 4 arquitecturas de familias distintas. PPO cubre la familia de **reinforcement learning**, complementando MLP/LSTM/Transformer (supervisados).
2. **Evidencia empírica**: sin incluir PPO, nuestra afirmación "el problema se resuelve mejor con aprendizaje supervisado" sería solo una hipótesis. Incluyéndolo y midiéndolo, pasa a ser una **conclusión experimental justificada**.
3. **Descartar el approach explícitamente** es más defendible que ignorarlo.

## 5. Conclusión para defensa

> **¿Por qué PPO rinde peor que modelos más simples?**
>
> Porque el problema de optimización de estanterías no encaja con los supuestos donde RL aporta valor: la recompensa es inmediata, calculable y determinista; el espacio de acción es cuadrático en productos; y la señal de etiqueta es gratuita para modelos supervisados. Además, nuestra implementación entrena sobre un único rack sin entropía regularizadora, lo que descarta generalización. PPO se incluye en el estudio para **dejar constancia experimental** de que el approach RL es inferior aquí — no como fracaso sino como resultado negativo informativo.

## 6. Qué haríamos con más tiempo

Si quisiéramos un PPO competitivo:
- **Feature-based actor** en lugar de `Linear(128, n_products)`: la política devolvería un *score* por producto en función de features (precio, margen, ventas), generalizando a cualquier rack.
- **Shaped reward**: premiar también uso de estanterías eye-level por productos con alto `price × margin × sales`.
- **Entropy bonus** `β = 0.01`.
- **Curriculum learning**: empezar con racks de 5 productos, ir subiendo.
- **10× más episodios** (5.000) en paralelo (vectorized env).

No lo hacemos porque los supervisados ya saturan el problema y mejorar PPO no aporta al objetivo de negocio.
