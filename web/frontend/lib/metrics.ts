/**
 * Canonical headline metrics.
 *
 * Single source of truth for any metric shown on dashboards, landing pages,
 * infographics, or investor-style reports. Use `HEADLINE.profitLift.display`
 * instead of hardcoding "+16.2%" in components.
 *
 * When a training run produces new numbers, update these constants and every
 * surface rebinds automatically.
 */

export const HEADLINE = {
  profitLift: {
    absoluteEur: 67_794,
    absoluteDisplay: "+67.794 €",
    percent: 16.2,
    display: "+16,2 %",
    label: "Incremento mensual de beneficio",
    hint: "Frente al layout original, sobre el mes de referencia",
  },
  productsOptimized: {
    value: 3_133,
    display: "3.133",
    label: "Productos optimizados",
  },
  racksCovered: {
    value: 149,
    display: "149",
    label: "Estanterías cubiertas",
  },
  bestPredictor: {
    model: "Transformer",
    rmseEur: 17.3,
    label: "Mejor predictor (RMSE)",
    display: "Transformer · 17,3 €",
  },
  bestOptimizer: {
    model: "MLP",
    liftEur: 1_648,
    label: "Mejor optimizador",
    display: "MLP · +1.648 €",
  },
} as const;
