import { HEADLINE } from './metrics';

describe('HEADLINE metrics', () => {
  it('exposes the expected profit lift values', () => {
    expect(HEADLINE.profitLift.absoluteEur).toBe(67794);
    expect(HEADLINE.profitLift.absoluteDisplay).toBe('+67.794 €');
  });

  it('contains optimized product and rack counts', () => {
    expect(HEADLINE.productsOptimized.value).toBe(3133);
    expect(HEADLINE.racksCovered.value).toBe(149);
  });

  it('describes the best predictor and optimizer', () => {
    expect(HEADLINE.bestPredictor.model).toBe('Transformer');
    expect(HEADLINE.bestOptimizer.model).toBe('MLP');
  });
});