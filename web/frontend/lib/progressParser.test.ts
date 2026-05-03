import { parseLine, INITIAL, type FriendlyProgress } from './progressParser';

describe('parseLine', () => {
  const ingestState = { stage: 'ingest' as const };
  const predictState = { stage: 'predict' as const };

  describe('INGEST_RULES', () => {
    it('detects data ingestion start', () => {
      const r = parseLine(ingestState, 'SHELF OPTIMIZER -- DATA INGESTION');
      expect(r.label).toBe('Preparando la ingesta de datos');
      expect(r.stage).toBe('ingest');
    });

    it('detects schema validation', () => {
      const r = parseLine(ingestState, '2026-01-01 Schema validation passed for 3 files');
      expect(r.label).toBe('Validando archivos CSV');
    });

    it('detects embeddings indexing', () => {
      const r = parseLine(ingestState, 'Embeddings: 150 products indexed');
      expect(r.label).toBe('Indexando productos por categoría');
    });

    it('detects MLP training', () => {
      const r = parseLine(ingestState, 'Training MLP on 500 samples');
      expect(r.label).toBe('Entrenando modelo rápido');
    });

    it('detects Transformer training', () => {
      const r = parseLine(ingestState, 'Training Transformer on 500 samples');
      expect(r.label).toBe('Entrenando modelo preciso');
    });

    it('detects MLP saved', () => {
      const r = parseLine(ingestState, 'MLP saved to models/mlp.pt');
      expect(r.label).toBe('Modelo rápido guardado');
    });

    it('detects models ready', () => {
      const r = parseLine(ingestState, 'Models: 2 trained');
      expect(r.label).toBe('Modelos listos');
    });
  });

  describe('PREDICT_RULES', () => {
    it('detects RAG step', () => {
      const r = parseLine(predictState, 'Step 1: RAG retrieval');
      expect(r.label).toBe('Buscando datos históricos relevantes');
    });

    it('detects LLM query', () => {
      const r = parseLine(predictState, 'Querying LLM for forecast');
      expect(r.label).toBe('Consultando la IA');
    });

    it('detects heuristic fallback', () => {
      const r = parseLine(predictState, 'falling back to heuristic method');
      expect(r.label).toBe('Usando reglas de temporada (la IA no responde)');
    });

    it('detects optimization step', () => {
      const r = parseLine(predictState, 'Step 5: ensemble optimization');
      expect(r.label).toBe('Calculando la nueva disposición');
    });

    it('detects rack reordering', () => {
      const r = parseLine(predictState, 'Rack R1: 15 products placed');
      expect(r.label).toBe('Recolocando productos');
    });

    it('detects results saving', () => {
      const r = parseLine(predictState, 'OPTIMIZATION RESULTS saved');
      expect(r.label).toBe('Guardando resultados');
    });
  });

  describe('epoch progress extraction', () => {
    it('extracts MLP epoch percentage', () => {
      const r = parseLine(ingestState, 'Epoch 30/100 loss=0.05');
      expect(r.stage).toBe('ingest');
      expect(r.label).toBe('Entrenando modelo rápido');
      expect(r.percent).toBe(30);
      expect(r.detail).toBe('Iteración 30 de 100');
    });

    it('extracts Transformer epoch percentage', () => {
      const r = parseLine(ingestState, 'Transformer Epoch 50/200 val=0.04');
      expect(r.label).toBe('Entrenando modelo preciso');
      expect(r.percent).toBe(25);
    });

    it('handles epoch at 100%', () => {
      const r = parseLine(ingestState, 'Epoch 100/100 done');
      expect(r.percent).toBe(100);
    });

    it('ignores epoch in predict stage', () => {
      const r = parseLine(predictState, 'Epoch 5/10');
      expect(r.percent).toBeUndefined();
    });
  });

  describe('unknown lines', () => {
    it('returns fallback with last label hint', () => {
      const r = parseLine({ stage: 'ingest', labelHint: 'Previo' }, 'some random log');
      expect(r.label).toBe('Previo');
    });

    it('returns "Procesando…" when no hint', () => {
      const r = parseLine(ingestState, 'unknown output');
      expect(r.label).toBe('Procesando…');
    });
  });

  describe('INITIAL', () => {
    it('has idle stage', () => {
      expect(INITIAL.stage).toBe('idle');
      expect(INITIAL.label).toBe('Listo para empezar');
    });
  });
});
