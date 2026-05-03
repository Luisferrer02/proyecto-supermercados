const MockES = (global as any).EventSource;

export function getLatestEventSource() {
  const instances = MockES._instances;
  return instances[instances.length - 1];
}

export function clearEventSources() {
  MockES._clear();
}

export function simulateOptimizeStream(es: any) {
  es.__emit('step', JSON.stringify({ name: 'ingest', index: 1, total: 2 }));
  es.__emit('log', JSON.stringify({ step: 'ingest', message: 'Loading CSVs...' }));
  es.__emit('step', JSON.stringify({ name: 'predict', index: 2, total: 2 }));
  es.__emit('log', JSON.stringify({ step: 'predict', message: 'Optimizing shelves...' }));
  es.__emit('done', JSON.stringify({ ok: true }));
}

export function simulateSingleStream(es: any, success = true) {
  es.__emit('log', JSON.stringify({ message: 'Processing...' }));
  if (success) {
    es.__emit('done', JSON.stringify({ message: 'Process completed successfully.' }));
  } else {
    es.__emit('done', JSON.stringify({ message: 'Process exited with code 1.' }));
  }
}

export function simulateTrainStream(es: any) {
  es.__emit('log', JSON.stringify({ message: '[MLP] Epoch 1/80  Train MSE: 45.2000' }));
  es.__emit('log', JSON.stringify({ message: '[MLP] Epoch 80/80  Train MSE: 12.5000' }));
  es.__emit('done', JSON.stringify({ message: 'Process completed successfully.' }));
}
