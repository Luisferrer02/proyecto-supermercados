interface MockEventSourceInstance {
  url: string;
  readyState: number;
  onerror: ((e: Event) => void) | null;
  addEventListener(type: string, cb: (e: MessageEvent) => void): void;
  removeEventListener(type: string, cb: (e: MessageEvent) => void): void;
  close(): void;
  __emit(type: string, data: string): void;
}

interface MockEventSourceConstructor {
  new (url: string): MockEventSourceInstance;
  CONNECTING: number;
  OPEN: number;
  CLOSED: number;
  _instances: MockEventSourceInstance[];
  _clear(): void;
}

declare const MockEventSource: MockEventSourceConstructor;

declare namespace globalThis {
  // eslint-disable-next-line no-var
  var EventSource: MockEventSourceConstructor;
}
