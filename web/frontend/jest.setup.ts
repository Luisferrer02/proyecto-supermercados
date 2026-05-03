import '@testing-library/jest-dom';

// jsdom stubs
Element.prototype.scrollIntoView = jest.fn();

// Mock EventSource (not available in jsdom)
class MockEventSource {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 2;

  url: string;
  readyState = MockEventSource.OPEN;
  onerror: ((e: Event) => void) | null = null;
  private listeners: Record<string, ((e: MessageEvent) => void)[]> = {};

  constructor(url: string) {
    this.url = url;
    MockEventSource._instances.push(this);
  }

  addEventListener(type: string, cb: (e: MessageEvent) => void) {
    if (!this.listeners[type]) this.listeners[type] = [];
    this.listeners[type].push(cb);
  }

  removeEventListener(type: string, cb: (e: MessageEvent) => void) {
    if (this.listeners[type]) {
      this.listeners[type] = this.listeners[type].filter((f) => f !== cb);
    }
  }

  close() {
    this.readyState = MockEventSource.CLOSED;
  }

  // Test helper — emit a server-sent event
  __emit(type: string, data: string) {
    const event = { data } as MessageEvent;
    (this.listeners[type] || []).forEach((cb) => cb(event));
  }

  static _instances: MockEventSource[] = [];
  static _clear() {
    MockEventSource._instances = [];
  }
}

Object.assign(global, { EventSource: MockEventSource });
