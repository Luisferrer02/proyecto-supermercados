import { render, screen, act } from '@testing-library/react';
import { LiveLog } from './LiveLog';

// Access the mock EventSource from jest.setup.ts
const MockES = globalThis.EventSource;

beforeEach(() => MockES._clear());

describe('LiveLog', () => {
  it('shows waiting message on initial render', () => {
    render(<LiveLog url="/api/train/stream" autoStart={false} />);
    expect(screen.getByText(/Esperando salida/)).toBeInTheDocument();
  });

  it('auto-starts EventSource and shows running indicator', () => {
    render(<LiveLog url="/api/train/stream" />);
    expect(MockES._instances).toHaveLength(1);
    expect(MockES._instances[0].url).toBe('/api/train/stream');
    expect(screen.getByText(/En ejecución/)).toBeInTheDocument();
  });

  it('renders log lines from EventSource messages', () => {
    render(<LiveLog url="/api/test" />);
    const es = MockES._instances[0];

    act(() => {
      es.__emit('log', JSON.stringify({ message: 'Training MLP' }));
    });
    expect(screen.getByText('Training MLP')).toBeInTheDocument();

    act(() => {
      es.__emit('log', JSON.stringify({ message: 'Epoch 5/10' }));
    });
    expect(screen.getByText('Epoch 5/10')).toBeInTheDocument();
  });

  it('calls onDone(true) when process completes successfully', () => {
    const onDone = jest.fn();
    render(<LiveLog url="/api/test" onDone={onDone} />);
    const es = MockES._instances[0];

    act(() => {
      es.__emit('done', JSON.stringify({ message: 'Process completed successfully.' }));
    });

    expect(onDone).toHaveBeenCalledWith(true);
    expect(es.readyState).toBe(MockES.CLOSED);
  });

  it('calls onDone(false) when process fails', () => {
    const onDone = jest.fn();
    render(<LiveLog url="/api/test" onDone={onDone} />);
    const es = MockES._instances[0];

    act(() => {
      es.__emit('done', JSON.stringify({ message: 'Process exited with code 1.' }));
    });

    expect(onDone).toHaveBeenCalledWith(false);
  });

  it('renders error lines with error styling', () => {
    render(<LiveLog url="/api/test" />);
    const es = MockES._instances[0];

    act(() => {
      es.__emit('error', JSON.stringify({ message: 'Something went wrong' }));
    });

    const errorLine = screen.getByText('Something went wrong');
    expect(errorLine).toBeInTheDocument();
    expect(errorLine.className).toContain('text-red');
  });

  it('shows completed status after done event', () => {
    render(<LiveLog url="/api/test" />);
    const es = MockES._instances[0];

    act(() => {
      es.__emit('done', JSON.stringify({ message: 'Process exited with code 0' }));
    });

    expect(screen.getByText('Completado')).toBeInTheDocument();
    expect(screen.queryByText(/En ejecución/)).toBeNull();
  });
});
