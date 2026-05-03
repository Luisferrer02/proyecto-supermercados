import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import PredictPage from './page';
import { setupFetchMock } from '@/mocks/handlers';
import { getLatestEventSource, clearEventSources, simulateSingleStream } from '@/mocks/sse-helpers';

jest.mock('@/components/LiveLog', () => ({
  LiveLog: ({ url, onDone }: { url: string; onDone?: (s: boolean) => void }) => {
    const MockES = (global as any).EventSource;
    const es = new MockES(url);
    es.addEventListener('done', (e: any) => {
      const d = JSON.parse(e.data);
      const success = d.message?.includes('successfully') || d.message?.includes('code 0');
      onDone?.(success);
    });
    return <div data-testid="live-log">LiveLog: {url}</div>;
  },
}));

jest.mock('@/components/ShelfMap', () => ({
  ShelfMap: ({ products, rackId }: any) => (
    <div data-testid="shelf-map">ShelfMap: rack={rackId} products={products.length}</div>
  ),
}));

jest.mock('@/components/ShelfSankey', () => ({
  ShelfSankey: ({ movements }: any) => (
    <div data-testid="shelf-sankey">ShelfSankey: {movements.length} movements</div>
  ),
}));

beforeEach(() => {
  setupFetchMock();
  clearEventSources();
});

describe('PredictPage', () => {
  it('renders heading and launch button', () => {
    render(<PredictPage />);
    expect(screen.getByText('Predicción avanzada')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Lanzar predicción/ })).toBeInTheDocument();
  });

  it('loads past months dropdown', async () => {
    render(<PredictPage />);
    await waitFor(() => {
      expect(screen.getByText('Cargar resultado anterior:')).toBeInTheDocument();
    });
    expect(screen.getByText('2025-12')).toBeInTheDocument();
    expect(screen.getByText('2025-11')).toBeInTheDocument();
  });

  it('starts streaming on button click', async () => {
    render(<PredictPage />);
    await userEvent.click(screen.getByRole('button', { name: /Lanzar predicción/ }));

    expect(screen.getByTestId('live-log')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Ejecutando/ })).toBeDisabled();
    expect(screen.getByRole('button', { name: /Cancelar/ })).toBeInTheDocument();
  });

  it('loads results after stream completes and shows ShelfMap', async () => {
    render(<PredictPage />);
    await userEvent.click(screen.getByRole('button', { name: /Lanzar predicción/ }));

    const es = getLatestEventSource();
    act(() => simulateSingleStream(es, true));

    await waitFor(() => {
      expect(screen.getByTestId('shelf-map')).toBeInTheDocument();
    });
    expect(screen.getByText(/Detalle por estantería/)).toBeInTheDocument();
  });

  it('displays forecast multipliers after results load', async () => {
    render(<PredictPage />);
    await userEvent.click(screen.getByRole('button', { name: /Lanzar predicción/ }));

    const es = getLatestEventSource();
    act(() => simulateSingleStream(es, true));

    await waitFor(() => {
      expect(screen.getByText('Fruta')).toBeInTheDocument();
    });
    expect(screen.getByText('×1.15')).toBeInTheDocument();
  });

  it('shows rack summary table after results load', async () => {
    render(<PredictPage />);
    await userEvent.click(screen.getByRole('button', { name: /Lanzar predicción/ }));

    const es = getLatestEventSource();
    act(() => simulateSingleStream(es, true));

    await waitFor(() => {
      expect(screen.getByText(/Resumen de beneficio por estantería/)).toBeInTheDocument();
    });
  });

  it('shows error when results have no products', async () => {
    const originalFetch = global.fetch;
    render(<PredictPage />);
    await userEvent.click(screen.getByRole('button', { name: /Lanzar predicción/ }));

    // Override fetch to return error for predict results
    global.fetch = jest.fn((url: any, init?: any) => {
      const urlStr = typeof url === 'string' ? url : url.toString();
      if (urlStr.includes('/api/predict/results')) {
        return Promise.resolve({
          ok: true, status: 200,
          json: () => Promise.resolve({ error: 'No data available' }),
        } as Response);
      }
      if (urlStr.includes('/api/optimize/results')) {
        return Promise.resolve({
          ok: true, status: 200,
          json: () => Promise.resolve({ movements: [] }),
        } as Response);
      }
      return (originalFetch as any)(url, init);
    }) as any;

    const es = getLatestEventSource();
    act(() => simulateSingleStream(es, true));

    await waitFor(() => {
      expect(screen.getByText(/No data available/)).toBeInTheDocument();
    });
  });

  it('calls stop API on cancel', async () => {
    const mockFetch = setupFetchMock();
    render(<PredictPage />);
    await userEvent.click(screen.getByRole('button', { name: /Lanzar predicción/ }));
    await userEvent.click(screen.getByRole('button', { name: /Cancelar/ }));

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/predict/stop'),
      expect.objectContaining({ method: 'POST' }),
    );
  });

  it('shows initial info text when no results', () => {
    render(<PredictPage />);
    expect(screen.getByText(/Lanza la ingesta primero/)).toBeInTheDocument();
  });
});
