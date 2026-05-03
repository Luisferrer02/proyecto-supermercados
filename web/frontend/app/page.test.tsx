import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import HomePage from './page';
import { setupFetchMock } from '@/mocks/handlers';
import { getLatestEventSource, clearEventSources, simulateOptimizeStream } from '@/mocks/sse-helpers';

jest.mock('@/components/ShelfSankey', () => ({
  ShelfSankey: ({ movements }: any) => (
    <div data-testid="shelf-sankey">ShelfSankey: {movements.length} movements</div>
  ),
}));

beforeEach(() => {
  setupFetchMock();
  clearEventSources();
});

describe('HomePage', () => {
  it('renders heading and optimize button', async () => {
    render(<HomePage />);
    expect(screen.getByText('Optimiza tu supermercado')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Optimizar/ })).toBeInTheDocument();
    });
  });

  it('loads file list on mount', async () => {
    render(<HomePage />);
    await waitFor(() => {
      expect(screen.getByText('sales_2025_12_december.csv')).toBeInTheDocument();
      expect(screen.getByText('sales_2025_11_november.csv')).toBeInTheDocument();
    });
  });

  it('loads default month on mount', async () => {
    render(<HomePage />);
    await waitFor(() => {
      const monthInput = screen.getByDisplayValue('2026-01');
      expect(monthInput).toBeInTheDocument();
    });
  });

  it('shows error when no files uploaded', async () => {
    global.fetch = jest.fn((url: any) => {
      const urlStr = typeof url === 'string' ? url : url.toString();
      if (urlStr.includes('/api/upload/files')) {
        return Promise.resolve({
          ok: true, status: 200,
          json: () => Promise.resolve({ files: [] }),
        } as Response);
      }
      if (urlStr.includes('/api/optimize/default-month')) {
        return Promise.resolve({
          ok: true, status: 200,
          json: () => Promise.resolve({ month: '2026-01' }),
        } as Response);
      }
      return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) } as Response);
    }) as any;

    render(<HomePage />);
    await waitFor(() => screen.getByRole('button', { name: /Optimizar/ }));

    expect(screen.getByRole('button', { name: /Optimizar/ })).toBeDisabled();
  });

  it('transitions to running phase on optimize click', async () => {
    render(<HomePage />);
    await waitFor(() => screen.getByText('sales_2025_12_december.csv'));
    await userEvent.click(screen.getByRole('button', { name: /Optimizar/ }));

    expect(screen.getByText(/Optimizando tu supermercado/)).toBeInTheDocument();
    expect(screen.getByText(/Cancelar/)).toBeInTheDocument();
  });

  it('shows progress during streaming', async () => {
    render(<HomePage />);
    await waitFor(() => screen.getByText('sales_2025_12_december.csv'));
    await userEvent.click(screen.getByRole('button', { name: /Optimizar/ }));

    const es = getLatestEventSource();
    act(() => {
      es.__emit('step', JSON.stringify({ name: 'ingest', index: 1, total: 2 }));
      es.__emit('log', JSON.stringify({ message: 'Loading CSVs...' }));
    });

    expect(screen.getByText(/Paso 1 de 2/)).toBeInTheDocument();
  });

  it('shows KPIs after optimize completes', async () => {
    render(<HomePage />);
    await waitFor(() => screen.getByText('sales_2025_12_december.csv'));
    await userEvent.click(screen.getByRole('button', { name: /Optimizar/ }));

    const es = getLatestEventSource();
    act(() => simulateOptimizeStream(es));

    await waitFor(() => {
      expect(screen.getByText(/Beneficio extra al mes/)).toBeInTheDocument();
    });
    expect(screen.getByTestId('shelf-sankey')).toBeInTheDocument();
  });

  it('shows racks table after results load', async () => {
    render(<HomePage />);
    await waitFor(() => screen.getByText('sales_2025_12_december.csv'));
    await userEvent.click(screen.getByRole('button', { name: /Optimizar/ }));

    const es = getLatestEventSource();
    act(() => simulateOptimizeStream(es));

    await waitFor(() => {
      expect(screen.getByText(/Estanterías con mayor mejora/)).toBeInTheDocument();
    });
  });

  it('shows error on failed done event', async () => {
    render(<HomePage />);
    await waitFor(() => screen.getByText('sales_2025_12_december.csv'));
    await userEvent.click(screen.getByRole('button', { name: /Optimizar/ }));

    const es = getLatestEventSource();
    act(() => {
      es.__emit('done', JSON.stringify({ ok: false, failedStep: 'predict', code: 1 }));
    });

    await waitFor(() => {
      expect(screen.getByText(/Fallo en el paso "predict"/)).toBeInTheDocument();
    });
  });

  it('calls stop API on cancel and returns to upload phase', async () => {
    const mockFetch = setupFetchMock();
    render(<HomePage />);
    await waitFor(() => screen.getByText('sales_2025_12_december.csv'));
    await userEvent.click(screen.getByRole('button', { name: /Optimizar/ }));
    await userEvent.click(screen.getByText(/Cancelar/));

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/optimize/stop'),
      expect.objectContaining({ method: 'POST' }),
    );
  });

  it('allows returning to upload phase after results', async () => {
    render(<HomePage />);
    await waitFor(() => screen.getByText('sales_2025_12_december.csv'));
    await userEvent.click(screen.getByRole('button', { name: /Optimizar/ }));

    const es = getLatestEventSource();
    act(() => simulateOptimizeStream(es));

    await waitFor(() => screen.getByText(/Optimizar otro mes/));
    await userEvent.click(screen.getByText(/Optimizar otro mes/));

    expect(screen.getByText('Optimiza tu supermercado')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Optimizar/ })).toBeInTheDocument();
  });
});
