import { render, screen, waitFor, act, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import EvaluatePage from './page';
import { setupFetchMock } from '@/mocks/handlers';
import { getLatestEventSource, clearEventSources, simulateSingleStream } from '@/mocks/sse-helpers';

jest.mock('@/components/LiveLog', () => ({
  LiveLog: ({ url, onDone }: { url: string; onDone?: (s: boolean) => void }) => {
    const MockES = globalThis.EventSource;
    const es = new MockES(url);
    es.addEventListener('done', (e: MessageEvent) => {
      const d = JSON.parse(e.data);
      const success = d.message?.includes('successfully') || d.message?.includes('code 0');
      onDone?.(success);
    });
    return <div data-testid="live-log">LiveLog: {url}</div>;
  },
}));

beforeEach(() => {
  setupFetchMock();
  clearEventSources();
});

describe('EvaluatePage', () => {
  it('renders heading and regenerate button', () => {
    render(<EvaluatePage />);
    expect(screen.getByText('Gráficas de evaluación')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Regenerar gráficas/ })).toBeInTheDocument();
  });

  it('loads and displays existing charts', async () => {
    render(<EvaluatePage />);
    await waitFor(() => {
      expect(screen.getByText('2/4 disponibles')).toBeInTheDocument();
    });
    expect(screen.getByAltText('mse_comparison.png')).toBeInTheDocument();
    expect(screen.getByAltText('profit_comparison.png')).toBeInTheDocument();
  });

  it('shows empty state when no charts exist', async () => {
    global.fetch = jest.fn(() => Promise.resolve({
      ok: true, status: 200,
      json: () => Promise.resolve({
        charts: [
          { name: 'mse_comparison.png', exists: false, url: '/results/mse_comparison.png' },
        ],
        running: false,
      }),
    } as Response)) as typeof fetch;

    render(<EvaluatePage />);
    await waitFor(() => {
      expect(screen.getByText(/Aún no hay gráficas/)).toBeInTheDocument();
    });
  });

  it('starts streaming on button click', async () => {
    render(<EvaluatePage />);
    await userEvent.click(screen.getByRole('button', { name: /Regenerar gráficas/ }));

    expect(screen.getByTestId('live-log')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Generando/ })).toBeDisabled();
    expect(screen.getByRole('button', { name: /Cancelar/ })).toBeInTheDocument();
  });

  it('reloads charts after streaming completes', async () => {
    const mockFetch = setupFetchMock();
    render(<EvaluatePage />);
    await userEvent.click(screen.getByRole('button', { name: /Regenerar gráficas/ }));

    const es = getLatestEventSource();
    act(() => simulateSingleStream(es, true));

    await waitFor(() => {
      const statusCalls = mockFetch.mock.calls.filter(
        ([url]: [string | URL | Request, (RequestInit | undefined)?]) => typeof url === 'string' && url.includes('/api/evaluate/status')
      );
      expect(statusCalls.length).toBeGreaterThanOrEqual(2);
    });
  });

  it('opens lightbox on chart click', async () => {
    render(<EvaluatePage />);
    await waitFor(() => screen.getByAltText('mse_comparison.png'));

    const card = screen.getByRole('button', { name: /Ampliar Precisión de los modelos/ });
    await userEvent.click(card);

    const dialog = screen.getByRole('dialog');
    expect(dialog).toBeInTheDocument();
    expect(dialog).toHaveAttribute('aria-label', 'Precisión de los modelos (error típico)');
  });

  it('closes lightbox on Escape key', async () => {
    render(<EvaluatePage />);
    await waitFor(() => screen.getByAltText('mse_comparison.png'));

    const card = screen.getByRole('button', { name: /Ampliar Precisión de los modelos/ });
    await userEvent.click(card);
    expect(screen.getByRole('dialog')).toBeInTheDocument();

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('closes lightbox on X button click', async () => {
    render(<EvaluatePage />);
    await waitFor(() => screen.getByAltText('mse_comparison.png'));

    const card = screen.getByRole('button', { name: /Ampliar Precisión de los modelos/ });
    await userEvent.click(card);
    expect(screen.getByRole('dialog')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /Cerrar/ }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('calls stop API on cancel', async () => {
    const mockFetch = setupFetchMock();
    render(<EvaluatePage />);
    await userEvent.click(screen.getByRole('button', { name: /Regenerar gráficas/ }));
    await userEvent.click(screen.getByRole('button', { name: /Cancelar/ }));

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/evaluate/stop'),
      expect.objectContaining({ method: 'POST' }),
    );
  });
});
