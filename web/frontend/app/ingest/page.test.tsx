import { render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import IngestPage from './page';
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

describe('IngestPage', () => {
  it('renders the page heading and start button', () => {
    render(<IngestPage />);
    expect(screen.getByText('Ingesta manual')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Lanzar ingesta/ })).toBeInTheDocument();
  });

  it('shows info text before starting', () => {
    render(<IngestPage />);
    expect(screen.getByText(/Asegúrate de haber subido los CSVs/)).toBeInTheDocument();
  });

  it('starts streaming on button click', async () => {
    render(<IngestPage />);
    await userEvent.click(screen.getByRole('button', { name: /Lanzar ingesta/ }));

    expect(screen.getByTestId('live-log')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Cancelar/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Ingiriendo/ })).toBeDisabled();
  });

  it('shows success message after done event', async () => {
    render(<IngestPage />);
    await userEvent.click(screen.getByRole('button', { name: /Lanzar ingesta/ }));

    const es = getLatestEventSource();
    act(() => simulateSingleStream(es, true));

    expect(screen.getByText(/Ingesta completada/)).toBeInTheDocument();
  });

  it('shows error message after failed done event', async () => {
    render(<IngestPage />);
    await userEvent.click(screen.getByRole('button', { name: /Lanzar ingesta/ }));

    const es = getLatestEventSource();
    act(() => simulateSingleStream(es, false));

    expect(screen.getByText(/terminado con errores/)).toBeInTheDocument();
  });

  it('calls stop API on cancel', async () => {
    const mockFetch = setupFetchMock();
    render(<IngestPage />);
    await userEvent.click(screen.getByRole('button', { name: /Lanzar ingesta/ }));
    await userEvent.click(screen.getByRole('button', { name: /Cancelar/ }));

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/ingest/stop'),
      expect.objectContaining({ method: 'POST' }),
    );
  });
});
