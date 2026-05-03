import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TrainPage from './page';
import { setupFetchMock } from '@/mocks/handlers';
import { getLatestEventSource, clearEventSources, simulateTrainStream } from '@/mocks/sse-helpers';

// Mock recharts to avoid jsdom SVG issues
jest.mock('recharts', () => ({
  BarChart: ({ children }: any) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => <div />,
  XAxis: () => <div />,
  YAxis: () => <div />,
  Tooltip: () => <div />,
  Legend: () => <div />,
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
}));

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

beforeEach(() => {
  setupFetchMock();
  clearEventSources();
});

describe('TrainPage', () => {
  it('renders heading and start button', () => {
    render(<TrainPage />);
    expect(screen.getByText('Comparación de modelos')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Lanzar entrenamiento/ })).toBeInTheDocument();
  });

  it('loads results on mount and displays chart', async () => {
    render(<TrainPage />);
    await waitFor(() => {
      expect(screen.getAllByTestId('bar-chart').length).toBeGreaterThan(0);
    });
  });

  it('starts streaming on button click', async () => {
    render(<TrainPage />);
    await userEvent.click(screen.getByRole('button', { name: /Lanzar entrenamiento/ }));

    expect(screen.getByTestId('live-log')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Entrenando/ })).toBeDisabled();
  });

  it('reloads results after training completes', async () => {
    const mockFetch = setupFetchMock();
    render(<TrainPage />);
    await userEvent.click(screen.getByRole('button', { name: /Lanzar entrenamiento/ }));

    const es = getLatestEventSource();
    act(() => simulateTrainStream(es));

    await waitFor(() => {
      const resultsCalls = mockFetch.mock.calls.filter(
        ([url]: any) => typeof url === 'string' && url.includes('/api/train/results')
      );
      expect(resultsCalls.length).toBeGreaterThanOrEqual(2);
    });
  });

  it('shows cancel button during streaming', async () => {
    render(<TrainPage />);
    await userEvent.click(screen.getByRole('button', { name: /Lanzar entrenamiento/ }));
    expect(screen.getByRole('button', { name: /Cancelar/ })).toBeInTheDocument();
  });
});
