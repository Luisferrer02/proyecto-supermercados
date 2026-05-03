import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import UploadPage from './page';
import { setupFetchMock } from '@/mocks/handlers';

beforeEach(() => {
  setupFetchMock();
});

describe('UploadPage', () => {
  it('renders heading and drop zone', () => {
    render(<UploadPage />);
    expect(screen.getByText('Subir archivos CSV')).toBeInTheDocument();
    expect(screen.getByText(/Arrastra los archivos CSV/)).toBeInTheDocument();
  });

  it('loads and displays file list from API', async () => {
    render(<UploadPage />);
    await waitFor(() => {
      expect(screen.getByText('sales_2025_12_december.csv')).toBeInTheDocument();
      expect(screen.getByText('sales_2025_11_november.csv')).toBeInTheDocument();
    });
  });

  it('shows badge with file count', async () => {
    render(<UploadPage />);
    await waitFor(() => {
      expect(screen.getByText('2')).toBeInTheDocument();
    });
  });

  it('shows empty state when no files', async () => {
    global.fetch = jest.fn(() => Promise.resolve({
      ok: true, status: 200,
      json: () => Promise.resolve({ files: [] }),
    } as Response)) as any;

    render(<UploadPage />);
    await waitFor(() => {
      expect(screen.getByText(/no se ha subido ningún archivo/)).toBeInTheDocument();
    });
  });

  it('calls delete API on click', async () => {
    const mockFetch = setupFetchMock();
    render(<UploadPage />);

    await waitFor(() => screen.getByText('sales_2025_12_december.csv'));
    const deleteButtons = screen.getAllByText('Eliminar');
    await userEvent.click(deleteButtons[0]);

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/upload/files/sales_2025_12_december.csv'),
      expect.objectContaining({ method: 'DELETE' }),
    );
  });
});
