import { render, screen, fireEvent } from '@testing-library/react';
import { TopNav } from './TopNav';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  usePathname: jest.fn(() => '/'),
}));

// Mock next/link — render a plain <a> tag
jest.mock('next/link', () => {
  return ({ children, href, ...rest }: any) => (
    <a href={href} {...rest}>{children}</a>
  );
});

import { usePathname } from 'next/navigation';
const mockUsePathname = usePathname as jest.MockedFunction<typeof usePathname>;

describe('TopNav', () => {
  beforeEach(() => {
    mockUsePathname.mockReturnValue('/');
  });

  it('renders the SHELFOPT brand link', () => {
    render(<TopNav />);
    expect(screen.getByText('SHELFOPT')).toBeInTheDocument();
  });

  it('renders the Optimizar main link', () => {
    render(<TopNav />);
    expect(screen.getByText('Optimizar')).toBeInTheDocument();
  });

  it('renders the Avanzado dropdown button', () => {
    render(<TopNav />);
    expect(screen.getByText('Avanzado')).toBeInTheDocument();
  });

  it('opens advanced dropdown on click', () => {
    render(<TopNav />);
    const btn = screen.getByText('Avanzado');
    fireEvent.click(btn);
    expect(screen.getByText(/Subir archivos/)).toBeInTheDocument();
    expect(screen.getByText(/Ingesta manual/)).toBeInTheDocument();
    expect(screen.getByText(/Entrenamiento comparado/)).toBeInTheDocument();
    expect(screen.getByText(/Gráficas de evaluación/)).toBeInTheDocument();
    expect(screen.getByText(/Predicción paso a paso/)).toBeInTheDocument();
  });

  it('closes dropdown on second click', () => {
    render(<TopNav />);
    const btn = screen.getByText('Avanzado');
    fireEvent.click(btn);
    expect(screen.getByText(/Subir archivos/)).toBeInTheDocument();
    fireEvent.click(btn);
    expect(screen.queryByText(/Subir archivos/)).toBeNull();
  });

  it('highlights Avanzado when on an advanced route', () => {
    mockUsePathname.mockReturnValue('/train');
    render(<TopNav />);
    const btn = screen.getByText('Avanzado');
    expect(btn.closest('button')!.className).toContain('bg-primary');
  });
});
