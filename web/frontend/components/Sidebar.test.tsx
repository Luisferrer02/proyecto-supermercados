import type React from 'react';
import { render, screen } from '@testing-library/react';
import { Sidebar } from './Sidebar';
import { usePathname } from 'next/navigation';

jest.mock('next/link', () => {
  return ({ children, href, ...rest }: { children: React.ReactNode; href: string; [key: string]: unknown }) => <a href={href} {...rest}>{children}</a>;
});

jest.mock('next/navigation', () => ({
  usePathname: jest.fn(),
}));
const mockUsePathname = usePathname as jest.MockedFunction<typeof usePathname>;

describe('Sidebar', () => {
  beforeEach(() => {
    mockUsePathname.mockReturnValue('/');
  });

  it('renders the navigation and footer', () => {
    render(<Sidebar />);

    expect(screen.getByText('ShelfOpt')).toBeInTheDocument();
    expect(screen.getByText('MLOps Pipeline')).toBeInTheDocument();
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Mercadona · 4,772 products')).toBeInTheDocument();
  });

  it('highlights the active route', () => {
    mockUsePathname.mockReturnValue('/train');

    render(<Sidebar />);

    const activeLink = screen.getByRole('link', { name: 'Train Models' });
    expect(activeLink.className).toContain('bg-sidebar-primary');
  });
});