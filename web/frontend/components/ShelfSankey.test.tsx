import { render, screen } from '@testing-library/react';
import { ShelfSankey } from './ShelfSankey';

describe('ShelfSankey', () => {
  it('renders empty state message when no movements', () => {
    render(<ShelfSankey movements={[]} />);
    expect(screen.getByText(/No hay reubicaciones/)).toBeInTheDocument();
  });

  it('renders SVG with ribbons for valid movements', () => {
    const movements = [
      { from: 3, to: 4, count: 10 },
      { from: 1, to: 5, count: 5 },
    ];
    const { container } = render(<ShelfSankey movements={movements} />);
    const svg = container.querySelector('svg');
    expect(svg).toBeInTheDocument();
    // Should have path elements for ribbons
    const paths = container.querySelectorAll('path');
    expect(paths.length).toBe(2);
  });

  it('renders shelf labels for all 7 shelves on each side', () => {
    const movements = [{ from: 1, to: 7, count: 3 }];
    const { container } = render(<ShelfSankey movements={movements} />);
    const texts = container.querySelectorAll('text');
    // 7 left labels + 7 right labels + 7 left counts + 7 right counts + 2 headers = 30
    expect(texts.length).toBe(30);
  });

  it('marks eye-level shelves with (ojos) label', () => {
    const movements = [{ from: 4, to: 4, count: 2 }];
    render(<ShelfSankey movements={movements} />);
    const eyeLabels = screen.getAllByText(/ojos/);
    // Shelves 3, 4, 5 are eye level — each appears on left and right = 6
    expect(eyeLabels.length).toBe(6);
  });

  it('shows product count in ribbon tooltip', () => {
    const movements = [{ from: 2, to: 5, count: 8 }];
    render(<ShelfSankey movements={movements} />);
    expect(screen.getByText(/8 productos/)).toBeInTheDocument();
  });
});
