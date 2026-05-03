import { render, screen, fireEvent } from '@testing-library/react';
import { ShelfMap, type Product } from './ShelfMap';

const PRODUCTS: Product[] = [
  {
    name: 'Apple',
    Category: 'Fruit',
    shelf_level: '4',
    rack_id: 'R1',
    product_width_cm: '20',
    price_numeric: '3.00',
    profit_margin_percentage: '25',
    estimated_monthly_sales: '300',
  },
  {
    name: 'Milk',
    Category: 'Dairy',
    shelf_level: '2',
    rack_id: 'R1',
    product_width_cm: '15',
    price_numeric: '1.50',
    profit_margin_percentage: '10',
    estimated_monthly_sales: '50',
  },
];

describe('ShelfMap', () => {
  it('renders 7 shelves', () => {
    render(<ShelfMap products={PRODUCTS} rackId="R1" />);
    for (let i = 1; i <= 7; i++) {
      expect(screen.getByText(new RegExp(`Balda ${i}`))).toBeInTheDocument();
    }
  });

  it('marks eye-level shelves (3, 4, 5) with (ojos) label', () => {
    render(<ShelfMap products={PRODUCTS} rackId="R1" />);
    expect(screen.getByText(/Balda 3 \(ojos\)/)).toBeInTheDocument();
    expect(screen.getByText(/Balda 4 \(ojos\)/)).toBeInTheDocument();
    expect(screen.getByText(/Balda 5 \(ojos\)/)).toBeInTheDocument();
  });

  it('shows product count per shelf', () => {
    render(<ShelfMap products={PRODUCTS} rackId="R1" />);
    // One product on shelf 4, one on shelf 2, others empty
    const counts = screen.getAllByText('1');
    expect(counts.length).toBe(2);
  });

  it('shows profit legend', () => {
    render(<ShelfMap products={PRODUCTS} rackId="R1" />);
    expect(screen.getByText(/Beneficio mensual/)).toBeInTheDocument();
    expect(screen.getByText(/Bajo/)).toBeInTheDocument();
    expect(screen.getByText(/Alto/)).toBeInTheDocument();
  });

  it('shows hover tooltip with product details', () => {
    const { container } = render(<ShelfMap products={PRODUCTS} rackId="R1" />);
    const productBlocks = container.querySelectorAll('.cursor-pointer');
    expect(productBlocks.length).toBe(2);

    fireEvent.mouseEnter(productBlocks[0]);
    expect(screen.getByText('Apple')).toBeInTheDocument();
    expect(screen.getByText(/Precio/)).toBeInTheDocument();
    expect(screen.getByText(/Margen/)).toBeInTheDocument();

    fireEvent.mouseLeave(productBlocks[0]);
    expect(screen.queryByText('Apple')).toBeNull();
  });

  it('renders empty shelves when no products', () => {
    render(<ShelfMap products={[]} rackId="R1" />);
    const zeros = screen.getAllByText('0');
    expect(zeros.length).toBe(7);
  });
});
