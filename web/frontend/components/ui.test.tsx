import { render, screen } from '@testing-library/react';

import { Badge } from './ui/badge';
import { Button } from './ui/button';
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from './ui/card';

jest.mock('radix-ui', () => ({
  Slot: {
    Root: ({ children, ...props }: any) => <span {...props}>{children}</span>,
  },
}));

describe('Button', () => {
  it('renders default and variant classes', () => {
    render(<Button>Save</Button>);
    const button = screen.getByRole('button', { name: 'Save' });
    expect(button).toHaveAttribute('data-variant', 'default');
    expect(button).toHaveAttribute('data-size', 'default');
  });

  it('supports asChild rendering', () => {
    render(<Button asChild><a href="/upload">Upload</a></Button>);
    expect(screen.getByRole('link', { name: 'Upload' })).toBeInTheDocument();
  });

  it('applies custom size and variant', () => {
    render(<Button variant="ghost" size="sm">Ghost</Button>);
    const button = screen.getByRole('button', { name: 'Ghost' });
    expect(button).toHaveAttribute('data-variant', 'ghost');
    expect(button).toHaveAttribute('data-size', 'sm');
  });
});

describe('Badge', () => {
  it('renders default badge and asChild', () => {
    render(<Badge>2</Badge>);
    expect(screen.getByText('2')).toHaveAttribute('data-variant', 'default');
  });

  it('renders asChild with outline variant', () => {
    render(<Badge asChild variant="outline"><a href="/predict">Predict</a></Badge>);
    expect(screen.getByRole('link', { name: 'Predict' })).toBeInTheDocument();
  });
});

describe('Card', () => {
  it('renders card sections and custom size', () => {
    render(
      <Card size="sm">
        <CardHeader>
          <CardTitle>Title</CardTitle>
          <CardDescription>Description</CardDescription>
          <CardAction>Action</CardAction>
        </CardHeader>
        <CardContent>Body</CardContent>
        <CardFooter>Footer</CardFooter>
      </Card>,
    );

    const card = screen.getByText('Body').closest('[data-slot="card"]');
    expect(card).toHaveAttribute('data-size', 'sm');
    expect(screen.getByText('Title')).toBeInTheDocument();
    expect(screen.getByText('Description')).toBeInTheDocument();
    expect(screen.getByText('Action')).toBeInTheDocument();
    expect(screen.getByText('Footer')).toBeInTheDocument();
  });
});