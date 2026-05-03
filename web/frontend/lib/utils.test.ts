import { cn } from './utils';

describe('cn', () => {
  it('merges class names', () => {
    expect(cn('foo', 'bar')).toBe('foo bar');
  });

  it('handles conditionals from clsx', () => {
    expect(cn('base', false && 'hidden', 'extra')).toBe('base extra');
  });

  it('deduplicates Tailwind classes via twMerge', () => {
    expect(cn('p-4', 'p-2')).toBe('p-2');
  });

  it('handles empty inputs', () => {
    expect(cn()).toBe('');
  });
});
