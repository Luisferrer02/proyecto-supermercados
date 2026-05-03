import { parseCsvLine, parseCsv } from './csv';

describe('parseCsvLine', () => {
  it('splits a simple unquoted line', () => {
    expect(parseCsvLine('a,b,c')).toEqual(['a', 'b', 'c']);
  });

  it('preserves empty cells between consecutive commas', () => {
    expect(parseCsvLine('a,,c')).toEqual(['a', '', 'c']);
  });

  it('preserves trailing empty cell', () => {
    expect(parseCsvLine('a,b,')).toEqual(['a', 'b', '']);
  });

  it('preserves leading empty cell', () => {
    expect(parseCsvLine(',b,c')).toEqual(['', 'b', 'c']);
  });

  it('keeps commas inside quoted fields', () => {
    expect(parseCsvLine('"a,b",c,"d,e"')).toEqual(['a,b', 'c', 'd,e']);
  });

  it('unescapes doubled quotes inside quoted fields', () => {
    expect(parseCsvLine('"he said ""hi""",ok')).toEqual(['he said "hi"', 'ok']);
  });

  it('trims whitespace from values', () => {
    expect(parseCsvLine('  a , b ,  c  ')).toEqual(['a', 'b', 'c']);
  });

  it('handles a single field', () => {
    expect(parseCsvLine('only')).toEqual(['only']);
  });
});

describe('parseCsv', () => {
  it('returns an empty array for fewer than 2 lines', () => {
    expect(parseCsv('')).toEqual([]);
    expect(parseCsv('only-headers')).toEqual([]);
  });

  it('parses headers and rows into objects', () => {
    const text = 'name,price,stock\napple,1.5,10\nbanana,0.8,5';
    expect(parseCsv(text)).toEqual([
      { name: 'apple', price: '1.5', stock: '10' },
      { name: 'banana', price: '0.8', stock: '5' },
    ]);
  });

  it('normalizes Windows line endings', () => {
    const text = 'a,b\r\n1,2\r\n3,4';
    expect(parseCsv(text)).toEqual([
      { a: '1', b: '2' },
      { a: '3', b: '4' },
    ]);
  });

  it('fills missing trailing cells with empty string', () => {
    const text = 'a,b,c\n1,2';
    // value for c is undefined in the parsed array; parseCsv coalesces to ''
    expect(parseCsv(text)).toEqual([{ a: '1', b: '2', c: '' }]);
  });

  it('preserves empty middle columns (regression for shelf_level corruption)', () => {
    const text = 'name,promo,shelf_level\nA,,3\nB,yes,5';
    expect(parseCsv(text)).toEqual([
      { name: 'A', promo: '', shelf_level: '3' },
      { name: 'B', promo: 'yes', shelf_level: '5' },
    ]);
  });

  it('handles quoted fields with commas across rows', () => {
    const text = 'name,desc\n"a","x,y"\n"b","z"';
    expect(parseCsv(text)).toEqual([
      { name: 'a', desc: 'x,y' },
      { name: 'b', desc: 'z' },
    ]);
  });
});
