// RFC-4180-ish CSV parser. Handles quoted fields with commas, escaped quotes
// (""), and EMPTY fields between consecutive commas. The previous regex-based
// approach (`/(".*?"|[^,]+)(?=,|$)/g`) silently dropped empty cells, which
// shifted later columns left and corrupted shelf_level downstream.

export function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"' && line[i + 1] === '"') { cur += '"'; i += 1; }
      else if (c === '"') { inQuotes = false; }
      else { cur += c; }
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ',') {
      out.push(cur);
      cur = '';
    } else {
      cur += c;
    }
  }
  out.push(cur);
  return out.map(v => v.trim());
}

export function parseCsv(text: string): Record<string, string>[] {
  const lines = text.replace(/\r\n/g, '\n').trim().split('\n');
  if (lines.length < 2) return [];
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map(line => {
    const values = parseCsvLine(line);
    const rec: Record<string, string> = {};
    headers.forEach((h, i) => { rec[h] = values[i] ?? ''; });
    return rec;
  });
}
