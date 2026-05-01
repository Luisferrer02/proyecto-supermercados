"use client";

/**
 * ShelfSankey — alluvial/sankey-style visualisation of every product
 * relocation produced by the optimisation pipeline.
 *
 * Two vertical columns of 7 shelves (origin → destination). Between
 * them we render curved SVG ribbons proportional to how many products
 * moved between each shelf pair.
 *
 * Design choices
 * --------------
 *  - Pure SVG, no 3rd-party chart lib. Sankey libs either pull the whole
 *    d3-sankey or are finicky with custom styling; 100 LOC of SVG gets
 *    us 80% of the look with none of the dependency cost.
 *  - Eye-level shelves (3–5) are highlighted in the accent colour so a
 *    non-technical user instantly sees where the premium real-estate is.
 *  - Ribbons between the SAME shelf (no move) are drawn grey and thin
 *    so the eye focuses on actual relocations.
 */

interface Movement { from: number; to: number; count: number; }

interface Props {
  movements: Movement[];
  width?: number;
  height?: number;
}

const SHELVES = [1, 2, 3, 4, 5, 6, 7];
const EYE_LEVEL = new Set([3, 4, 5]);

export function ShelfSankey({ movements, width = 640, height = 420 }: Props) {
  // Aggregate counts per shelf so we can size bars proportionally
  const totalPerFrom: Record<number, number> = {};
  const totalPerTo:   Record<number, number> = {};
  let totalMovements = 0;
  for (const m of movements) {
    totalPerFrom[m.from] = (totalPerFrom[m.from] ?? 0) + m.count;
    totalPerTo[m.to]     = (totalPerTo[m.to]     ?? 0) + m.count;
    totalMovements += m.count;
  }

  const maxCount = Math.max(...SHELVES.map(s =>
    Math.max(totalPerFrom[s] ?? 0, totalPerTo[s] ?? 0)));
  if (totalMovements === 0 || maxCount === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-8">
        No hay reubicaciones para mostrar todavía.
      </div>
    );
  }

  const barWidth = 36;
  const barHeight = (height - 40) / SHELVES.length;
  const leftX = 60;
  const rightX = width - leftX - barWidth;
  const topY = 20;
  const barInner = barHeight * 0.82;

  const shelfY = (shelf: number) => topY + (shelf - 1) * barHeight + (barHeight - barInner) / 2;

  // Ribbon paths — order: draw "no move" first (thin grey), then the
  // real moves on top with colour so they stand out.
  const noMoves = movements.filter(m => m.from === m.to);
  const realMoves = movements.filter(m => m.from !== m.to)
    .sort((a, b) => a.count - b.count);  // smallest first, biggest on top

  const colourForMove = (m: Movement): string => {
    if (m.from === m.to) return "rgba(100,100,100,0.18)";
    // Upward moves (better placement) → green-ish
    // Downward moves (worse placement) → amber
    // Eye-level target moves → strong accent
    if (EYE_LEVEL.has(m.to) && !EYE_LEVEL.has(m.from)) return "rgba(9, 84, 61, 0.55)";
    if (EYE_LEVEL.has(m.from) && !EYE_LEVEL.has(m.to)) return "rgba(244, 92, 36, 0.45)";
    if (m.to < m.from) return "rgba(9, 84, 61, 0.35)";     // promoted
    return "rgba(200, 135, 40, 0.35)";                      // demoted
  };

  // Layout within each bar: stack sub-segments proportional to counts so
  // ribbons attach at distinct vertical positions.
  const fromOffset: Record<number, number> = {};
  const toOffset: Record<number, number> = {};
  SHELVES.forEach(s => { fromOffset[s] = 0; toOffset[s] = 0; });

  const ribbonFor = (m: Movement) => {
    const fromTotal = totalPerFrom[m.from] ?? 0;
    const toTotal = totalPerTo[m.to] ?? 0;
    const fromThickness = fromTotal > 0 ? (m.count / fromTotal) * barInner : 0;
    const toThickness = toTotal > 0 ? (m.count / toTotal) * barInner : 0;

    const y1a = shelfY(m.from) + fromOffset[m.from];
    const y1b = y1a + fromThickness;
    fromOffset[m.from] += fromThickness;

    const y2a = shelfY(m.to) + toOffset[m.to];
    const y2b = y2a + toThickness;
    toOffset[m.to] += toThickness;

    const x1 = leftX + barWidth;
    const x2 = rightX;
    const cx1 = x1 + (x2 - x1) * 0.45;
    const cx2 = x1 + (x2 - x1) * 0.55;

    return `M ${x1} ${y1a} C ${cx1} ${y1a}, ${cx2} ${y2a}, ${x2} ${y2a} L ${x2} ${y2b} C ${cx2} ${y2b}, ${cx1} ${y1b}, ${x1} ${y1b} Z`;
  };

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
      {/* Ribbons (in order: no-move → real moves) */}
      <g>
        {[...noMoves, ...realMoves].map((m, i) => (
          <path
            key={i}
            d={ribbonFor(m)}
            fill={colourForMove(m)}
            stroke="none"
          >
            <title>
              Balda {m.from} → Balda {m.to}: {m.count} productos
            </title>
          </path>
        ))}
      </g>

      {/* Left column: origin shelves */}
      <g>
        {SHELVES.map(s => {
          const y = shelfY(s);
          const count = totalPerFrom[s] ?? 0;
          const isEye = EYE_LEVEL.has(s);
          return (
            <g key={`L-${s}`}>
              <rect
                x={leftX}
                y={y}
                width={barWidth}
                height={barInner}
                rx={3}
                fill={isEye ? "var(--color-primary, #09543d)" : "#a8a095"}
                opacity={count > 0 ? 0.9 : 0.25}
              />
              <text x={leftX - 8} y={y + barInner / 2 + 4} textAnchor="end"
                    fontSize="12" fill="currentColor" fontWeight={isEye ? 600 : 400}>
                Balda {s}{isEye ? " (ojos)" : ""}
              </text>
              <text x={leftX + barWidth / 2} y={y + barInner / 2 + 4}
                    textAnchor="middle" fontSize="10" fill="white">
                {count}
              </text>
            </g>
          );
        })}
      </g>

      {/* Right column: destination shelves */}
      <g>
        {SHELVES.map(s => {
          const y = shelfY(s);
          const count = totalPerTo[s] ?? 0;
          const isEye = EYE_LEVEL.has(s);
          return (
            <g key={`R-${s}`}>
              <rect
                x={rightX}
                y={y}
                width={barWidth}
                height={barInner}
                rx={3}
                fill={isEye ? "var(--color-primary, #09543d)" : "#a8a095"}
                opacity={count > 0 ? 0.9 : 0.25}
              />
              <text x={rightX + barWidth + 8} y={y + barInner / 2 + 4}
                    textAnchor="start"
                    fontSize="12" fill="currentColor" fontWeight={isEye ? 600 : 400}>
                Balda {s}{isEye ? " (ojos)" : ""}
              </text>
              <text x={rightX + barWidth / 2} y={y + barInner / 2 + 4}
                    textAnchor="middle" fontSize="10" fill="white">
                {count}
              </text>
            </g>
          );
        })}
      </g>

      {/* Column headers */}
      <text x={leftX + barWidth / 2} y={height - 4} textAnchor="middle"
            fontSize="11" fill="currentColor" opacity={0.6}>
        Antes
      </text>
      <text x={rightX + barWidth / 2} y={height - 4} textAnchor="middle"
            fontSize="11" fill="currentColor" opacity={0.6}>
        Después
      </text>
    </svg>
  );
}
