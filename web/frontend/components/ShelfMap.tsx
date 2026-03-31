"use client";

import { useState } from "react";

export interface Product {
  name?: string;
  Category?: string;
  shelf_level?: string;
  rack_id?: string;
  product_width_cm?: string;
  price_numeric?: string;
  profit_margin_percentage?: string;
  estimated_monthly_sales?: string;
  [key: string]: string | undefined;
}

interface Props {
  products: Product[];
  rackId: string;
}

const SHELF_SCALE = 0.72; // px per cm (300cm * 0.72 = 216px)
const EYE_LEVEL = new Set([3, 4, 5]);

function profitColor(p: Product): string {
  const price = parseFloat(p.price_numeric || "0");
  const margin = parseFloat(p.profit_margin_percentage || "0") / 100;
  const sales = parseFloat(p.estimated_monthly_sales || "0");
  const profit = price * margin * sales;
  if (profit > 500) return "#ef4444";
  if (profit > 200) return "#f97316";
  if (profit > 80) return "#eab308";
  return "#22c55e";
}

export function ShelfMap({ products, rackId }: Props) {
  const [hovered, setHovered] = useState<Product | null>(null);
  const shelves = [7, 6, 5, 4, 3, 2, 1]; // top to bottom

  const shelfWidth = 300 * SHELF_SCALE;

  return (
    <div className="space-y-3">
      {/* Legend */}
      <div className="flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-red-500 inline-block" /> High profit</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-orange-500 inline-block" /> Medium-high</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-yellow-500 inline-block" /> Medium</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-green-500 inline-block" /> Low profit</span>
        <span className="flex items-center gap-1 ml-2">★ Eye-level shelf</span>
      </div>

      <div className="overflow-x-auto pb-2">
        <div className="space-y-1" style={{ minWidth: shelfWidth + 120 }}>
          {shelves.map((shelfNum) => {
            const shelfProducts = products.filter(
              (p) => parseInt(p.shelf_level || "0") === shelfNum
            );
            const isEye = EYE_LEVEL.has(shelfNum);
            let xOffset = 0;

            return (
              <div key={shelfNum} className="flex items-center gap-2">
                {/* Label */}
                <div
                  className={`w-20 text-right text-xs shrink-0 ${
                    isEye ? "text-amber-600 font-semibold" : "text-muted-foreground"
                  }`}
                >
                  {isEye ? `★ Shelf ${shelfNum}` : `Shelf ${shelfNum}`}
                </div>

                {/* Shelf bar */}
                <div
                  className={`relative border-b-2 ${
                    isEye ? "border-amber-500/60 bg-amber-500/5" : "border-border bg-card"
                  } rounded-sm`}
                  style={{ width: shelfWidth, height: 48 }}
                >
                  {shelfProducts.map((p, i) => {
                    const w = Math.max(
                      parseFloat(p.product_width_cm || "10") * SHELF_SCALE,
                      10
                    );
                    const left = xOffset;
                    xOffset += w + 2;
                    const color = profitColor(p);
                    return (
                      <div
                        key={i}
                        className="absolute bottom-0 rounded-sm cursor-pointer transition-opacity hover:opacity-80"
                        style={{
                          left,
                          width: w,
                          height: 40,
                          backgroundColor: color,
                          opacity: 0.85,
                        }}
                        onMouseEnter={() => setHovered(p)}
                        onMouseLeave={() => setHovered(null)}
                      />
                    );
                  })}
                </div>

                {/* Count */}
                <span className="text-xs text-muted-foreground shrink-0">
                  {shelfProducts.length}
                </span>
              </div>
            );
          })}
        </div>

        {/* 300cm ruler */}
        <div className="flex items-center gap-2 mt-1">
          <div className="w-20" />
          <div
            className="flex justify-between text-xs text-muted-foreground"
            style={{ width: shelfWidth }}
          >
            <span>0 cm</span>
            <span>150 cm</span>
            <span>300 cm</span>
          </div>
        </div>
      </div>

      {/* Hover tooltip */}
      {hovered && (
        <div className="rounded-lg border border-border bg-card p-3 text-xs space-y-1 shadow-sm">
          <div className="font-semibold text-sm">{hovered.name || "—"}</div>
          <div className="text-muted-foreground">Category: {hovered.Category || rackId}</div>
          <div className="grid grid-cols-3 gap-2 mt-1">
            <div>
              <div className="text-muted-foreground">Price</div>
              <div>€{parseFloat(hovered.price_numeric || "0").toFixed(2)}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Margin</div>
              <div>{hovered.profit_margin_percentage || "—"}%</div>
            </div>
            <div>
              <div className="text-muted-foreground">Sales/mo</div>
              <div>{hovered.estimated_monthly_sales || "—"}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
