import React, { useMemo } from 'react';

interface HeatmapCell {
  layer: number;
  position: string;
  value: number | null;
}

interface HeatmapData {
  metric: string;
  component: string;
  layers: number[];
  positions: string[];
  cells: HeatmapCell[];
  min_value: number | null;
  max_value: number | null;
}

export interface HeatmapProps {
  data: HeatmapData | null;
  isLoading?: boolean;
  onCellClick?: (layer: number, position: string) => void;
  selectedCell?: { layer: number; position: string } | null;
  className?: string;
}

// Plasma colormap for heatmap
function getPlasmaColor(t: number): string {
  const stops = [
    [0.050383, 0.029803, 0.527975],
    [0.417642, 0.000564, 0.658390],
    [0.692840, 0.165141, 0.564522],
    [0.881443, 0.392529, 0.383229],
    [0.988260, 0.652325, 0.211364],
    [0.940015, 0.975158, 0.131326],
  ];

  t = Math.max(0, Math.min(1, t));
  const idx = t * (stops.length - 1);
  const i = Math.floor(idx);
  const f = idx - i;

  if (i >= stops.length - 1) {
    const c = stops[stops.length - 1];
    return `rgb(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)})`;
  }

  const c0 = stops[i];
  const c1 = stops[i + 1];
  const r = c0[0] + f * (c1[0] - c0[0]);
  const g = c0[1] + f * (c1[1] - c0[1]);
  const b = c0[2] + f * (c1[2] - c0[2]);

  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

export const Heatmap: React.FC<HeatmapProps> = ({
  data,
  isLoading = false,
  onCellClick,
  selectedCell,
  className = '',
}) => {
  // Build cell lookup for quick access
  const cellMap = useMemo(() => {
    if (!data) return new Map<string, number | null>();
    const map = new Map<string, number | null>();
    data.cells.forEach((cell) => {
      map.set(`${cell.layer}-${cell.position}`, cell.value);
    });
    return map;
  }, [data]);

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-48 ${className}`}>
        <div className="w-6 h-6 border-2 border-[#D97757] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!data || data.cells.length === 0 || data.layers.length === 0 || data.positions.length === 0) {
    return (
      <div className={`flex items-center justify-center h-48 text-gray-400 ${className}`}>
        No data available
      </div>
    );
  }

  const { layers, positions, min_value, max_value } = data;
  const range = (max_value ?? 1) - (min_value ?? 0) || 1;

  // Shorten position labels for display
  const formatPosition = (pos: string) => {
    if (pos.startsWith('P') && /^P\d+$/.test(pos)) {
      return pos.slice(1); // Just the number
    }
    // Abbreviate long names
    return pos
      .replace('long_term_', 'LT_')
      .replace('short_term_', 'ST_')
      .replace('_reward', '_R')
      .replace('_timestamp', '_T')
      .replace('response', 'resp');
  };

  return (
    <div className={`overflow-auto ${className}`}>
      <div className="inline-block min-w-full">
        {/* Column headers (positions) */}
        <div className="flex">
          <div className="w-12 h-6 flex-shrink-0" /> {/* Corner spacer */}
          {positions.map((pos) => (
            <div
              key={pos}
              className="w-8 h-6 flex-shrink-0 text-[8px] text-gray-500 flex items-end justify-center pb-0.5 transform -rotate-45 origin-bottom-left ml-1"
              title={pos}
            >
              {formatPosition(pos)}
            </div>
          ))}
        </div>

        {/* Rows */}
        {layers.map((layer) => (
          <div key={layer} className="flex">
            {/* Row label (layer) */}
            <div className="w-12 h-6 flex-shrink-0 text-[10px] text-gray-500 flex items-center justify-end pr-1">
              L{layer}
            </div>

            {/* Cells */}
            {positions.map((pos) => {
              const value = cellMap.get(`${layer}-${pos}`) ?? null;
              const isSelected =
                selectedCell?.layer === layer && selectedCell?.position === pos;
              const t = value !== null ? (value - (min_value ?? 0)) / range : 0;
              const bgColor = value !== null ? getPlasmaColor(t) : '#e5e5e5';

              return (
                <div
                  key={`${layer}-${pos}`}
                  className={`w-8 h-6 flex-shrink-0 cursor-pointer transition-all duration-150 ${
                    isSelected ? 'ring-2 ring-black ring-offset-1 ring-offset-white' : ''
                  }`}
                  style={{ backgroundColor: bgColor }}
                  onClick={() => onCellClick?.(layer, pos)}
                  title={`L${layer} ${pos}: ${value !== null ? value.toFixed(3) : 'N/A'}`}
                />
              );
            })}
          </div>
        ))}

        {/* Color scale legend */}
        <div className="flex items-center mt-3 ml-12">
          <span className="text-[10px] text-gray-500 mr-1">
            {min_value !== null && min_value !== undefined ? min_value.toFixed(2) : '0'}
          </span>
          <div
            className="h-3 w-24 rounded"
            style={{
              // Plasma colormap matching getPlasmaColor() function
              background: 'linear-gradient(to right, #0d0887, #6a00a8, #b12a90, #e16462, #fca636, #f0f921)',
            }}
          />
          <span className="text-[10px] text-gray-500 ml-1">
            {max_value !== null && max_value !== undefined ? max_value.toFixed(2) : '1'}
          </span>
          <span className="text-[10px] text-gray-400 ml-2">
            {data.metric === 'r2' ? 'R²' : data.metric}
          </span>
        </div>
      </div>
    </div>
  );
};

export default Heatmap;
