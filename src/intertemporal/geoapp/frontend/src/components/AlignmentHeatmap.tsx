import React, { useMemo } from 'react';

interface AlignmentData {
  labels: string[];
  matrix: number[][];
  n_targets: number;
  error?: string;
}

interface AlignmentHeatmapProps {
  data: AlignmentData | null;
  isLoading?: boolean;
  position?: string;
  className?: string;
}

// Component colors (matching ScreePlot)
const COMPONENT_COLORS: Record<string, string> = {
  'resid_pre': '#E91E63',   // Pink
  'attn_out': '#2196F3',    // Blue
  'mlp_out': '#4CAF50',     // Green
  'resid_post': '#FF9800',  // Orange
};

// Get component from label like "L0_resid_pre"
function getComponentFromLabel(label: string): string {
  const parts = label.split('_');
  if (parts.length >= 2) {
    // Handle cases like "L0_resid_pre" -> "resid_pre" or "L0_attn_out" -> "attn_out"
    return parts.slice(1).join('_');
  }
  return '';
}

// Viridis-like color scale
function getColor(value: number): string {
  // Map 0-1 to viridis-like colors
  const colors = [
    [68, 1, 84],      // 0.0 - dark purple
    [72, 40, 120],    // 0.2
    [62, 74, 137],    // 0.3
    [49, 104, 142],   // 0.4
    [38, 130, 142],   // 0.5
    [31, 158, 137],   // 0.6
    [53, 183, 121],   // 0.7
    [109, 205, 89],   // 0.8
    [180, 222, 44],   // 0.9
    [253, 231, 37],   // 1.0 - yellow
  ];

  const idx = Math.min(Math.floor(value * (colors.length - 1)), colors.length - 2);
  const t = (value * (colors.length - 1)) - idx;
  const c1 = colors[idx];
  const c2 = colors[idx + 1];

  const r = Math.round(c1[0] + t * (c2[0] - c1[0]));
  const g = Math.round(c1[1] + t * (c2[1] - c1[1]));
  const b = Math.round(c1[2] + t * (c2[2] - c1[2]));

  return `rgb(${r}, ${g}, ${b})`;
}

export const AlignmentHeatmap: React.FC<AlignmentHeatmapProps> = ({
  data,
  isLoading = false,
  position = '',
  className = '',
}) => {
  const { cellSize, width, height } = useMemo(() => {
    if (!data?.matrix) return { cellSize: 14, width: 600, height: 600 };
    const n = data.matrix.length;
    // Bigger cells: min 12px, max 24px, target ~700px total
    const cs = Math.min(Math.max(700 / n, 12), 24);
    return {
      cellSize: cs,
      width: n * cs + 150,  // More room for labels
      height: n * cs + 140, // More room for rotated x-labels
    };
  }, [data]);

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-full bg-[#faf8f5] dark:bg-[#1a1613] ${className}`}>
        <div className="text-gray-500 dark:text-gray-400">Loading alignment data...</div>
      </div>
    );
  }

  if (!data?.matrix || data.matrix.length === 0) {
    return (
      <div className={`flex items-center justify-center h-full bg-[#faf8f5] dark:bg-[#1a1613] ${className}`}>
        <div className="text-gray-500 dark:text-gray-400">
          {data?.error || 'No alignment data available'}
        </div>
      </div>
    );
  }

  const n = data.matrix.length;
  const padding = { top: 40, right: 70, bottom: 80, left: 80 };

  return (
    <div className={`flex items-center justify-center w-full h-full bg-[#faf8f5] dark:bg-[#1a1613] ${className}`}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-full max-w-full max-h-full bg-[#faf8f5] dark:bg-[#1a1613]"
        style={{ minWidth: '600px', minHeight: '600px' }}
      >
        {/* Title */}
        <text
          x={width / 2}
          y={15}
          textAnchor="middle"
          className="text-xs font-semibold fill-gray-700 dark:fill-gray-200"
        >
          Direction Alignment: Position={position}
        </text>

        {/* Heatmap cells */}
        <g transform={`translate(${padding.left}, ${padding.top})`}>
          {data.matrix.map((row, i) =>
            row.map((value, j) => (
              <rect
                key={`${i}-${j}`}
                x={j * cellSize}
                y={i * cellSize}
                width={cellSize}
                height={cellSize}
                fill={getColor(Math.abs(value))}
                className="stroke-white/50 dark:stroke-black/30"
                strokeWidth={0.5}
              >
                <title>{`${data.labels[i]} vs ${data.labels[j]}: ${value.toFixed(3)}`}</title>
              </rect>
            ))
          )}
        </g>

        {/* Y-axis labels */}
        <g transform={`translate(${padding.left - 5}, ${padding.top})`}>
          {data.labels.map((label, i) => {
            const comp = getComponentFromLabel(label);
            const color = COMPONENT_COLORS[comp] || '#666';
            return (
              <text
                key={i}
                x={0}
                y={i * cellSize + cellSize / 2}
                textAnchor="end"
                dominantBaseline="middle"
                className="text-[9px] font-medium"
                fill={color}
              >
                {label}
              </text>
            );
          })}
        </g>

        {/* X-axis labels */}
        <g transform={`translate(${padding.left}, ${padding.top + n * cellSize + 5})`}>
          {data.labels.map((label, i) => {
            const comp = getComponentFromLabel(label);
            const color = COMPONENT_COLORS[comp] || '#666';
            return (
              <text
                key={i}
                x={i * cellSize + cellSize / 2}
                y={0}
                textAnchor="start"
                transform={`rotate(45, ${i * cellSize + cellSize / 2}, 0)`}
                className="text-[9px] font-medium"
                fill={color}
              >
                {label}
              </text>
            );
          })}
        </g>

        {/* Color bar */}
        <g transform={`translate(${width - 50}, ${padding.top})`}>
          <defs>
            <linearGradient id="colorbar" x1="0" y1="1" x2="0" y2="0">
              {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((v, i) => (
                <stop key={i} offset={`${v * 100}%`} stopColor={getColor(v)} />
              ))}
            </linearGradient>
          </defs>
          <rect
            x={0}
            y={0}
            width={15}
            height={n * cellSize}
            fill="url(#colorbar)"
            className="stroke-gray-400 dark:stroke-gray-600"
            strokeWidth={0.5}
          />
          {/* Color bar labels */}
          {[0, 0.5, 1.0].map(v => (
            <text
              key={v}
              x={20}
              y={n * cellSize * (1 - v)}
              dominantBaseline="middle"
              className="text-[8px] fill-gray-600 dark:fill-gray-400"
            >
              {v.toFixed(1)}
            </text>
          ))}
          <text
            x={20}
            y={n * cellSize + 15}
            className="text-[8px] fill-gray-600 dark:fill-gray-400"
          >
            cos sim
          </text>
        </g>
      </svg>
    </div>
  );
};

export default AlignmentHeatmap;
