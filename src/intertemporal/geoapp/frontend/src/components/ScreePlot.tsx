import React, { useMemo } from 'react';

interface ScreeSeries {
  label: string;
  layer: number;
  component: string;
  values: number[];
}

interface ScreePlotProps {
  data: { series: ScreeSeries[] } | null;
  isLoading?: boolean;
  selectedLayer?: number;
  selectedComponent?: string;
  className?: string;
}

// Color palette for different layers
const LAYER_COLORS = [
  '#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3',
  '#03A9F4', '#00BCD4', '#009688', '#4CAF50', '#8BC34A',
  '#CDDC39', '#FFEB3B', '#FFC107', '#FF9800', '#FF5722',
];

// Colors for different components
const COMPONENT_COLORS: Record<string, string> = {
  'resid_pre': '#E91E63',   // Pink
  'attn_out': '#2196F3',    // Blue
  'mlp_out': '#4CAF50',     // Green
  'resid_post': '#FF9800',  // Orange
};

export const ScreePlot: React.FC<ScreePlotProps> = ({
  data,
  isLoading = false,
  selectedLayer,
  selectedComponent,
  className = '',
}) => {
  // Filter and prepare series for rendering
  const { filteredSeries, maxComponents, yMax } = useMemo(() => {
    if (!data?.series) return { filteredSeries: [], maxComponents: 10, yMax: 1 };

    // Filter to show only selected layer/component or top 8 key targets
    let series = data.series;
    if (selectedLayer !== undefined) {
      series = series.filter(s => s.layer === selectedLayer);
    }
    if (selectedComponent) {
      series = series.filter(s => s.component === selectedComponent);
    }

    // If no filter, take representative samples
    if (series.length > 10) {
      // Sample evenly across layers
      const step = Math.ceil(series.length / 10);
      series = series.filter((_, i) => i % step === 0).slice(0, 10);
    }

    const maxComp = Math.max(...series.map(s => s.values.length), 10);
    const maxY = Math.max(...series.flatMap(s => s.values), 1);

    return { filteredSeries: series, maxComponents: maxComp, yMax: maxY };
  }, [data, selectedLayer, selectedComponent]);

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-full bg-[#faf8f5] dark:bg-[#1a1613] ${className}`}>
        <div className="text-gray-500 dark:text-gray-400">Loading Scree data...</div>
      </div>
    );
  }

  if (!data?.series || filteredSeries.length === 0) {
    return (
      <div className={`flex items-center justify-center h-full bg-[#faf8f5] dark:bg-[#1a1613] ${className}`}>
        <div className="text-gray-500 dark:text-gray-400">No Scree data available</div>
      </div>
    );
  }

  const width = 800;
  const height = 500;
  const padding = { top: 50, right: 150, bottom: 60, left: 80 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const xScale = (i: number) => padding.left + (i / (maxComponents - 1)) * plotWidth;
  const yScale = (v: number) => padding.top + plotHeight - (v / yMax) * plotHeight;

  return (
    <div className={`flex items-center justify-center w-full h-full bg-[#faf8f5] dark:bg-[#1a1613] ${className}`}>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full" style={{ minWidth: '600px', minHeight: '400px' }}>
        {/* Title */}
        <text x={width / 2} y={25} textAnchor="middle" className="text-base font-semibold fill-gray-700 dark:fill-gray-200">
          Scree Plot: Cumulative Variance Explained
        </text>

        {/* Grid lines */}
        {[0.25, 0.5, 0.75, 1.0].map(v => (
          <line
            key={v}
            x1={padding.left}
            x2={width - padding.right}
            y1={yScale(v * yMax)}
            y2={yScale(v * yMax)}
            className="stroke-gray-200 dark:stroke-gray-700"
            strokeDasharray="4,4"
          />
        ))}

        {/* 90% threshold line */}
        <line
          x1={padding.left}
          x2={width - padding.right}
          y1={yScale(0.9)}
          y2={yScale(0.9)}
          className="stroke-gray-400 dark:stroke-gray-500"
          strokeDasharray="8,4"
          strokeWidth={1.5}
        />
        <text x={width - padding.right + 5} y={yScale(0.9) + 4} className="text-[10px] fill-gray-500 dark:fill-gray-400">
          90%
        </text>

        {/* X axis */}
        <line
          x1={padding.left}
          x2={width - padding.right}
          y1={height - padding.bottom}
          y2={height - padding.bottom}
          className="stroke-gray-600 dark:stroke-gray-400"
        />
        <text x={width / 2} y={height - 15} textAnchor="middle" className="text-sm fill-gray-600 dark:fill-gray-400">
          Number of Components
        </text>
        {Array.from({ length: maxComponents }, (_, i) => (
          <g key={i}>
            <line
              x1={xScale(i)}
              x2={xScale(i)}
              y1={height - padding.bottom}
              y2={height - padding.bottom + 5}
              className="stroke-gray-600 dark:stroke-gray-400"
            />
            <text x={xScale(i)} y={height - padding.bottom + 18} textAnchor="middle" className="text-[10px] fill-gray-600 dark:fill-gray-400">
              {i}
            </text>
          </g>
        ))}

        {/* Y axis */}
        <line
          x1={padding.left}
          x2={padding.left}
          y1={padding.top}
          y2={height - padding.bottom}
          className="stroke-gray-600 dark:stroke-gray-400"
        />
        <text
          x={20}
          y={height / 2}
          textAnchor="middle"
          transform={`rotate(-90, 20, ${height / 2})`}
          className="text-sm fill-gray-600 dark:fill-gray-400"
        >
          Cumulative Variance Explained
        </text>
        {[0, 0.25, 0.5, 0.75, 1.0].map(v => (
          <g key={v}>
            <line
              x1={padding.left - 5}
              x2={padding.left}
              y1={yScale(v * yMax)}
              y2={yScale(v * yMax)}
              className="stroke-gray-600 dark:stroke-gray-400"
            />
            <text x={padding.left - 8} y={yScale(v * yMax) + 4} textAnchor="end" className="text-[10px] fill-gray-600 dark:fill-gray-400">
              {(v * yMax).toFixed(2)}
            </text>
          </g>
        ))}

        {/* Plot lines */}
        {filteredSeries.map((series) => {
          // Use component color if available, fallback to layer color
          const color = COMPONENT_COLORS[series.component] || LAYER_COLORS[series.layer % LAYER_COLORS.length];
          const points = series.values.map((v, i) => `${xScale(i)},${yScale(v)}`).join(' ');

          return (
            <g key={series.label}>
              <polyline
                points={points}
                fill="none"
                stroke={color}
                strokeWidth={2}
                opacity={0.8}
              />
              {/* Points */}
              {series.values.map((v, i) => (
                <circle
                  key={i}
                  cx={xScale(i)}
                  cy={yScale(v)}
                  r={3}
                  fill={color}
                />
              ))}
            </g>
          );
        })}

        {/* Legend */}
        <g transform={`translate(${width - padding.right + 10}, ${padding.top})`}>
          {filteredSeries.slice(0, 8).map((series, idx) => {
            const color = COMPONENT_COLORS[series.component] || LAYER_COLORS[series.layer % LAYER_COLORS.length];
            return (
              <g key={series.label} transform={`translate(0, ${idx * 16})`}>
                <line x1={0} x2={20} y1={0} y2={0} stroke={color} strokeWidth={2} />
                <circle cx={10} cy={0} r={3} fill={color} />
                <text x={25} y={4} className="text-[11px] font-medium" fill={color}>
                  {series.label}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
};

export default ScreePlot;
