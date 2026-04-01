import { useRef, useState, useCallback, useMemo, useEffect } from 'react';
import { PointData } from './PointCloud';
import { Tooltip, TooltipData } from './Tooltip';

// Padding constant - outside component to avoid recreation
const PADDING = { left: 60, right: 30, top: 50, bottom: 60 } as const;

// Helper to determine if a color is dark
function isDarkColor(hexColor: string): boolean {
  // Remove # if present
  const hex = hexColor.replace('#', '');
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  // Calculate relative luminance
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance < 0.5;
}

export interface TrajectoryPlotProps {
  /** PC1 values for each point on X-axis: Map<xValue, Float32Array of PC1 values per sample> */
  trajectoryData: Map<string, Float32Array>;
  /** X-axis values in order */
  xValues: string[];
  /** X-axis label */
  xAxisLabel?: string;
  /** Y-axis label */
  yAxisLabel?: string;
  /** Title for the plot */
  title?: string;
  /** Colors for each sample (RGB triplets) */
  colors: Float32Array;
  /** Point data for tooltips */
  pointData?: PointData[];
  /** Background color */
  backgroundColor?: string;
  /** Show grid lines */
  showGrid?: boolean;
  /** Callback when hovering a sample */
  onPointHover?: (index: number | null, data: PointData | null) => void;
  /** Callback when selecting a sample */
  onPointSelect?: (index: number | null, data: PointData | null) => void;
  /** Currently selected sample index */
  selectedSampleIdx?: number | null;
  /** Line opacity (0-1) */
  lineOpacity?: number;
  /** Whether data is still loading */
  isLoading?: boolean;
  /** Loading progress (0-1) */
  loadingProgress?: number;
  /** Optional filter mask - true values are shown, false values are hidden */
  filterMask?: boolean[] | null;
  className?: string;
  style?: React.CSSProperties;
}

export function TrajectoryPlot({
  trajectoryData,
  xValues,
  xAxisLabel = 'X',
  yAxisLabel = 'Normalized PC1 projection',
  title = 'PC1 Trajectory',
  colors,
  pointData = [],
  backgroundColor = '#faf8f5',
  showGrid = true,
  onPointHover,
  onPointSelect,
  selectedSampleIdx,
  lineOpacity = 0.6,
  isLoading = false,
  loadingProgress = 0,
  filterMask = null,
  className,
  style,
}: TrajectoryPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [tooltipData, setTooltipData] = useState<TooltipData | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Convert selectedSampleIdx to array index
  const selectedIndex = useMemo(() => {
    if (selectedSampleIdx === null || selectedSampleIdx === undefined) {
      return null;
    }
    const index = pointData.findIndex(p => p.sampleIdx === selectedSampleIdx);
    return index === -1 ? null : index;
  }, [selectedSampleIdx, pointData]);

  // Calculate number of samples from first data point
  const numSamples = useMemo(() => {
    if (trajectoryData.size === 0) return 0;
    const firstData = trajectoryData.values().next().value;
    return firstData ? firstData.length : 0;
  }, [trajectoryData]);

  // Calculate visible sample count (respecting filter)
  const visibleSampleCount = useMemo(() => {
    if (!filterMask) return numSamples;
    return filterMask.filter(Boolean).length;
  }, [filterMask, numSamples]);

  // Calculate Y bounds across all x values
  const yBounds = useMemo(() => {
    let minY = Infinity;
    let maxY = -Infinity;

    trajectoryData.forEach((values) => {
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (isFinite(v)) {
          minY = Math.min(minY, v);
          maxY = Math.max(maxY, v);
        }
      }
    });

    if (!isFinite(minY)) minY = -2;
    if (!isFinite(maxY)) maxY = 2;

    // Add padding
    const pad = (maxY - minY) * 0.1 || 0.5;
    return { minY: minY - pad, maxY: maxY + pad };
  }, [trajectoryData]);

  // Get device pixel ratio for sharp rendering
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

  // Resize observer with proper initialization
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const updateSize = () => {
      const rect = container.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setDimensions({ width: rect.width, height: rect.height });
      }
    };

    // Use RAF to ensure layout is complete
    requestAnimationFrame(updateSize);

    const observer = new ResizeObserver(updateSize);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Transform coordinates
  const toCanvas = useCallback((xIdx: number, value: number) => {
    const plotWidth = dimensions.width - PADDING.left - PADDING.right;
    const plotHeight = dimensions.height - PADDING.top - PADDING.bottom;

    const x = PADDING.left + (xIdx / Math.max(1, xValues.length - 1)) * plotWidth;
    // Guard against division by zero
    const yRange = yBounds.maxY - yBounds.minY || 1;
    const y = PADDING.top + (1 - (value - yBounds.minY) / yRange) * plotHeight;

    return { x, y };
  }, [dimensions, xValues.length, yBounds]);

  // Draw the plot
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || dimensions.width === 0 || dimensions.height === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size with DPR for sharp rendering
    canvas.width = dimensions.width * dpr;
    canvas.height = dimensions.height * dpr;
    canvas.style.width = `${dimensions.width}px`;
    canvas.style.height = `${dimensions.height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const plotWidth = dimensions.width - PADDING.left - PADDING.right;
    const plotHeight = dimensions.height - PADDING.top - PADDING.bottom;

    // Determine colors based on background brightness
    const isDark = isDarkColor(backgroundColor);
    const textColor = isDark ? '#e0e0e0' : '#1a1613';
    const mutedTextColor = isDark ? '#a0a0a0' : '#7a6b8a';
    const gridColor = isDark ? 'rgba(100, 100, 120, 0.3)' : 'rgba(200, 180, 220, 0.3)';
    const axisColor = isDark ? '#808080' : '#7a6b8a';

    // Clear canvas
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, dimensions.width, dimensions.height);

    // Show loading state
    if (isLoading) {
      ctx.fillStyle = mutedTextColor;
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        `Loading... ${Math.round(loadingProgress * 100)}%`,
        dimensions.width / 2,
        dimensions.height / 2
      );
      return;
    }

    if (xValues.length === 0 || trajectoryData.size === 0) {
      ctx.fillStyle = mutedTextColor;
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No data available', dimensions.width / 2, dimensions.height / 2);
      return;
    }

    // Draw grid
    if (showGrid) {
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 1;

      // Horizontal grid lines
      const yTicks = 5;
      for (let i = 0; i <= yTicks; i++) {
        const y = PADDING.top + (i / yTicks) * plotHeight;
        ctx.beginPath();
        ctx.moveTo(PADDING.left, y);
        ctx.lineTo(dimensions.width - PADDING.right, y);
        ctx.stroke();
      }

      // Vertical grid lines at each x value
      xValues.forEach((_, idx) => {
        const { x } = toCanvas(idx, 0);
        ctx.beginPath();
        ctx.moveTo(x, PADDING.top);
        ctx.lineTo(x, dimensions.height - PADDING.bottom);
        ctx.stroke();
      });
    }

    // Draw axes
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1;

    // Y axis
    ctx.beginPath();
    ctx.moveTo(PADDING.left, PADDING.top);
    ctx.lineTo(PADDING.left, dimensions.height - PADDING.bottom);
    ctx.stroke();

    // X axis
    ctx.beginPath();
    ctx.moveTo(PADDING.left, dimensions.height - PADDING.bottom);
    ctx.lineTo(dimensions.width - PADDING.right, dimensions.height - PADDING.bottom);
    ctx.stroke();

    // Draw Y axis labels
    ctx.fillStyle = mutedTextColor;
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const yTicksLabel = 5;
    // Compute appropriate decimal places based on data range
    const yRange = yBounds.maxY - yBounds.minY || 1;
    const decimalsY = Math.max(0, Math.min(4, Math.ceil(-Math.log10(yRange / yTicksLabel)) + 1));

    for (let i = 0; i <= yTicksLabel; i++) {
      const value = yBounds.maxY - (i / yTicksLabel) * yRange;
      const y = PADDING.top + (i / yTicksLabel) * plotHeight;
      ctx.fillText(value.toFixed(decimalsY), PADDING.left - 8, y);
    }

    // Draw X axis labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.font = '9px monospace';

    // Only show some labels if too many
    const maxLabels = Math.floor(plotWidth / 50);
    const step = Math.max(1, Math.ceil(xValues.length / maxLabels));

    xValues.forEach((xVal, idx) => {
      if (idx % step !== 0 && idx !== xValues.length - 1) return;
      const { x } = toCanvas(idx, 0);
      ctx.fillText(xVal, x, dimensions.height - PADDING.bottom + 8);
    });

    // Draw axis titles
    ctx.font = '11px sans-serif';
    ctx.fillStyle = textColor;

    // X axis title
    ctx.textAlign = 'center';
    ctx.fillText(xAxisLabel, dimensions.width / 2, dimensions.height - 12);

    // Y axis title (rotated)
    ctx.save();
    ctx.translate(15, dimensions.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(yAxisLabel, 0, 0);
    ctx.restore();

    // Helper to get color with bounds check
    const getColor = (idx: number) => {
      if (idx * 3 + 2 >= colors.length) {
        return { r: 128, g: 128, b: 128 }; // Default gray
      }
      return {
        r: Math.round((colors[idx * 3] ?? 0.5) * 255),
        g: Math.round((colors[idx * 3 + 1] ?? 0.5) * 255),
        b: Math.round((colors[idx * 3 + 2] ?? 0.5) * 255),
      };
    };

    // Draw trajectories
    // First pass: draw non-selected, non-hovered lines
    for (let sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
      if (sampleIdx === selectedIndex || sampleIdx === hoveredIndex) continue;
      // Skip filtered samples
      if (filterMask && !filterMask[sampleIdx]) continue;

      const { r, g, b } = getColor(sampleIdx);

      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${lineOpacity})`;
      ctx.lineWidth = 1;

      ctx.beginPath();
      let started = false;

      xValues.forEach((xVal, idx) => {
        const data = trajectoryData.get(xVal);
        if (!data || sampleIdx >= data.length) return;

        const value = data[sampleIdx];
        if (!isFinite(value)) return;

        const { x, y } = toCanvas(idx, value);

        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();
    }

    // Second pass: draw hovered line (thicker)
    if (hoveredIndex !== null && hoveredIndex !== selectedIndex) {
      const sampleIdx = hoveredIndex;
      const { r, g, b } = getColor(sampleIdx);

      ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.lineWidth = 3;

      ctx.beginPath();
      let started = false;

      xValues.forEach((xVal, idx) => {
        const data = trajectoryData.get(xVal);
        if (!data || sampleIdx >= data.length) return;

        const value = data[sampleIdx];
        if (!isFinite(value)) return;

        const { x, y } = toCanvas(idx, value);

        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();
    }

    // Third pass: draw selected line (thickest, with glow)
    if (selectedIndex !== null) {
      const sampleIdx = selectedIndex;
      const { r, g, b } = getColor(sampleIdx);

      // Glow effect
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.3)`;
      ctx.lineWidth = 8;

      ctx.beginPath();
      let started = false;

      xValues.forEach((xVal, idx) => {
        const data = trajectoryData.get(xVal);
        if (!data || sampleIdx >= data.length) return;

        const value = data[sampleIdx];
        if (!isFinite(value)) return;

        const { x, y } = toCanvas(idx, value);

        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();

      // Main line
      ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.lineWidth = 3;

      ctx.beginPath();
      started = false;

      xValues.forEach((xVal, idx) => {
        const data = trajectoryData.get(xVal);
        if (!data || sampleIdx >= data.length) return;

        const value = data[sampleIdx];
        if (!isFinite(value)) return;

        const { x, y } = toCanvas(idx, value);

        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();

      // Draw points at each x value
      xValues.forEach((xVal, idx) => {
        const data = trajectoryData.get(xVal);
        if (!data || sampleIdx >= data.length) return;

        const value = data[sampleIdx];
        if (!isFinite(value)) return;

        const { x, y } = toCanvas(idx, value);

        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    }

  }, [trajectoryData, xValues, colors, dimensions, yBounds, toCanvas, showGrid, backgroundColor, numSamples, selectedIndex, hoveredIndex, lineOpacity, dpr, xAxisLabel, yAxisLabel, isLoading, loadingProgress, filterMask]);

  // Find sample at mouse position
  const findSampleAt = useCallback((canvasX: number, canvasY: number): number | null => {
    const hitRadius = 10;

    // Check each sample's trajectory
    for (let sampleIdx = numSamples - 1; sampleIdx >= 0; sampleIdx--) {
      for (let xIdx = 0; xIdx < xValues.length; xIdx++) {
        const xVal = xValues[xIdx];
        const data = trajectoryData.get(xVal);
        if (!data || sampleIdx >= data.length) continue;

        const value = data[sampleIdx];
        if (!isFinite(value)) continue;

        const { x, y } = toCanvas(xIdx, value);
        const dx = canvasX - x;
        const dy = canvasY - y;

        if (dx * dx + dy * dy <= hitRadius * hitRadius) {
          return sampleIdx;
        }
      }
    }

    return null;
  }, [numSamples, xValues, trajectoryData, toCanvas]);

  // Mouse handlers
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setMousePosition({ x: e.clientX, y: e.clientY });

    const index = findSampleAt(x, y);
    setHoveredIndex(index);

    if (index !== null && pointData[index]) {
      setTooltipVisible(true);
      setTooltipData({
        ...pointData[index],
        position: { x: 0, y: 0, z: 0 },
      });
      onPointHover?.(index, pointData[index]);
    } else {
      setTooltipVisible(false);
      setTooltipData(null);
      onPointHover?.(null, null);
    }
  }, [findSampleAt, pointData, onPointHover]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const index = findSampleAt(x, y);
    if (index !== null && pointData[index]) {
      onPointSelect?.(index, pointData[index]);
    } else {
      onPointSelect?.(null, null);
    }
  }, [findSampleAt, pointData, onPointSelect]);

  const handleMouseLeave = useCallback(() => {
    setTooltipVisible(false);
    setTooltipData(null);
    setHoveredIndex(null);
    onPointHover?.(null, null);
  }, [onPointHover]);

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        minHeight: '400px',
        borderRadius: '16px',
        overflow: 'hidden',
        ...style,
      }}
    >
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onClick={handleClick}
        onMouseLeave={handleMouseLeave}
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
          cursor: hoveredIndex !== null ? 'pointer' : 'default',
        }}
      />

      {/* Title */}
      <div
        style={{
          position: 'absolute',
          top: '8px',
          left: '50%',
          transform: 'translateX(-50%)',
          padding: '4px 12px',
          fontSize: '12px',
          fontWeight: 600,
          color: '#1a1613',
          background: 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(8px)',
          borderRadius: '8px',
          border: '1px solid rgba(180, 160, 200, 0.2)',
          zIndex: 10,
        }}
      >
        {title}
      </div>

      {/* Sample count */}
      <div
        style={{
          position: 'absolute',
          top: '12px',
          left: '12px',
          padding: '6px 12px',
          fontSize: '11px',
          fontWeight: 600,
          color: '#7a6b8a',
          background: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(8px)',
          borderRadius: '8px',
          border: '1px solid rgba(180, 160, 200, 0.2)',
          fontFamily: 'monospace',
          zIndex: 10,
        }}
      >
        {visibleSampleCount.toLocaleString()} samples
        {filterMask && visibleSampleCount !== numSamples && (
          <span style={{ color: '#9a8baa', fontWeight: 400 }}> / {numSamples.toLocaleString()}</span>
        )}
      </div>

      {/* Selected indicator */}
      {selectedSampleIdx !== null && selectedSampleIdx !== undefined && (
        <div
          style={{
            position: 'absolute',
            top: '12px',
            right: '12px',
            padding: '6px 12px',
            fontSize: '11px',
            fontWeight: 600,
            color: '#D97757',
            background: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(8px)',
            borderRadius: '8px',
            border: '1px solid rgba(198, 120, 221, 0.3)',
            fontFamily: 'monospace',
            zIndex: 10,
          }}
        >
          Selected: #{selectedSampleIdx}
        </div>
      )}

      {/* Tooltip */}
      <Tooltip
        data={tooltipData}
        mousePosition={mousePosition}
        visible={tooltipVisible}
      />
    </div>
  );
}

export default TrajectoryPlot;
