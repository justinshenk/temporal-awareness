import { useRef, useState, useCallback, useMemo, useEffect, memo } from 'react';
import { PointData } from './PointCloud';
import { Tooltip, TooltipData } from './Tooltip';

export interface ScatterPlot2DProps {
  positions: Float32Array;
  colors: Float32Array;
  pointData?: PointData[];
  pointSize?: number;
  backgroundColor?: string;
  showAxes?: boolean;
  showGrid?: boolean;
  onPointHover?: (index: number | null, data: PointData | null) => void;
  onPointSelect?: (index: number | null, data: PointData | null) => void;
  className?: string;
  style?: React.CSSProperties;
  selectedSampleIdx?: number | null;
  xAxis?: number; // Which component for X axis (0, 1, or 2)
  yAxis?: number; // Which component for Y axis (0, 1, or 2)
  /** Visibility mask - 1.0 for visible, 0.0 for hidden */
  visibility?: Float32Array;
}

function ScatterPlot2DInner({
  positions,
  colors,
  pointData = [],
  pointSize = 6,
  backgroundColor = '#faf8f5',
  showAxes = true,
  showGrid = true,
  onPointHover,
  onPointSelect,
  className,
  style,
  selectedSampleIdx,
  xAxis = 0,
  yAxis = 1,
  visibility,
}: ScatterPlot2DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [tooltipData, setTooltipData] = useState<TooltipData | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [transform, setTransform] = useState({ scale: 1, offsetX: 0, offsetY: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  // Get device pixel ratio for sharp rendering
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

  // Convert selectedSampleIdx to array index
  const selectedIndex = useMemo(() => {
    if (selectedSampleIdx === null || selectedSampleIdx === undefined) {
      return null;
    }
    const index = pointData.findIndex(p => p.sampleIdx === selectedSampleIdx);
    return index === -1 ? null : index;
  }, [selectedSampleIdx, pointData]);

  // Calculate data bounds
  const bounds = useMemo(() => {
    if (positions.length === 0) {
      return { minX: -5, maxX: 5, minY: -5, maxY: 5 };
    }

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i + xAxis];
      const y = positions[i + yAxis];
      if (isFinite(x) && isFinite(y)) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }

    // Handle edge cases
    if (!isFinite(minX)) { minX = -5; maxX = 5; }
    if (!isFinite(minY)) { minY = -5; maxY = 5; }

    // Add padding (10%)
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const padX = rangeX * 0.1;
    const padY = rangeY * 0.1;

    return {
      minX: minX - padX,
      maxX: maxX + padX,
      minY: minY - padY,
      maxY: maxY + padY,
    };
  }, [positions, xAxis, yAxis]);

  // Resize observer with proper initialization
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const updateSize = () => {
      const rect = container.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setDimensions({
          width: rect.width,
          height: rect.height,
        });
      }
    };

    // Initial size
    updateSize();

    const observer = new ResizeObserver(updateSize);
    observer.observe(container);

    return () => observer.disconnect();
  }, []);

  // Padding for axes labels
  const padding = { left: 60, right: 30, top: 30, bottom: 50 };

  // Transform data coordinates to canvas coordinates
  const toCanvas = useCallback((dataX: number, dataY: number) => {
    const { minX, maxX, minY, maxY } = bounds;
    const plotWidth = dimensions.width - padding.left - padding.right;
    const plotHeight = dimensions.height - padding.top - padding.bottom;

    if (plotWidth <= 0 || plotHeight <= 0) {
      return { x: 0, y: 0 };
    }

    // Map data to plot area
    const normalizedX = (dataX - minX) / (maxX - minX);
    const normalizedY = (dataY - minY) / (maxY - minY);

    let x = padding.left + normalizedX * plotWidth;
    let y = padding.top + (1 - normalizedY) * plotHeight; // Flip Y

    // Apply zoom and pan centered on plot center
    const centerX = padding.left + plotWidth / 2;
    const centerY = padding.top + plotHeight / 2;
    x = centerX + (x - centerX) * transform.scale + transform.offsetX;
    y = centerY + (y - centerY) * transform.scale + transform.offsetY;

    return { x, y };
  }, [bounds, dimensions, transform, padding]);

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
    ctx.scale(dpr, dpr);

    const plotWidth = dimensions.width - padding.left - padding.right;
    const plotHeight = dimensions.height - padding.top - padding.bottom;

    // Clear canvas
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, dimensions.width, dimensions.height);

    if (plotWidth <= 0 || plotHeight <= 0) return;

    // Draw plot area background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.fillRect(padding.left, padding.top, plotWidth, plotHeight);

    // Draw grid
    if (showGrid) {
      ctx.strokeStyle = 'rgba(180, 160, 200, 0.25)';
      ctx.lineWidth = 1;

      const gridLines = 8;
      const { minX, maxX, minY, maxY } = bounds;
      const stepX = (maxX - minX) / gridLines;
      const stepY = (maxY - minY) / gridLines;

      for (let i = 0; i <= gridLines; i++) {
        // Vertical lines
        const vStart = toCanvas(minX + stepX * i, minY);
        const vEnd = toCanvas(minX + stepX * i, maxY);
        ctx.beginPath();
        ctx.moveTo(vStart.x, vStart.y);
        ctx.lineTo(vEnd.x, vEnd.y);
        ctx.stroke();

        // Horizontal lines
        const hStart = toCanvas(minX, minY + stepY * i);
        const hEnd = toCanvas(maxX, minY + stepY * i);
        ctx.beginPath();
        ctx.moveTo(hStart.x, hStart.y);
        ctx.lineTo(hEnd.x, hEnd.y);
        ctx.stroke();
      }
    }

    // Draw axes through origin if visible
    if (showAxes) {
      const origin = toCanvas(0, 0);
      const isOriginVisible =
        origin.x >= padding.left && origin.x <= dimensions.width - padding.right &&
        origin.y >= padding.top && origin.y <= dimensions.height - padding.bottom;

      if (isOriginVisible) {
        ctx.strokeStyle = 'rgba(198, 120, 221, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 5]);

        // X axis through origin
        const xStart = toCanvas(bounds.minX, 0);
        const xEnd = toCanvas(bounds.maxX, 0);
        ctx.beginPath();
        ctx.moveTo(Math.max(padding.left, xStart.x), origin.y);
        ctx.lineTo(Math.min(dimensions.width - padding.right, xEnd.x), origin.y);
        ctx.stroke();

        // Y axis through origin
        const yStart = toCanvas(0, bounds.minY);
        const yEnd = toCanvas(0, bounds.maxY);
        ctx.beginPath();
        ctx.moveTo(origin.x, Math.max(padding.top, yEnd.y));
        ctx.lineTo(origin.x, Math.min(dimensions.height - padding.bottom, yStart.y));
        ctx.stroke();

        ctx.setLineDash([]);
      }
    }

    // Draw plot border
    ctx.strokeStyle = 'rgba(180, 160, 200, 0.5)';
    ctx.lineWidth = 1;
    ctx.strokeRect(padding.left, padding.top, plotWidth, plotHeight);

    // Draw points (non-selected first, then hovered, then selected)
    const numPoints = positions.length / 3;
    // Base alpha for points - slightly transparent for better overlap visualization
    const baseAlpha = numPoints > 1000 ? 0.7 : numPoints > 500 ? 0.8 : 0.9;

    const drawPoint = (i: number, sizeMultiplier: number = 1, strokeColor?: string, strokeWidth?: number, alpha: number = baseAlpha) => {
      const dataX = positions[i * 3 + xAxis];
      const dataY = positions[i * 3 + yAxis];

      if (!isFinite(dataX) || !isFinite(dataY)) return;

      const { x, y } = toCanvas(dataX, dataY);

      // Skip if outside plot area
      if (x < padding.left - 10 || x > dimensions.width - padding.right + 10 ||
          y < padding.top - 10 || y > dimensions.height - padding.bottom + 10) {
        return;
      }

      const r = Math.round(colors[i * 3] * 255);
      const g = Math.round(colors[i * 3 + 1] * 255);
      const b = Math.round(colors[i * 3 + 2] * 255);

      const size = pointSize * sizeMultiplier * Math.min(1.5, Math.max(0.5, transform.scale * 0.8));

      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      ctx.fill();

      if (strokeColor) {
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = strokeWidth || 2;
        ctx.stroke();
      }
    };

    // Helper to check if point is visible
    const isVisible = (i: number) => !visibility || visibility[i] >= 0.5;

    // Draw non-selected, non-hovered points
    for (let i = 0; i < numPoints; i++) {
      if (i === selectedIndex || i === hoveredIndex) continue;
      if (!isVisible(i)) continue;
      drawPoint(i, 1, undefined, undefined, baseAlpha);
    }

    // Draw hovered point with full opacity
    if (hoveredIndex !== null && hoveredIndex !== selectedIndex && isVisible(hoveredIndex)) {
      drawPoint(hoveredIndex, 1.4, 'rgba(255, 255, 255, 0.8)', 2, 1.0);
    }

    // Draw selected point with glow
    if (selectedIndex !== null && isVisible(selectedIndex)) {
      const dataX = positions[selectedIndex * 3 + xAxis];
      const dataY = positions[selectedIndex * 3 + yAxis];
      if (isFinite(dataX) && isFinite(dataY)) {
        const { x, y } = toCanvas(dataX, dataY);
        const r = Math.round(colors[selectedIndex * 3] * 255);
        const g = Math.round(colors[selectedIndex * 3 + 1] * 255);
        const b = Math.round(colors[selectedIndex * 3 + 2] * 255);
        const size = pointSize * 1.6 * Math.min(1.5, Math.max(0.5, transform.scale * 0.8));

        // Glow
        ctx.beginPath();
        ctx.arc(x, y, size + 6, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.3)`;
        ctx.fill();

        // Point
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fill();
        ctx.strokeStyle = '#D97757';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
    }

    // Draw axis labels
    ctx.fillStyle = '#1a1613';
    ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, sans-serif';

    // X axis label
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(`PC${xAxis + 1}`, padding.left + plotWidth / 2, dimensions.height - 20);

    // Y axis label
    ctx.save();
    ctx.translate(18, padding.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`PC${yAxis + 1}`, 0, 0);
    ctx.restore();

    // Draw tick labels
    ctx.fillStyle = '#7a6b8a';
    ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';

    const { minX, maxX, minY, maxY } = bounds;
    const ticks = 5;

    // Compute appropriate decimal places based on data range
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const decimalsX = Math.max(0, Math.min(4, Math.ceil(-Math.log10(rangeX / ticks)) + 1));
    const decimalsY = Math.max(0, Math.min(4, Math.ceil(-Math.log10(rangeY / ticks)) + 1));

    // X ticks
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= ticks; i++) {
      const val = minX + (maxX - minX) * (i / ticks);
      const { x } = toCanvas(val, minY);
      if (x >= padding.left && x <= dimensions.width - padding.right) {
        ctx.fillText(val.toFixed(decimalsX), x, dimensions.height - padding.bottom + 5);
      }
    }

    // Y ticks
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= ticks; i++) {
      const val = minY + (maxY - minY) * (i / ticks);
      const { y } = toCanvas(minX, val);
      if (y >= padding.top && y <= dimensions.height - padding.bottom) {
        ctx.fillText(val.toFixed(decimalsY), padding.left - 8, y);
      }
    }

  }, [positions, colors, dimensions, bounds, toCanvas, showAxes, showGrid, backgroundColor, pointSize, selectedIndex, hoveredIndex, transform, xAxis, yAxis, padding, dpr, visibility]);

  // Find point at position
  const findPointAt = useCallback((canvasX: number, canvasY: number): number | null => {
    const hitRadius = (pointSize + 5) * Math.min(1.5, Math.max(0.5, transform.scale * 0.8));
    const numPoints = positions.length / 3;

    // Search in reverse to prioritize points drawn on top
    for (let i = numPoints - 1; i >= 0; i--) {
      // Skip invisible points
      if (visibility && visibility[i] < 0.5) continue;

      const dataX = positions[i * 3 + xAxis];
      const dataY = positions[i * 3 + yAxis];
      if (!isFinite(dataX) || !isFinite(dataY)) continue;

      const { x, y } = toCanvas(dataX, dataY);
      const dx = canvasX - x;
      const dy = canvasY - y;

      if (dx * dx + dy * dy <= hitRadius * hitRadius) {
        return i;
      }
    }
    return null;
  }, [positions, toCanvas, pointSize, transform, xAxis, yAxis, visibility]);

  // Mouse handlers
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setMousePosition({ x: e.clientX, y: e.clientY });

    if (isPanning) {
      setTransform(prev => ({
        ...prev,
        offsetX: prev.offsetX + (e.clientX - panStart.x),
        offsetY: prev.offsetY + (e.clientY - panStart.y),
      }));
      setPanStart({ x: e.clientX, y: e.clientY });
      return;
    }

    const index = findPointAt(x, y);
    setHoveredIndex(index);

    if (index !== null && pointData[index]) {
      const dataX = positions[index * 3 + xAxis];
      const dataY = positions[index * 3 + yAxis];
      setTooltipVisible(true);
      setTooltipData({
        ...pointData[index],
        position: { x: dataX, y: dataY, z: 0 },
      });
      onPointHover?.(index, pointData[index]);
    } else {
      setTooltipVisible(false);
      setTooltipData(null);
      onPointHover?.(null, null);
    }
  }, [findPointAt, pointData, positions, onPointHover, isPanning, panStart, xAxis, yAxis]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isPanning) return;

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const index = findPointAt(x, y);
    if (index !== null && pointData[index]) {
      onPointSelect?.(index, pointData[index]);
    } else {
      onPointSelect?.(null, null);
    }
  }, [findPointAt, pointData, onPointSelect, isPanning]);

  // Handle wheel zoom with native event listener (passive: false for preventDefault)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setTransform(prev => ({
        ...prev,
        scale: Math.max(0.2, Math.min(10, prev.scale * delta)),
      }));
    };

    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
      e.preventDefault();
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
    }
  }, []);

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setTooltipVisible(false);
    setTooltipData(null);
    setHoveredIndex(null);
    setIsPanning(false);
    onPointHover?.(null, null);
  }, [onPointHover]);

  const handleReset = useCallback(() => {
    setTransform({ scale: 1, offsetX: 0, offsetY: 0 });
  }, []);

  const numPoints = positions.length / 3;

  // Count visible points
  const visibleCount = useMemo(() => {
    if (!visibility) return numPoints;
    let count = 0;
    for (let i = 0; i < numPoints; i++) {
      if (visibility[i] >= 0.5) count++;
    }
    return count;
  }, [visibility, numPoints]);

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        position: 'absolute',
        inset: 0,
        borderRadius: '16px',
        overflow: 'hidden',
        ...style,
      }}
    >
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onClick={handleClick}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
          cursor: isPanning ? 'grabbing' : hoveredIndex !== null ? 'pointer' : 'crosshair',
        }}
      />

      {/* Point count indicator */}
      <div
        style={{
          position: 'absolute',
          top: '12px',
          left: '12px',
          padding: '6px 12px',
          fontSize: '11px',
          fontWeight: 600,
          color: '#7a6b8a',
          background: 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(8px)',
          borderRadius: '8px',
          border: '1px solid rgba(180, 160, 200, 0.2)',
          fontFamily: 'ui-monospace, monospace',
          zIndex: 10,
        }}
      >
        {visibleCount.toLocaleString()}{visibility && visibleCount !== numPoints ? ` / ${numPoints.toLocaleString()}` : ''} points
      </div>

      {/* Selected point indicator */}
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
            fontFamily: 'ui-monospace, monospace',
            zIndex: 10,
          }}
        >
          Selected: #{selectedSampleIdx}
        </div>
      )}

      {/* Zoom indicator */}
      {transform.scale !== 1 && (
        <div
          style={{
            position: 'absolute',
            top: '48px',
            left: '12px',
            padding: '4px 8px',
            fontSize: '10px',
            fontWeight: 500,
            color: '#9a8baa',
            background: 'rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(8px)',
            borderRadius: '6px',
            zIndex: 10,
          }}
        >
          {transform.scale.toFixed(1)}x
        </div>
      )}

      {/* Controls */}
      <div
        style={{
          position: 'absolute',
          bottom: '12px',
          right: '12px',
          display: 'flex',
          gap: '8px',
          zIndex: 10,
        }}
      >
        <button
          onClick={handleReset}
          style={{
            padding: '6px 12px',
            fontSize: '11px',
            fontWeight: 600,
            color: '#7a6b8a',
            background: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(8px)',
            border: '1px solid rgba(180, 160, 200, 0.3)',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 0.15s',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(255, 255, 255, 1)';
            e.currentTarget.style.borderColor = 'rgba(198, 120, 221, 0.5)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.9)';
            e.currentTarget.style.borderColor = 'rgba(180, 160, 200, 0.3)';
          }}
        >
          Reset View
        </button>
      </div>

      {/* Instructions */}
      <div
        style={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          padding: '4px 8px',
          fontSize: '9px',
          color: '#9a8baa',
          background: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(8px)',
          borderRadius: '6px',
          zIndex: 10,
        }}
      >
        Scroll to zoom • Shift+drag to pan
      </div>

      {/* Tooltip */}
      <Tooltip
        data={tooltipData}
        mousePosition={mousePosition}
        visible={tooltipVisible}
      />
    </div>
  );
}

// Memoize to prevent unnecessary re-renders when parent state changes
export const ScatterPlot2D = memo(ScatterPlot2DInner);
export default ScatterPlot2D;
