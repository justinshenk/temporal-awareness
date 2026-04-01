import { useRef, useState, useCallback, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { PointData } from './PointCloud';
import { Tooltip, TooltipData } from './Tooltip';

export interface TrajectoryPlot3DProps {
  /** PC1 values for each point on X-axis */
  pc1Data: Map<string, Float32Array>;
  /** PC2 values for each point on X-axis */
  pc2Data: Map<string, Float32Array>;
  /** X-axis values in order */
  xValues: string[];
  /** X-axis label */
  xAxisLabel?: string;
  /** Title for the plot */
  title?: string;
  /** Colors for each sample (RGB triplets) */
  colors: Float32Array;
  /** Point data for tooltips */
  pointData?: PointData[];
  /** Background color */
  backgroundColor?: string;
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
  /** Optional filter mask */
  filterMask?: boolean[] | null;
  className?: string;
  style?: React.CSSProperties;
}

// Calculate bounds for normalization
function calculateBounds(pc1Data: Map<string, Float32Array>, pc2Data: Map<string, Float32Array>, xValues: string[]) {
  let minPc1 = Infinity, maxPc1 = -Infinity;
  let minPc2 = Infinity, maxPc2 = -Infinity;

  xValues.forEach((xVal) => {
    const pc1 = pc1Data.get(xVal);
    const pc2 = pc2Data.get(xVal);
    if (pc1) {
      for (let i = 0; i < pc1.length; i++) {
        if (isFinite(pc1[i])) {
          minPc1 = Math.min(minPc1, pc1[i]);
          maxPc1 = Math.max(maxPc1, pc1[i]);
        }
      }
    }
    if (pc2) {
      for (let i = 0; i < pc2.length; i++) {
        if (isFinite(pc2[i])) {
          minPc2 = Math.min(minPc2, pc2[i]);
          maxPc2 = Math.max(maxPc2, pc2[i]);
        }
      }
    }
  });

  // Add padding
  const pc1Range = maxPc1 - minPc1 || 1;
  const pc2Range = maxPc2 - minPc2 || 1;
  const pad1 = pc1Range * 0.1;
  const pad2 = pc2Range * 0.1;

  return {
    minPc1: minPc1 - pad1,
    maxPc1: maxPc1 + pad1,
    minPc2: minPc2 - pad2,
    maxPc2: maxPc2 + pad2,
  };
}

// Normalize value to [-1, 1] range
function normalize(value: number, min: number, max: number): number {
  const range = max - min || 1;
  return ((value - min) / range) * 2 - 1;
}

interface TrajectoryLinesProps {
  pc1Data: Map<string, Float32Array>;
  pc2Data: Map<string, Float32Array>;
  xValues: string[];
  colors: Float32Array;
  filterMask: boolean[] | null;
  selectedSampleIdx: number | null;
  lineOpacity: number;
  bounds: { minPc1: number; maxPc1: number; minPc2: number; maxPc2: number };
  onHover: (index: number | null) => void;
  onClick: (index: number | null) => void;
}

function TrajectoryLines({
  pc1Data,
  pc2Data,
  xValues,
  colors,
  filterMask,
  selectedSampleIdx,
  lineOpacity,
  bounds,
  onHover,
  onClick,
}: TrajectoryLinesProps) {
  const { camera, raycaster, pointer } = useThree();
  const groupRef = useRef<THREE.Group>(null);

  // Get number of samples from first data point
  const numSamples = useMemo(() => {
    if (xValues.length === 0) return 0;
    const firstPc1 = pc1Data.get(xValues[0]);
    return firstPc1 ? firstPc1.length : 0;
  }, [pc1Data, xValues]);

  // Build line geometries for each sample
  const lineData = useMemo(() => {
    const lines: { points: THREE.Vector3[]; color: THREE.Color; sampleIdx: number; isSelected: boolean }[] = [];

    for (let sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
      // Skip filtered samples
      if (filterMask && !filterMask[sampleIdx]) continue;

      const points: THREE.Vector3[] = [];
      xValues.forEach((xVal, xIdx) => {
        const pc1 = pc1Data.get(xVal);
        const pc2 = pc2Data.get(xVal);
        if (!pc1 || !pc2 || sampleIdx >= pc1.length) return;

        const pc1Val = pc1[sampleIdx];
        const pc2Val = pc2[sampleIdx];
        if (!isFinite(pc1Val) || !isFinite(pc2Val)) return;

        // X: position index normalized to [-1, 1]
        const x = (xIdx / Math.max(1, xValues.length - 1)) * 2 - 1;
        // Y: PC1 normalized
        const y = normalize(pc1Val, bounds.minPc1, bounds.maxPc1);
        // Z: PC2 normalized
        const z = normalize(pc2Val, bounds.minPc2, bounds.maxPc2);

        points.push(new THREE.Vector3(x, y, z));
      });

      if (points.length >= 2) {
        const r = colors[sampleIdx * 3] ?? 0.5;
        const g = colors[sampleIdx * 3 + 1] ?? 0.5;
        const b = colors[sampleIdx * 3 + 2] ?? 0.5;
        const color = new THREE.Color(r, g, b);
        const isSelected = sampleIdx === selectedSampleIdx;

        lines.push({ points, color, sampleIdx, isSelected });
      }
    }

    return lines;
  }, [pc1Data, pc2Data, xValues, colors, filterMask, selectedSampleIdx, numSamples, bounds]);

  // Separate selected and non-selected lines
  const regularLines = lineData.filter((l) => !l.isSelected);
  const selectedLine = lineData.find((l) => l.isSelected);

  return (
    <group ref={groupRef}>
      {/* Regular lines */}
      {regularLines.map((line, i) => (
        <Line
          key={`line-${line.sampleIdx}`}
          points={line.points}
          color={line.color}
          lineWidth={1}
          transparent
          opacity={lineOpacity}
        />
      ))}

      {/* Selected line (thicker, on top) */}
      {selectedLine && (
        <>
          {/* Glow effect */}
          <Line
            points={selectedLine.points}
            color={selectedLine.color}
            lineWidth={6}
            transparent
            opacity={0.3}
          />
          {/* Main line */}
          <Line
            points={selectedLine.points}
            color={selectedLine.color}
            lineWidth={3}
          />
          {/* Points at each position */}
          {selectedLine.points.map((point, i) => (
            <mesh key={`point-${i}`} position={point}>
              <sphereGeometry args={[0.02, 16, 16]} />
              <meshBasicMaterial color={selectedLine.color} />
            </mesh>
          ))}
        </>
      )}
    </group>
  );
}

interface AxisLabelsProps {
  xValues: string[];
  bounds: { minPc1: number; maxPc1: number; minPc2: number; maxPc2: number };
  xAxisLabel: string;
}

function AxisLabels({ xValues, bounds, xAxisLabel }: AxisLabelsProps) {
  // Detect if this is position axis (longer text labels) vs layer axis (short numbers)
  const isPositionAxis = xAxisLabel === 'Position';

  return (
    <group>
      {/* X-axis labels (positions/layers) */}
      {xValues.map((xVal, i) => {
        const x = (i / Math.max(1, xValues.length - 1)) * 2 - 1;

        if (isPositionAxis) {
          // For positions: horizontal text, smaller font, below the grid
          return (
            <Text
              key={`x-${i}`}
              position={[x, -1.15, 0.5]}
              fontSize={0.04}
              color="#666"
              anchorX="center"
              anchorY="top"
              rotation={[-Math.PI / 2, 0, 0]}
            >
              {xVal}
            </Text>
          );
        }

        // For layers: tilted numeric labels
        return (
          <Text
            key={`x-${i}`}
            position={[x, -1.2, 0]}
            fontSize={0.06}
            color="#666"
            anchorX="center"
            anchorY="top"
            rotation={[-Math.PI / 4, 0, 0]}
          >
            {xVal}
          </Text>
        );
      })}

      {/* Axis title */}
      <Text
        position={[0, -1.4, 0]}
        fontSize={0.08}
        color="#333"
        anchorX="center"
        anchorY="top"
      >
        {xAxisLabel}
      </Text>

      {/* Y-axis label (PC1) */}
      <Text
        position={[-1.3, 0, 0]}
        fontSize={0.08}
        color="#333"
        anchorX="center"
        anchorY="middle"
        rotation={[0, 0, Math.PI / 2]}
      >
        PC1
      </Text>

      {/* Z-axis label (PC2) */}
      <Text
        position={[0, 0, 1.3]}
        fontSize={0.08}
        color="#333"
        anchorX="center"
        anchorY="middle"
      >
        PC2
      </Text>

      {/* Grid lines */}
      <gridHelper args={[2, 10, '#ddd', '#eee']} position={[0, -1, 0]} />
    </group>
  );
}

export function TrajectoryPlot3D({
  pc1Data,
  pc2Data,
  xValues,
  xAxisLabel = 'Position',
  title = 'PC1 vs PC2 Trajectory',
  colors,
  pointData = [],
  backgroundColor = '#faf8f5',
  onPointHover,
  onPointSelect,
  selectedSampleIdx,
  lineOpacity = 0.6,
  isLoading = false,
  filterMask = null,
  className,
  style,
}: TrajectoryPlot3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [tooltipData, setTooltipData] = useState<TooltipData | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Calculate bounds for normalization
  const bounds = useMemo(() => calculateBounds(pc1Data, pc2Data, xValues), [pc1Data, pc2Data, xValues]);

  // Get number of samples
  const numSamples = useMemo(() => {
    if (xValues.length === 0) return 0;
    const firstPc1 = pc1Data.get(xValues[0]);
    return firstPc1 ? firstPc1.length : 0;
  }, [pc1Data, xValues]);

  // Calculate visible sample count
  const visibleSampleCount = useMemo(() => {
    if (!filterMask) return numSamples;
    return filterMask.filter(Boolean).length;
  }, [filterMask, numSamples]);

  const handleHover = useCallback((index: number | null) => {
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
  }, [pointData, onPointHover]);

  const handleClick = useCallback((index: number | null) => {
    if (index !== null && pointData[index]) {
      onPointSelect?.(index, pointData[index]);
    } else {
      onPointSelect?.(null, null);
    }
  }, [pointData, onPointSelect]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    setMousePosition({ x: e.clientX, y: e.clientY });
  }, []);

  if (isLoading) {
    return (
      <div
        ref={containerRef}
        className={className}
        style={{
          position: 'relative',
          width: '100%',
          height: '100%',
          minHeight: '400px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor,
          borderRadius: '16px',
          ...style,
        }}
      >
        <div style={{ color: '#666' }}>Loading...</div>
      </div>
    );
  }

  if (xValues.length === 0 || pc1Data.size === 0) {
    return (
      <div
        ref={containerRef}
        className={className}
        style={{
          position: 'relative',
          width: '100%',
          height: '100%',
          minHeight: '400px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor,
          borderRadius: '16px',
          ...style,
        }}
      >
        <div style={{ color: '#666' }}>No data available</div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={className}
      onMouseMove={handleMouseMove}
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
      <Canvas
        camera={{ position: [2, 1.5, 2], fov: 50 }}
        style={{ background: backgroundColor }}
      >
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />

        <TrajectoryLines
          pc1Data={pc1Data}
          pc2Data={pc2Data}
          xValues={xValues}
          colors={colors}
          filterMask={filterMask}
          selectedSampleIdx={selectedSampleIdx ?? null}
          lineOpacity={lineOpacity}
          bounds={bounds}
          onHover={handleHover}
          onClick={handleClick}
        />

        <AxisLabels
          xValues={xValues}
          bounds={bounds}
          xAxisLabel={xAxisLabel}
        />

        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={1}
          maxDistance={10}
        />
      </Canvas>

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

export default TrajectoryPlot3D;
