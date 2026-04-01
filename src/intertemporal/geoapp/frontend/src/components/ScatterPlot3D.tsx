import { useRef, useState, useCallback, useMemo, Suspense, memo, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

import { PointCloud, PointData } from './PointCloud';
import { Tooltip, TooltipData } from './Tooltip';
import {
  CameraControlsUI,
  CameraControlsInner,
  useCameraControls,
} from './CameraControls';

// Logging helper
const log = (message: string, data?: Record<string, unknown>) => {
  const ts = new Date().toISOString().slice(11, 23);
  const dataStr = data ? ` | ${Object.entries(data).map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' ')}` : '';
  console.log(`[${ts}] [CLIENT] [ScatterPlot3D] ${message}${dataStr}`);
};

export interface ScatterPlot3DProps {
  positions: Float32Array;
  colors: Float32Array;
  pointData?: PointData[];
  pointSize?: number;
  backgroundColor?: string;
  showAxes?: boolean;
  showGrid?: boolean;
  onPointHover?: (index: number | null, data: PointData | null) => void;
  onPointSelect?: (index: number | null, data: PointData | null) => void;
  initialCameraPosition?: [number, number, number];
  initialCameraTarget?: [number, number, number];
  className?: string;
  style?: React.CSSProperties;
  /** The currently selected sample index (from pointData.sampleIdx) - used for both display and point highlighting */
  selectedSampleIdx?: number | null;
  /** Visibility mask - 1.0 for visible, 0.0 for hidden */
  visibility?: Float32Array;
}

// Axes helper component
function AxesHelper({ size = 5 }: { size?: number }) {
  return <axesHelper args={[size]} />;
}

// Grid helper component
function GridHelper({
  size = 10,
  divisions = 10,
  colorCenterLine = '#D97757',
  colorGrid = '#e0dae8',
}: {
  size?: number;
  divisions?: number;
  colorCenterLine?: string;
  colorGrid?: string;
}) {
  return (
    <gridHelper
      args={[size, divisions, colorCenterLine, colorGrid]}
      position={[0, -0.01, 0]}
      rotation={[0, 0, 0]}
    />
  );
}

// Loading fallback
function LoadingFallback() {
  return (
    <mesh>
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshBasicMaterial color="#D97757" wireframe />
    </mesh>
  );
}

// Scene content (inside Canvas) - memoized for performance
interface SceneContentProps {
  positions: Float32Array;
  colors: Float32Array;
  pointData: PointData[];
  pointSize: number;
  showAxes: boolean;
  showGrid: boolean;
  selectedIndex: number | null;
  onHover: (index: number | null, point: THREE.Vector3 | null, data: PointData | null) => void;
  onSelect: (index: number | null, point: THREE.Vector3 | null, data: PointData | null) => void;
  // Pre-computed values from parent
  center: [number, number, number];
  cameraDistance: number;
  gridSize: number;
  visibility?: Float32Array;
  backgroundColor: string;
  // Camera animation state ref
  cameraStateRef: React.MutableRefObject<{
    targetPosition: THREE.Vector3 | null;
    targetLookAt: THREE.Vector3 | null;
    isAnimating: boolean;
    animationProgress: number;
    startPosition: THREE.Vector3;
    startTarget: THREE.Vector3;
  }>;
  // Disable pointer events during camera interaction
  isInteracting: boolean;
}

const SceneContent = memo(function SceneContent({
  positions,
  colors,
  pointData,
  pointSize,
  showAxes,
  showGrid,
  selectedIndex,
  onHover,
  onSelect,
  center,
  cameraDistance,
  gridSize,
  visibility,
  backgroundColor,
  cameraStateRef,
  isInteracting,
}: SceneContentProps) {
  // Performance measurement
  const renderStart = useRef(performance.now());

  // Log render time after component mounts/updates
  useEffect(() => {
    const renderTime = performance.now() - renderStart.current;
    if (positions.length > 0) {
      console.log(`[Perf] SceneContent render: ${renderTime.toFixed(1)}ms for ${positions.length / 3} points`);
    }
    renderStart.current = performance.now();
  });

  // Track if we've initialized the camera (only set position once)
  const hasInitialized = useRef(false);
  const initialCameraPos = useRef<[number, number, number] | null>(null);

  // Store initial camera position only once (first time we have valid data)
  if (!hasInitialized.current && positions.length > 0) {
    const offset = cameraDistance / Math.sqrt(3);
    initialCameraPos.current = [
      center[0] + offset,
      center[1] + offset,
      center[2] + offset,
    ];
    hasInitialized.current = true;
  }

  // Use stored initial position, or fallback for first render
  const cameraPosition = initialCameraPos.current || [5, 5, 5];

  return (
    <>
      {/* Set scene background color (required with alpha: false in Canvas gl config) */}
      <color attach="background" args={[backgroundColor]} />

      <PerspectiveCamera
        makeDefault
        position={cameraPosition}
        fov={60}
        near={0.01}
        far={1000}
      />

      <CameraControlsInner
        enableDamping={true}
        dampingFactor={0.05}
        minDistance={0.1}
        maxDistance={1000}
        cameraStateRef={cameraStateRef}
        initialTarget={center}
      />

      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 10, 5]} intensity={0.8} />
      <directionalLight position={[-10, -10, -5]} intensity={0.3} />

      {/* Scene helpers */}
      {showAxes && <AxesHelper size={gridSize / 2} />}
      {showGrid && (
        <GridHelper
          size={gridSize}
          divisions={Math.floor(gridSize)}
          colorCenterLine="#D97757"
          colorGrid="rgba(200, 180, 220, 0.3)"
        />
      )}

      {/* Point cloud */}
      <Suspense fallback={<LoadingFallback />}>
        <PointCloud
          positions={positions}
          colors={colors}
          pointSize={pointSize}
          pointData={pointData}
          onHover={onHover}
          onSelect={onSelect}
          selectedIndex={selectedIndex}
          hoverScale={1.5}
          visibility={visibility}
          disablePointerEvents={isInteracting}
        />
      </Suspense>
    </>
  );
});

function ScatterPlot3DInner({
  positions,
  colors,
  pointData = [],
  pointSize = 3,
  backgroundColor = '#faf8f5',
  showAxes = true,
  showGrid = true,
  onPointHover,
  onPointSelect,
  initialCameraPosition: _initialCameraPosition = [5, 5, 5],
  initialCameraTarget: _initialCameraTarget = [0, 0, 0],
  className,
  style,
  selectedSampleIdx,
  visibility,
}: ScatterPlot3DProps) {
  const renderCount = useRef(0);
  renderCount.current++;
  const n_points = positions.length / 3;
  log(`Render #${renderCount.current}`, { n_points, selectedSampleIdx });

  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [tooltipData, setTooltipData] = useState<TooltipData | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isInteracting, setIsInteracting] = useState(false);
  const interactionTimeoutRef = useRef<number | null>(null);

  // Compute data bounds for camera positioning - cached for performance
  const { center, cameraDistance, gridSize } = useMemo(() => {
    if (positions.length === 0) {
      return {
        center: [0, 0, 0] as [number, number, number],
        cameraDistance: 10,
        gridSize: 10,
      };
    }

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < positions.length; i += 3) {
      minX = Math.min(minX, positions[i]);
      minY = Math.min(minY, positions[i + 1]);
      minZ = Math.min(minZ, positions[i + 2]);
      maxX = Math.max(maxX, positions[i]);
      maxY = Math.max(maxY, positions[i + 1]);
      maxZ = Math.max(maxZ, positions[i + 2]);
    }

    const centerPt: [number, number, number] = [
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2,
    ];
    const maxRange = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.1);
    const camDist = maxRange * 2;
    const grid = maxRange * 1.5;

    return { center: centerPt, cameraDistance: camDist, gridSize: grid };
  }, [positions]);

  // Convert selectedSampleIdx (sample ID) to array index for point highlighting
  // This is necessary because filtering can cause array indices to differ from sample IDs
  const selectedIndex = useMemo(() => {
    if (selectedSampleIdx === null || selectedSampleIdx === undefined) {
      return null;
    }
    const index = pointData.findIndex(p => p.sampleIdx === selectedSampleIdx);
    return index === -1 ? null : index;
  }, [selectedSampleIdx, pointData]);

  // Camera controls hook - use data-aware initial position
  const dataAwareInitialPosition = useMemo((): [number, number, number] => {
    const offset = cameraDistance / Math.sqrt(3);
    return [center[0] + offset, center[1] + offset, center[2] + offset];
  }, [center, cameraDistance]);

  const {
    currentPreset,
    cameraStateRef,
    handlePresetClick,
    handleResetClick,
  } = useCameraControls(dataAwareInitialPosition, center);

  // Track mouse position for tooltip
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    setMousePosition({ x: e.clientX, y: e.clientY });
  }, []);

  // Clear tooltip when mouse leaves the container entirely
  // This handles edge cases where Three.js onPointerOut doesn't fire
  const handleMouseLeave = useCallback(() => {
    setTooltipVisible(false);
    setTooltipData(null);
    if (onPointHover) {
      onPointHover(null, null);
    }
  }, [onPointHover]);

  // Track camera interaction to disable pointer events during rotation/zoom
  // This prevents expensive raycasting during camera manipulation
  const handleInteractionStart = useCallback(() => {
    setIsInteracting(true);
    // Clear any pending timeout
    if (interactionTimeoutRef.current !== null) {
      window.clearTimeout(interactionTimeoutRef.current);
      interactionTimeoutRef.current = null;
    }
  }, []);

  const handleInteractionEnd = useCallback(() => {
    // Delay re-enabling pointer events to avoid spurious raycasts
    if (interactionTimeoutRef.current !== null) {
      window.clearTimeout(interactionTimeoutRef.current);
    }
    interactionTimeoutRef.current = window.setTimeout(() => {
      setIsInteracting(false);
      interactionTimeoutRef.current = null;
    }, 100);
  }, []);

  const handleWheel = useCallback(() => {
    handleInteractionStart();
    // Reset on wheel - treat as brief interaction
    if (interactionTimeoutRef.current !== null) {
      window.clearTimeout(interactionTimeoutRef.current);
    }
    interactionTimeoutRef.current = window.setTimeout(() => {
      setIsInteracting(false);
      interactionTimeoutRef.current = null;
    }, 150);
  }, [handleInteractionStart]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (interactionTimeoutRef.current !== null) {
        window.clearTimeout(interactionTimeoutRef.current);
      }
    };
  }, []);

  // Handle point hover
  const handlePointHover = useCallback(
    (index: number | null, point: THREE.Vector3 | null, data: PointData | null) => {
      if (index !== null && point !== null && data !== null) {
        setTooltipVisible(true);
        setTooltipData({
          ...data,
          position: { x: point.x, y: point.y, z: point.z },
        });
      } else {
        setTooltipVisible(false);
        setTooltipData(null);
      }

      if (onPointHover) {
        onPointHover(index, data);
      }
    },
    [onPointHover]
  );

  // Handle point selection
  // Note: We don't maintain local selection state - the parent controls selection via selectedSampleIdx prop
  const handlePointSelect = useCallback(
    (index: number | null, _point: THREE.Vector3 | null, data: PointData | null) => {
      if (onPointSelect) {
        onPointSelect(index, data);
      }
    },
    [onPointSelect]
  );

  // Memoize point data to ensure stable references
  const stablePointData = useMemo(() => pointData, [pointData]);

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
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onMouseDown={handleInteractionStart}
      onMouseUp={handleInteractionEnd}
      onWheel={handleWheel}
    >
      <Canvas
        gl={{
          antialias: false, // Disable for performance
          alpha: false, // Opaque context for proper point rendering
          powerPreference: 'high-performance',
          stencil: false,
          depth: true,
        }}
        dpr={1} // Fixed DPR for consistent performance
        style={{ background: backgroundColor, width: '100%', height: '100%' }}
        performance={{ min: 0.5 }}
      >
        <SceneContent
          positions={positions}
          colors={colors}
          pointData={stablePointData}
          pointSize={pointSize}
          showAxes={showAxes}
          showGrid={showGrid}
          selectedIndex={selectedIndex}
          onHover={handlePointHover}
          onSelect={handlePointSelect}
          center={center}
          cameraDistance={cameraDistance}
          gridSize={gridSize}
          visibility={visibility}
          backgroundColor={backgroundColor}
          cameraStateRef={cameraStateRef}
          isInteracting={isInteracting}
        />
      </Canvas>

      {/* Camera controls UI */}
      <CameraControlsUI
        onPresetClick={handlePresetClick}
        onResetClick={handleResetClick}
        currentPreset={currentPreset}
        center={center}
        cameraDistance={cameraDistance}
      />

      {/* Tooltip overlay */}
      <Tooltip
        data={tooltipData}
        mousePosition={mousePosition}
        visible={tooltipVisible}
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
          background: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(8px)',
          WebkitBackdropFilter: 'blur(8px)',
          borderRadius: '8px',
          border: '1px solid rgba(180, 160, 200, 0.2)',
          fontFamily: 'monospace',
          zIndex: 10, // Ensure it stays above canvas but below tooltip
        }}
      >
        {visibility
          ? `${visibility.filter(v => v > 0.5).length.toLocaleString()} / ${(positions.length / 3).toLocaleString()} visible`
          : `${(positions.length / 3).toLocaleString()} points`
        }
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
            WebkitBackdropFilter: 'blur(8px)',
            borderRadius: '8px',
            border: '1px solid rgba(198, 120, 221, 0.3)',
            fontFamily: 'monospace',
            zIndex: 10, // Ensure it stays above canvas but below tooltip
          }}
        >
          Selected: #{selectedSampleIdx}
        </div>
      )}
    </div>
  );
}

// Memoize to prevent unnecessary re-renders when parent state changes
export const ScatterPlot3D = memo(ScatterPlot3DInner);
export default ScatterPlot3D;
