import { useRef, useState, useCallback, useMemo, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

import { PointCloud, PointData } from './PointCloud';
import { Tooltip, TooltipData } from './Tooltip';
import { OrbitControls } from '@react-three/drei';
import {
  CameraControlsUI,
  useCameraControls,
} from './CameraControls';

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

// Scene content (inside Canvas)
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
  cameraStateRef: React.MutableRefObject<{
    targetPosition: THREE.Vector3 | null;
    targetLookAt: THREE.Vector3 | null;
    isAnimating: boolean;
    animationProgress: number;
    startPosition: THREE.Vector3;
    startTarget: THREE.Vector3;
  }>;
}

function SceneContent({
  positions,
  colors,
  pointData,
  pointSize,
  showAxes,
  showGrid,
  selectedIndex,
  onHover,
  onSelect,
  cameraStateRef: _cameraStateRef,
}: SceneContentProps) {
  // Track if we've initialized the camera (only set position once)
  const hasInitialized = useRef(false);
  const initialCameraPos = useRef<[number, number, number] | null>(null);
  const initialTarget = useRef<[number, number, number] | null>(null);

  // Calculate scene bounds for auto-scaling
  const bounds = useMemo(() => {
    if (positions.length === 0) {
      return { min: new THREE.Vector3(-5, -5, -5), max: new THREE.Vector3(5, 5, 5) };
    }

    const min = new THREE.Vector3(Infinity, Infinity, Infinity);
    const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);

    for (let i = 0; i < positions.length; i += 3) {
      min.x = Math.min(min.x, positions[i]);
      min.y = Math.min(min.y, positions[i + 1]);
      min.z = Math.min(min.z, positions[i + 2]);
      max.x = Math.max(max.x, positions[i]);
      max.y = Math.max(max.y, positions[i + 1]);
      max.z = Math.max(max.z, positions[i + 2]);
    }

    return { min, max };
  }, [positions]);

  // Compute center of data
  const center = useMemo(() => {
    return new THREE.Vector3(
      (bounds.min.x + bounds.max.x) / 2,
      (bounds.min.y + bounds.max.y) / 2,
      (bounds.min.z + bounds.max.z) / 2
    );
  }, [bounds]);

  // Compute data extent and appropriate camera distance
  const { gridSize, cameraDistance } = useMemo(() => {
    const range = bounds.max.clone().sub(bounds.min);
    const maxRange = Math.max(range.x, range.y, range.z, 0.1); // Prevent zero
    const grid = maxRange * 1.5;
    // Position camera at distance of ~2x the data extent for good view
    const camDist = maxRange * 2;
    return { gridSize: grid, cameraDistance: camDist };
  }, [bounds]);

  // Store initial camera position only once (first time we have valid data)
  if (!hasInitialized.current && positions.length > 0) {
    const offset = cameraDistance / Math.sqrt(3);
    initialCameraPos.current = [
      center.x + offset,
      center.y + offset,
      center.z + offset,
    ];
    initialTarget.current = [center.x, center.y, center.z];
    hasInitialized.current = true;
  }

  // Use stored initial position, or fallback for first render
  const cameraPosition = initialCameraPos.current || [5, 5, 5];
  const targetPosition = initialTarget.current || [0, 0, 0];

  return (
    <>
      <PerspectiveCamera
        makeDefault
        position={cameraPosition}
        fov={60}
        near={0.01}
        far={1000}
      />

      <OrbitControls
        makeDefault
        enableDamping={true}
        dampingFactor={0.05}
        minDistance={0.1}
        maxDistance={1000}
        target={targetPosition}
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
        />
      </Suspense>
    </>
  );
}

export function ScatterPlot3D({
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
}: ScatterPlot3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [tooltipData, setTooltipData] = useState<TooltipData | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  // Compute data bounds for camera positioning
  const { center, cameraDistance } = useMemo(() => {
    if (positions.length === 0) {
      return {
        center: [0, 0, 0] as [number, number, number],
        cameraDistance: 10,
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

    return { center: centerPt, cameraDistance: camDist };
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

  // Camera reset key - incrementing this forces camera to reinitialize
  const [cameraResetKey, setCameraResetKey] = useState(0);

  // Camera controls hook - use data-aware initial position
  const dataAwareInitialPosition = useMemo((): [number, number, number] => {
    const offset = cameraDistance / Math.sqrt(3);
    return [center[0] + offset, center[1] + offset, center[2] + offset];
  }, [center, cameraDistance]);

  const {
    currentPreset,
    cameraStateRef,
    handlePresetClick: _handlePresetClick,
    handleResetClick: _handleResetClick,
  } = useCameraControls(dataAwareInitialPosition, center);

  // Override reset to force camera remount
  const handleResetClick = useCallback(() => {
    setCameraResetKey(k => k + 1);
  }, []);

  // Preset click also forces remount for simplicity
  const handlePresetClick = useCallback((_position: [number, number, number], _target: [number, number, number]) => {
    // For now, just reset - proper preset handling would need more work
    setCameraResetKey(k => k + 1);
  }, []);

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
    >
      <Canvas
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: 'high-performance',
        }}
        dpr={[1, 2]}
        style={{ background: backgroundColor }}
        performance={{ min: 0.5 }}
      >
        <SceneContent
          key={cameraResetKey}
          positions={positions}
          colors={colors}
          pointData={stablePointData}
          pointSize={pointSize}
          showAxes={showAxes}
          showGrid={showGrid}
          selectedIndex={selectedIndex}
          onHover={handlePointHover}
          onSelect={handlePointSelect}
          cameraStateRef={cameraStateRef}
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
        {(positions.length / 3).toLocaleString()} points
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

export default ScatterPlot3D;
