import { useRef, useState, useCallback, useMemo, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

import { PointCloud, PointData } from './PointCloud';
import { Tooltip, TooltipData } from './Tooltip';
import {
  CameraControlsUI,
  useCameraControls,
  CameraControlsInner,
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
}

// Axes helper component
function AxesHelper({ size = 5 }: { size?: number }) {
  return <axesHelper args={[size]} />;
}

// Grid helper component
function GridHelper({
  size = 10,
  divisions = 10,
  colorCenterLine = '#C678DD',
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
      <meshBasicMaterial color="#C678DD" wireframe />
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
  initialCameraPosition: [number, number, number];
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
  cameraStateRef,
  initialCameraPosition,
}: SceneContentProps) {
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

  const gridSize = useMemo(() => {
    const range = bounds.max.clone().sub(bounds.min);
    return Math.max(range.x, range.y, range.z) * 1.5;
  }, [bounds]);

  return (
    <>
      <PerspectiveCamera
        makeDefault
        position={initialCameraPosition}
        fov={60}
        near={0.1}
        far={1000}
      />

      <CameraControlsInner
        cameraStateRef={cameraStateRef}
        enableDamping={true}
        dampingFactor={0.05}
        minDistance={1}
        maxDistance={100}
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
          colorCenterLine="#C678DD"
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
  backgroundColor = '#fef6f9',
  showAxes = true,
  showGrid = true,
  onPointHover,
  onPointSelect,
  initialCameraPosition = [5, 5, 5],
  initialCameraTarget = [0, 0, 0],
  className,
  style,
}: ScatterPlot3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [tooltipData, setTooltipData] = useState<TooltipData | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  // Camera controls hook
  const {
    currentPreset,
    cameraStateRef,
    handlePresetClick,
    handleResetClick,
  } = useCameraControls(initialCameraPosition, initialCameraTarget);

  // Track mouse position for tooltip
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    setMousePosition({ x: e.clientX, y: e.clientY });
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
  const handlePointSelect = useCallback(
    (index: number | null, _point: THREE.Vector3 | null, data: PointData | null) => {
      setSelectedIndex(index);
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
          initialCameraPosition={initialCameraPosition}
        />
      </Canvas>

      {/* Camera controls UI */}
      <CameraControlsUI
        onPresetClick={handlePresetClick}
        onResetClick={handleResetClick}
        currentPreset={currentPreset}
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
        }}
      >
        {(positions.length / 3).toLocaleString()} points
      </div>

      {/* Selected point indicator */}
      {selectedIndex !== null && (
        <div
          style={{
            position: 'absolute',
            top: '12px',
            right: '12px',
            padding: '6px 12px',
            fontSize: '11px',
            fontWeight: 600,
            color: '#C678DD',
            background: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(8px)',
            WebkitBackdropFilter: 'blur(8px)',
            borderRadius: '8px',
            border: '1px solid rgba(198, 120, 221, 0.3)',
            fontFamily: 'monospace',
          }}
        >
          Selected: #{selectedIndex}
        </div>
      )}
    </div>
  );
}

export default ScatterPlot3D;
