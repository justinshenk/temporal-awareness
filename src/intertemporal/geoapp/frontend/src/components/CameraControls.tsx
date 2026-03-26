import { useRef, useCallback, useEffect, useState } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';

interface CameraPreset {
  name: string;
  label: string;
}

const CAMERA_PRESET_NAMES: CameraPreset[] = [
  { name: 'iso', label: 'Iso' },
  { name: 'top', label: 'Top' },
  { name: 'front', label: 'Front' },
  { name: 'side', label: 'Side' },
];

// Helper to compute preset position based on center and distance
export function getPresetPosition(
  presetName: string,
  center: [number, number, number],
  distance: number
): [number, number, number] {
  switch (presetName) {
    case 'iso': {
      const offset = distance / Math.sqrt(3);
      return [center[0] + offset, center[1] + offset, center[2] + offset];
    }
    case 'top':
      return [center[0], center[1] + distance, center[2]];
    case 'front':
      return [center[0], center[1], center[2] + distance];
    case 'side':
      return [center[0] + distance, center[1], center[2]];
    default:
      return [center[0] + distance, center[1] + distance, center[2] + distance];
  }
}

interface CameraControlsInnerProps {
  enableDamping?: boolean;
  dampingFactor?: number;
  autoRotate?: boolean;
  autoRotateSpeed?: number;
  minDistance?: number;
  maxDistance?: number;
  onCameraChange?: (position: THREE.Vector3, target: THREE.Vector3) => void;
  cameraStateRef: React.MutableRefObject<{
    targetPosition: THREE.Vector3 | null;
    targetLookAt: THREE.Vector3 | null;
    isAnimating: boolean;
    animationProgress: number;
    startPosition: THREE.Vector3;
    startTarget: THREE.Vector3;
  }>;
  /** Initial look-at target (center of data) */
  initialTarget?: [number, number, number];
}

// Inner component that must be inside Canvas
function CameraControlsInner({
  enableDamping = true,
  dampingFactor = 0.05,
  autoRotate = false,
  autoRotateSpeed = 1,
  minDistance = 1,
  maxDistance = 50,
  onCameraChange,
  cameraStateRef,
  initialTarget = [0, 0, 0],
}: CameraControlsInnerProps) {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const { camera } = useThree();
  const hasSetInitialTarget = useRef(false);

  // Set initial target on first render
  useEffect(() => {
    if (controlsRef.current && !hasSetInitialTarget.current) {
      controlsRef.current.target.set(...initialTarget);
      controlsRef.current.update();
      hasSetInitialTarget.current = true;
    }
  }, [initialTarget]);

  // Smooth camera animation
  useFrame(() => {
    const state = cameraStateRef.current;
    if (!state.isAnimating || !state.targetPosition || !state.targetLookAt) return;

    // Capture current camera position on first frame of animation
    if (state.animationProgress === 0) {
      state.startPosition.copy(camera.position);
      if (controlsRef.current) {
        state.startTarget.copy(controlsRef.current.target);
      }
    }

    state.animationProgress += 0.02;
    const t = easeOutCubic(Math.min(state.animationProgress, 1));

    camera.position.lerpVectors(state.startPosition, state.targetPosition, t);

    if (controlsRef.current) {
      const currentTarget = controlsRef.current.target.clone();
      currentTarget.lerpVectors(state.startTarget, state.targetLookAt, t);
      controlsRef.current.target.copy(currentTarget);
      controlsRef.current.update();
    }

    if (state.animationProgress >= 1) {
      state.isAnimating = false;
      state.targetPosition = null;
      state.targetLookAt = null;
    }
  });

  // Report camera changes
  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls || !onCameraChange) return;

    const handleChange = () => {
      onCameraChange(camera.position.clone(), controls.target.clone());
    };

    controls.addEventListener('change', handleChange);
    return () => controls.removeEventListener('change', handleChange);
  }, [camera, onCameraChange]);

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enabled={true}
      enableDamping={enableDamping}
      dampingFactor={dampingFactor}
      autoRotate={autoRotate}
      autoRotateSpeed={autoRotateSpeed}
      minDistance={minDistance}
      maxDistance={maxDistance}
      enablePan={true}
      enableZoom={true}
      enableRotate={true}
      screenSpacePanning={true}
      zoomSpeed={1}
      rotateSpeed={1}
      panSpeed={1}
    />
  );
}

// Easing function for smooth animation
function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

export interface CameraControlsProps {
  enableDamping?: boolean;
  dampingFactor?: number;
  autoRotate?: boolean;
  autoRotateSpeed?: number;
  minDistance?: number;
  maxDistance?: number;
  onCameraChange?: (position: THREE.Vector3, target: THREE.Vector3) => void;
  initialPosition?: [number, number, number];
  initialTarget?: [number, number, number];
}

// External controls UI component (must be OUTSIDE Canvas)
export interface CameraControlsUIProps {
  onPresetClick: (position: [number, number, number], target: [number, number, number]) => void;
  onResetClick: () => void;
  currentPreset?: string;
  center: [number, number, number];
  cameraDistance: number;
}

export function CameraControlsUI({
  onPresetClick,
  onResetClick,
  currentPreset,
  center,
  cameraDistance,
}: CameraControlsUIProps) {
  return (
    <div
      style={{
        position: 'absolute',
        bottom: '16px',
        left: '50%',
        transform: 'translateX(-50%)',
        display: 'flex',
        gap: '8px',
        padding: '8px 12px',
        background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 244, 255, 0.9) 100%)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        borderRadius: '12px',
        border: '1px solid rgba(180, 160, 200, 0.25)',
        boxShadow: '0 4px 20px rgba(100, 80, 120, 0.1)',
        zIndex: 100,
      }}
    >
      {CAMERA_PRESET_NAMES.map((preset) => (
        <button
          key={preset.name}
          onClick={() => onPresetClick(getPresetPosition(preset.name, center, cameraDistance), center)}
          style={{
            padding: '6px 12px',
            fontSize: '12px',
            fontWeight: 600,
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 150ms ease',
            background:
              currentPreset === preset.name
                ? 'linear-gradient(135deg, #D97757 0%, #348296 100%)'
                : 'rgba(248, 244, 255, 0.8)',
            color: currentPreset === preset.name ? '#fff' : '#1a1613',
            boxShadow:
              currentPreset === preset.name
                ? '0 2px 8px rgba(198, 120, 221, 0.3)'
                : 'none',
          }}
          onMouseEnter={(e) => {
            if (currentPreset !== preset.name) {
              e.currentTarget.style.background = 'rgba(198, 120, 221, 0.15)';
            }
          }}
          onMouseLeave={(e) => {
            if (currentPreset !== preset.name) {
              e.currentTarget.style.background = 'rgba(248, 244, 255, 0.8)';
            }
          }}
        >
          {preset.label}
        </button>
      ))}
      <div
        style={{
          width: '1px',
          background: 'rgba(180, 160, 200, 0.3)',
          margin: '0 4px',
        }}
      />
      <button
        onClick={onResetClick}
        style={{
          padding: '6px 12px',
          fontSize: '12px',
          fontWeight: 600,
          border: 'none',
          borderRadius: '8px',
          cursor: 'pointer',
          transition: 'all 150ms ease',
          background: 'rgba(248, 244, 255, 0.8)',
          color: '#7a6b8a',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'rgba(198, 120, 221, 0.15)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'rgba(248, 244, 255, 0.8)';
        }}
      >
        Reset
      </button>
    </div>
  );
}

// Hook to manage camera state and animations
export function useCameraControls(
  initialPosition: [number, number, number] = [5, 5, 5],
  initialTarget: [number, number, number] = [0, 0, 0]
) {
  const [currentPreset, setCurrentPreset] = useState<string | undefined>('iso');

  const cameraStateRef = useRef({
    targetPosition: null as THREE.Vector3 | null,
    targetLookAt: null as THREE.Vector3 | null,
    isAnimating: false,
    animationProgress: 0,
    startPosition: new THREE.Vector3(...initialPosition),
    startTarget: new THREE.Vector3(...initialTarget),
  });

  const animateTo = useCallback(
    (position: [number, number, number], target: [number, number, number]) => {
      const state = cameraStateRef.current;
      // Note: startPosition/startTarget are captured from actual camera in useFrame
      // when animationProgress is 0, so we don't need to set them here
      state.targetPosition = new THREE.Vector3(...position);
      state.targetLookAt = new THREE.Vector3(...target);
      state.animationProgress = 0;
      state.isAnimating = true;
    },
    []
  );

  const handlePresetClick = useCallback(
    (position: [number, number, number], target: [number, number, number]) => {
      // Note: We don't track preset name anymore since positions are dynamic
      setCurrentPreset(undefined);
      animateTo(position, target);
    },
    [animateTo]
  );

  const handleResetClick = useCallback(() => {
    setCurrentPreset('iso');
    animateTo(initialPosition, initialTarget);
  }, [animateTo, initialPosition, initialTarget]);

  return {
    currentPreset,
    cameraStateRef,
    handlePresetClick,
    handleResetClick,
    CameraControlsInner,
  };
}

export { CameraControlsInner, CAMERA_PRESET_NAMES };
export type { CameraPreset };
export default CameraControlsUI;
