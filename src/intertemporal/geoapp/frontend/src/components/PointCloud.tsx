import { useRef, useMemo, useCallback, useState, useEffect, memo } from 'react';
import { useThree, ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';

// Logging helper
const log = (message: string, data?: Record<string, unknown>) => {
  const ts = new Date().toISOString().slice(11, 23);
  const dataStr = data ? ` | ${Object.entries(data).map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' ')}` : '';
  console.log(`[${ts}] [CLIENT] [PointCloud] ${message}${dataStr}`);
};

export interface PointData {
  sampleIdx: number;
  timeHorizon?: number;
  timeScale?: string;
  choiceType?: string;
  shortTermFirst?: boolean;
  [key: string]: unknown;
}

interface PointCloudProps {
  positions: Float32Array;
  colors: Float32Array;
  pointSize?: number;
  pointData?: PointData[];
  onHover?: (index: number | null, point: THREE.Vector3 | null, data: PointData | null) => void;
  onSelect?: (index: number | null, point: THREE.Vector3 | null, data: PointData | null) => void;
  selectedIndex?: number | null;
  hoverScale?: number;
  /** Visibility mask - 1.0 for visible, 0.0 for hidden */
  visibility?: Float32Array;
  /** Disable pointer events during camera interaction for performance */
  disablePointerEvents?: boolean;
}

// Reusable vectors to avoid allocations
const tempVector = new THREE.Vector3();
const projVector = new THREE.Vector3();

// Cursor pick radius in screen pixels. Points render at a screen-space size
// (see the vertex shader), so hit-testing is done in screen space rather than
// with a fixed world-space raycaster threshold that never matches the visuals.
const PICK_RADIUS_PX = 18;

// Custom shader for crisp point rendering - BIG visible points
const vertexShader = `
  attribute float pointIndex;
  attribute float visibility;

  uniform float size;
  uniform float hoverScale;
  uniform float selectedIndex;
  uniform float hoveredIndex;
  uniform float pointCount;
  uniform float maxPointSize;

  varying vec3 vColor;
  varying float vIsSelected;
  varying float vIsHovered;
  varying float vVisibility;

  void main() {
    vColor = color;
    vVisibility = visibility;

    vIsSelected = (pointIndex == selectedIndex) ? 1.0 : 0.0;
    vIsHovered = (pointIndex == hoveredIndex) ? 1.0 : 0.0;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

    // Hide point if not visible (move off-screen and set size to 0)
    if (visibility < 0.5) {
      gl_PointSize = 0.0;
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Off-screen
      return;
    }

    // Scale up selected/hovered points
    float scale = 1.0;
    if (vIsSelected > 0.5) scale = 2.0;
    else if (vIsHovered > 0.5) scale = 1.5;

    // Base point size - start BIG
    float baseSize = size * 3.0;

    // Sparsity: fewer points = bigger (range 1.0 to 4.0)
    float sparsityFactor = clamp(sqrt(2000.0 / max(pointCount, 1.0)), 1.0, 4.0);

    // Distance: closer = bigger (when zoomed in)
    float dist = max(abs(mvPosition.z), 0.5);
    float zoomFactor = clamp(5.0 / sqrt(dist), 1.0, 4.0);

    // Final size
    gl_PointSize = baseSize * scale * sparsityFactor * zoomFactor;

    // HARD minimum of 12 pixels - ALWAYS clearly visible (sparse clouds read as
    // "empty" if points are too small, especially on phone-sized viewports)
    // Max of 100 pixels when very sparse and zoomed in
    gl_PointSize = clamp(gl_PointSize, 12.0, 100.0);

    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = `
  varying vec3 vColor;
  varying float vIsSelected;
  varying float vIsHovered;
  varying float vVisibility;

  void main() {
    // Discard hidden points
    if (vVisibility < 0.5) discard;

    // Create circular points with hard edges
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center);

    // Hard cutoff for crisp circles
    if (dist > 0.45) discard;

    vec3 finalColor = vColor;

    // Brighten hovered/selected points
    if (vIsSelected > 0.5) {
      finalColor = mix(vColor, vec3(1.0), 0.4);
    } else if (vIsHovered > 0.5) {
      finalColor = mix(vColor, vec3(1.0), 0.25);
    }

    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

function PointCloudInner({
  positions,
  colors,
  pointSize = 4,
  pointData = [],
  onHover,
  onSelect,
  selectedIndex = null,
  hoverScale = 1.5,
  visibility,
  disablePointerEvents = false,
}: PointCloudProps) {
  const renderCount = useRef(0);
  renderCount.current++;

  const pointsRef = useRef<THREE.Points>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const { raycaster, gl, camera } = useThree();

  const pointCount = positions.length / 3;
  const colorPointCount = colors.length / 3;

  // Log only on significant renders (every 10th or first)
  if (renderCount.current === 1 || renderCount.current % 10 === 0) {
    log(`Render #${renderCount.current}`, { pointCount, selectedIndex });
  }

  // Use refs to avoid recreating geometry/material
  const geometryRef = useRef<THREE.BufferGeometry | null>(null);
  const positionAttrRef = useRef<THREE.BufferAttribute | null>(null);
  const colorAttrRef = useRef<THREE.BufferAttribute | null>(null);
  const indexAttrRef = useRef<THREE.BufferAttribute | null>(null);
  const visibilityAttrRef = useRef<THREE.BufferAttribute | null>(null);
  // World-space threshold so R3F's event raycast registers the points object as
  // "hit" near the cursor. The precise point is then chosen in screen space.
  const pickThresholdRef = useRef(0.5);

  // Create or update geometry - avoid full recreation
  const geometry = useMemo(() => {
    const startTime = performance.now();
    // Create new geometry only if we don't have one or point count changed significantly
    const isNewGeometry = !geometryRef.current;
    if (!geometryRef.current) {
      geometryRef.current = new THREE.BufferGeometry();
    }
    const geo = geometryRef.current;

    // Update position attribute
    const isNewPositionAttr = !positionAttrRef.current || positionAttrRef.current.count !== pointCount;
    if (!positionAttrRef.current || positionAttrRef.current.count !== pointCount) {
      positionAttrRef.current = new THREE.BufferAttribute(positions, 3);
      geo.setAttribute('position', positionAttrRef.current);
    } else {
      positionAttrRef.current.array = positions;
      positionAttrRef.current.needsUpdate = true;
    }

    // Log geometry update
    log('Geometry update', {
      isNewGeometry,
      isNewPositionAttr,
      pointCount,
      elapsed_ms: (performance.now() - startTime).toFixed(2)
    });

    // Update color attribute
    const safeColors = colorPointCount === pointCount ? colors : (() => {
      const fallback = new Float32Array(pointCount * 3);
      for (let i = 0; i < pointCount; i++) {
        fallback[i * 3] = 0.85;
        fallback[i * 3 + 1] = 0.47;
        fallback[i * 3 + 2] = 0.34;
      }
      return fallback;
    })();

    if (!colorAttrRef.current || colorAttrRef.current.count !== pointCount) {
      colorAttrRef.current = new THREE.BufferAttribute(safeColors, 3);
      geo.setAttribute('color', colorAttrRef.current);
    } else {
      colorAttrRef.current.array = safeColors;
      colorAttrRef.current.needsUpdate = true;
    }

    // Update index attribute
    if (!indexAttrRef.current || indexAttrRef.current.count !== pointCount) {
      const indices = new Float32Array(pointCount);
      for (let i = 0; i < pointCount; i++) indices[i] = i;
      indexAttrRef.current = new THREE.BufferAttribute(indices, 1);
      geo.setAttribute('pointIndex', indexAttrRef.current);
    }

    // Update visibility attribute - default to all visible (1.0)
    const safeVisibility = visibility && visibility.length === pointCount
      ? visibility
      : (() => {
          const allVisible = new Float32Array(pointCount);
          for (let i = 0; i < pointCount; i++) allVisible[i] = 1.0;
          return allVisible;
        })();

    if (!visibilityAttrRef.current || visibilityAttrRef.current.count !== pointCount) {
      visibilityAttrRef.current = new THREE.BufferAttribute(safeVisibility, 1);
      geo.setAttribute('visibility', visibilityAttrRef.current);
    } else {
      visibilityAttrRef.current.array = safeVisibility;
      visibilityAttrRef.current.needsUpdate = true;
    }

    geo.computeBoundingSphere();
    const radius = geo.boundingSphere?.radius ?? 1;
    pickThresholdRef.current = Math.max(radius * 0.5, 0.5);
    return geo;
  }, [positions, colors, pointCount, colorPointCount, visibility]);

  // Cleanup on unmount only
  useEffect(() => {
    return () => {
      geometryRef.current?.dispose();
      geometryRef.current = null;
    };
  }, []);

  // Keep the R3F event raycaster's Points threshold in sync with the data scale,
  // so pointer events near a point actually reach our handlers. The old fixed
  // 0.1 was far smaller than the rendered points, so clicks were dropped.
  useEffect(() => {
    raycaster.params.Points = { threshold: pickThresholdRef.current };
  }, [raycaster, geometry]);

  // Custom shader material - crisp points with proper depth
  // Note: Do NOT include selectedIndex, pointSize, or hoverScale in dependencies
  // These are updated via useEffect to avoid recreating the material and causing
  // stale references in callbacks that capture the material
  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        size: { value: pointSize },
        hoverScale: { value: hoverScale },
        selectedIndex: { value: selectedIndex ?? -1 },
        hoveredIndex: { value: -1 },
        pointCount: { value: pointCount },
        maxPointSize: { value: 16.0 },  // Base max, will be scaled by sparsity
      },
      vertexColors: true,
      transparent: false,
      depthWrite: true,
      depthTest: true,
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Dispose material on unmount
  useEffect(() => {
    return () => {
      material.dispose();
    };
  }, [material]);

  // Update uniforms when props change
  useEffect(() => {
    if (material.uniforms) {
      material.uniforms.size.value = pointSize;
      material.uniforms.hoverScale.value = hoverScale;
      material.uniforms.selectedIndex.value = selectedIndex ?? -1;
      material.uniforms.pointCount.value = pointCount;
    }
  }, [material, pointSize, hoverScale, selectedIndex, pointCount]);

  // Throttle hover updates for performance
  const lastHoverTime = useRef(0);
  const HOVER_THROTTLE_MS = 16; // ~60fps

  // Screen-space nearest-point picking. Projects every visible point to pixels
  // and returns the index closest to the cursor within PICK_RADIUS_PX, or null.
  // This matches how points are actually drawn (screen-space sized), unlike a
  // world-space raycaster threshold.
  const pickNearest = useCallback(
    (clientX: number, clientY: number): number | null => {
      const rect = gl.domElement.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return null;
      const cursorX = clientX - rect.left;
      const cursorY = clientY - rect.top;

      let best = -1;
      let bestDist = PICK_RADIUS_PX;
      for (let i = 0; i < pointCount; i++) {
        if (visibility && visibility[i] < 0.5) continue;
        projVector.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        projVector.project(camera);
        // Skip points outside the clip volume (behind camera / clipped)
        if (projVector.z < -1 || projVector.z > 1) continue;
        const px = (projVector.x * 0.5 + 0.5) * rect.width;
        const py = (-projVector.y * 0.5 + 0.5) * rect.height;
        const dist = Math.hypot(px - cursorX, py - cursorY);
        if (dist < bestDist) {
          bestDist = dist;
          best = i;
        }
      }
      return best === -1 ? null : best;
    },
    [gl, camera, positions, pointCount, visibility]
  );

  // Hover detection via screen-space picking
  const handlePointerMove = useCallback(
    (event: ThreeEvent<PointerEvent>) => {
      // Skip during camera interaction for performance
      if (disablePointerEvents || !pointsRef.current) return;

      // Throttle hover detection
      const now = performance.now();
      if (now - lastHoverTime.current < HOVER_THROTTLE_MS) return;
      lastHoverTime.current = now;

      const index = pickNearest(event.clientX, event.clientY);

      if (index !== null && index !== hoveredIndex) {
        setHoveredIndex(index);
        material.uniforms.hoveredIndex.value = index;
        if (onHover) {
          // Reuse temp vector to avoid allocation
          tempVector.set(
            positions[index * 3],
            positions[index * 3 + 1],
            positions[index * 3 + 2]
          );
          onHover(index, tempVector, pointData[index] ?? null);
        }
      } else if (index === null && hoveredIndex !== null) {
        setHoveredIndex(null);
        material.uniforms.hoveredIndex.value = -1;
        if (onHover) {
          onHover(null, null, null);
        }
      }
    },
    [pickNearest, positions, pointData, onHover, hoveredIndex, material, disablePointerEvents]
  );

  const handlePointerOut = useCallback(() => {
    setHoveredIndex(null);
    material.uniforms.hoveredIndex.value = -1;
    if (onHover) {
      onHover(null, null, null);
    }
  }, [material, onHover]);

  const handleClick = useCallback(
    (event: ThreeEvent<MouseEvent>) => {
      if (!pointsRef.current || !onSelect) return;

      const index = pickNearest(event.clientX, event.clientY);
      if (index !== null) {
        // Reuse temp vector
        tempVector.set(
          positions[index * 3],
          positions[index * 3 + 1],
          positions[index * 3 + 2]
        );
        onSelect(index, tempVector, pointData[index] ?? null);
      } else {
        onSelect(null, null, null);
      }
    },
    [pickNearest, positions, pointData, onSelect]
  );

  // Update cursor style based on hover
  useEffect(() => {
    gl.domElement.style.cursor = hoveredIndex !== null ? 'pointer' : 'default';
    return () => {
      gl.domElement.style.cursor = 'default';
    };
  }, [hoveredIndex, gl]);

  return (
    <points
      ref={pointsRef}
      geometry={geometry}
      material={material}
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
      onClick={handleClick}
    />
  );
}

// Memoize to prevent unnecessary re-renders
export const PointCloud = memo(PointCloudInner);
export default PointCloud;
