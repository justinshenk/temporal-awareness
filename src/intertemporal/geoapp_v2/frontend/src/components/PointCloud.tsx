import { useRef, useMemo, useCallback, useState, useEffect } from 'react';
import { useThree, ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';

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
}

// Custom shader for crisp point rendering with zoom-based scaling
const vertexShader = `
  attribute float pointIndex;

  uniform float size;
  uniform float hoverScale;
  uniform float selectedIndex;
  uniform float hoveredIndex;

  varying vec3 vColor;
  varying float vIsSelected;
  varying float vIsHovered;

  void main() {
    vColor = color;

    vIsSelected = (pointIndex == selectedIndex) ? 1.0 : 0.0;
    vIsHovered = (pointIndex == hoveredIndex) ? 1.0 : 0.0;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

    // Scale up selected/hovered points
    float scale = 1.0;
    if (vIsSelected > 0.5) scale = hoverScale * 1.5;
    else if (vIsHovered > 0.5) scale = hoverScale;

    // Scale points based on distance - closer = bigger
    // Use perspective scaling so zooming in makes points larger
    float distanceScale = 200.0 / max(-mvPosition.z, 1.0);
    gl_PointSize = size * scale * distanceScale;

    // Clamp to reasonable range
    gl_PointSize = clamp(gl_PointSize, 2.0, 50.0);

    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = `
  varying vec3 vColor;
  varying float vIsSelected;
  varying float vIsHovered;

  void main() {
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

export function PointCloud({
  positions,
  colors,
  pointSize = 4,
  pointData = [],
  onHover,
  onSelect,
  selectedIndex = null,
  hoverScale = 1.5,
}: PointCloudProps) {
  const pointsRef = useRef<THREE.Points>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const { raycaster, gl } = useThree();

  const pointCount = positions.length / 3;

  // Create geometry with point indices for hover detection
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Add point indices for shader
    const indices = new Float32Array(pointCount);
    for (let i = 0; i < pointCount; i++) {
      indices[i] = i;
    }
    geo.setAttribute('pointIndex', new THREE.BufferAttribute(indices, 1));

    return geo;
  }, [positions, colors, pointCount]);

  // Custom shader material - crisp points with proper depth
  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        size: { value: pointSize },
        hoverScale: { value: hoverScale },
        selectedIndex: { value: selectedIndex ?? -1 },
        hoveredIndex: { value: -1 },
      },
      vertexColors: true,
      transparent: false,
      depthWrite: true,
      depthTest: true,
    });
  }, [pointSize, hoverScale, selectedIndex]);

  // Update uniforms when props change
  useEffect(() => {
    if (material.uniforms) {
      material.uniforms.size.value = pointSize;
      material.uniforms.hoverScale.value = hoverScale;
      material.uniforms.selectedIndex.value = selectedIndex ?? -1;
    }
  }, [material, pointSize, hoverScale, selectedIndex]);

  // Raycasting for hover detection
  const handlePointerMove = useCallback(
    (_event: ThreeEvent<PointerEvent>) => {
      if (!pointsRef.current) return;

      // Set raycaster threshold based on point size
      raycaster.params.Points = { threshold: 0.1 };

      const intersects = raycaster.intersectObject(pointsRef.current);

      if (intersects.length > 0) {
        const index = intersects[0].index;
        if (index !== undefined && index !== hoveredIndex) {
          setHoveredIndex(index);
          material.uniforms.hoveredIndex.value = index;

          if (onHover) {
            const point = new THREE.Vector3(
              positions[index * 3],
              positions[index * 3 + 1],
              positions[index * 3 + 2]
            );
            onHover(index, point, pointData[index] ?? null);
          }
        }
      } else if (hoveredIndex !== null) {
        setHoveredIndex(null);
        material.uniforms.hoveredIndex.value = -1;
        if (onHover) {
          onHover(null, null, null);
        }
      }
    },
    [raycaster, positions, pointData, onHover, hoveredIndex, material]
  );

  const handlePointerOut = useCallback(() => {
    setHoveredIndex(null);
    material.uniforms.hoveredIndex.value = -1;
    if (onHover) {
      onHover(null, null, null);
    }
  }, [material, onHover]);

  const handleClick = useCallback(
    (_event: ThreeEvent<MouseEvent>) => {
      if (!pointsRef.current || !onSelect) return;

      raycaster.params.Points = { threshold: 0.1 };
      const intersects = raycaster.intersectObject(pointsRef.current);

      if (intersects.length > 0) {
        const index = intersects[0].index;
        if (index !== undefined) {
          const point = new THREE.Vector3(
            positions[index * 3],
            positions[index * 3 + 1],
            positions[index * 3 + 2]
          );
          onSelect(index, point, pointData[index] ?? null);
        }
      } else {
        onSelect(null, null, null);
      }
    },
    [raycaster, positions, pointData, onSelect]
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

export default PointCloud;
