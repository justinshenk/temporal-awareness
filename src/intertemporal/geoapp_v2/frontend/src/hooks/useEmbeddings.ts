import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';
import { api } from '../lib/api';

// Types matching the actual backend API responses

export interface ConfigResponse {
  layers: number[];
  components: string[];
  positions: string[];
  color_options: string[];
  n_samples: number;
}

// Transformed config for easier use in components
export interface TransformedConfig {
  layers: number[];
  components: string[];
  positions: string[];
  methods: string[];
  colorByOptions: string[];
  totalSamples: number;
}

interface Point3D {
  x: number;
  y: number;
  z: number;
}

interface BackendEmbeddingResponse {
  layer: number;
  component: string;
  position: string;
  method: string;
  n_samples: number;
  coordinates: Point3D[];
}

export interface EmbeddingResponse {
  positions: number[];  // Flat array [x1, y1, z1, x2, y2, z2, ...]
  indices: number[];    // Sample indices
  metrics: {
    varianceExplained?: number[];
    r2Score?: number;
  };
}

interface BackendMetadataResponse {
  color_by: string;
  values: (number | string | boolean)[];
  dtype: 'numeric' | 'categorical' | 'boolean';
}

export interface MetadataResponse {
  values: number[];        // Color values for each point (normalized)
  labels?: string[];       // Optional labels for categorical data
  min: number;
  max: number;
}

interface BackendSampleResponse {
  idx: number;
  text: string;
  time_horizon_months: number | null;
  time_scale: string | null;
  choice_type: string | null;
  short_term_first: boolean | null;
  metadata: Record<string, unknown>;
}

export interface SampleResponse {
  idx: number;
  text: string;
  timeHorizon: number | null;
  timeScale: string | null;
  choiceType: string | null;
  shortTermFirst: boolean | null;
  label?: string;
}

// Hook to fetch app configuration
export function useConfig() {
  return useQuery({
    queryKey: ['config'],
    queryFn: async (): Promise<TransformedConfig> => {
      const response = await api.get<ConfigResponse>('/config');
      return {
        layers: response.layers,
        components: response.components,
        positions: response.positions,
        methods: ['pca', 'umap', 'tsne'],  // Fixed methods supported by backend
        colorByOptions: response.color_options,
        totalSamples: response.n_samples,
      };
    },
    staleTime: Infinity, // Config doesn't change during session
  });
}

// Hook to fetch 3D embedding coordinates
export function useEmbedding(
  layer: number,
  component: string,
  position: string,
  method: string
) {
  return useQuery({
    queryKey: ['embedding', layer, component, position, method],
    queryFn: async (): Promise<EmbeddingResponse> => {
      // Backend uses path params: /embedding/{layer}/{component}/{position}?method=pca
      const response = await api.get<BackendEmbeddingResponse>(
        `/embedding/${layer}/${component}/${position}?method=${method.toLowerCase()}`
      );

      // Transform Point3D array to flat Float32Array-compatible format
      const positions: number[] = [];
      const indices: number[] = [];

      response.coordinates.forEach((coord, i) => {
        positions.push(coord.x, coord.y, coord.z);
        indices.push(i);
      });

      return {
        positions,
        indices,
        metrics: {},  // Backend doesn't return metrics in embedding endpoint
      };
    },
    enabled: layer >= 0 && !!component && !!position && !!method,
  });
}

// Hook to fetch metadata for coloring points
export function useMetadata(colorBy: string) {
  return useQuery({
    queryKey: ['metadata', colorBy],
    queryFn: async (): Promise<MetadataResponse> => {
      // Backend uses snake_case: color_by
      const response = await api.get<BackendMetadataResponse>(
        `/metadata?color_by=${colorBy}`
      );

      // Transform based on dtype
      let numericValues: number[];
      let labels: string[] | undefined;
      let min = 0;
      let max = 1;

      if (response.dtype === 'numeric') {
        numericValues = response.values as number[];
        min = Math.min(...numericValues);
        max = Math.max(...numericValues);
      } else if (response.dtype === 'boolean') {
        numericValues = (response.values as boolean[]).map(v => v ? 1 : 0);
        labels = ['false', 'true'];
        min = 0;
        max = 1;
      } else {
        // Categorical - create numeric mapping
        const uniqueVals = [...new Set(response.values as string[])];
        labels = uniqueVals;
        numericValues = (response.values as string[]).map(v =>
          uniqueVals.indexOf(v)
        );
        min = 0;
        max = uniqueVals.length - 1;
      }

      return {
        values: numericValues,
        labels,
        min,
        max,
      };
    },
    enabled: !!colorBy,
  });
}

// Hook to fetch sample details
export function useSample(idx: number | null) {
  return useQuery({
    queryKey: ['sample', idx],
    queryFn: async (): Promise<SampleResponse> => {
      const response = await api.get<BackendSampleResponse>(`/sample/${idx}`);
      return {
        idx: response.idx,
        text: response.text,
        timeHorizon: response.time_horizon_months,
        timeScale: response.time_scale,
        choiceType: response.choice_type,
        shortTermFirst: response.short_term_first,
      };
    },
    enabled: idx !== null,
  });
}

// Hook to fetch all samples for filtering
export function useSamples() {
  return useQuery({
    queryKey: ['samples'],
    queryFn: () => api.get<SampleResponse[]>('/samples'),
  });
}

// Helper to convert flat array to Float32Array for Three.js
export function toFloat32Array(arr: number[]): Float32Array {
  return new Float32Array(arr);
}

// Helper to generate colors from values
export function valuesToColors(
  values: number[],
  min: number,
  max: number,
  colormap: 'viridis' | 'plasma' | 'turbo' = 'viridis'
): Float32Array {
  const colors = new Float32Array(values.length * 3);
  const range = max - min || 1;

  for (let i = 0; i < values.length; i++) {
    const t = (values[i] - min) / range;
    const [r, g, b] = getColormapColor(t, colormap);
    colors[i * 3] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }

  return colors;
}

// Colormap implementations
function getColormapColor(
  t: number,
  colormap: 'viridis' | 'plasma' | 'turbo'
): [number, number, number] {
  t = Math.max(0, Math.min(1, t));

  switch (colormap) {
    case 'viridis':
      return viridis(t);
    case 'plasma':
      return plasma(t);
    case 'turbo':
      return turbo(t);
    default:
      return viridis(t);
  }
}

function viridis(t: number): [number, number, number] {
  // Simplified viridis colormap
  const r = Math.max(0, Math.min(1, 0.267 + 0.004 * t + t * t * (0.329 - 0.6 * t)));
  const g = Math.max(0, Math.min(1, 0.004 + t * (0.873 - 0.385 * t)));
  const b = Math.max(0, Math.min(1, 0.329 + t * (0.42 + t * (-0.749 + 0.35 * t))));
  return [r, g, b];
}

function plasma(t: number): [number, number, number] {
  // Simplified plasma colormap
  const r = Math.max(0, Math.min(1, 0.05 + t * (1.1 - 0.3 * t)));
  const g = Math.max(0, Math.min(1, t * t * 0.8));
  const b = Math.max(0, Math.min(1, 0.53 + t * (-0.8 + 0.6 * t)));
  return [r, g, b];
}

function turbo(t: number): [number, number, number] {
  // Simplified turbo colormap
  const r = Math.max(0, Math.min(1, 0.13 + t * (2.5 - t * 2)));
  const g = Math.max(0, Math.min(1, 0.13 + t * (1.2 + t * (0.5 - t))));
  const b = Math.max(0, Math.min(1, 0.53 + t * (-0.6 - t * 0.3)));
  return [r, g, b];
}

// Generate categorical colors
export function categoricalColors(
  values: number[],
  _numCategories: number
): Float32Array {
  const colors = new Float32Array(values.length * 3);

  // Define a palette for categories
  const palette = [
    [0.78, 0.47, 0.87], // Primary purple
    [1.0, 0.42, 0.62],  // Primary pink
    [0.34, 0.71, 0.76], // Primary cyan
    [0.98, 0.73, 0.01], // Yellow
    [0.3, 0.69, 0.31],  // Green
    [0.94, 0.5, 0.5],   // Light red
    [0.59, 0.59, 0.8],  // Light purple
    [1.0, 0.65, 0.4],   // Orange
  ];

  for (let i = 0; i < values.length; i++) {
    const categoryIdx = Math.floor(values[i]) % palette.length;
    const color = palette[categoryIdx];
    colors[i * 3] = color[0];
    colors[i * 3 + 1] = color[1];
    colors[i * 3 + 2] = color[2];
  }

  return colors;
}

// Prefetching hook for background loading
export function usePrefetch(
  currentLayer: number,
  component: string,
  position: string,
  method: string,
  colorByOptions: string[],
  layers: number[]
) {
  const queryClient = useQueryClient();

  useEffect(() => {
    // Prefetch all color options in background
    colorByOptions.forEach((colorBy) => {
      queryClient.prefetchQuery({
        queryKey: ['metadata', colorBy],
        queryFn: async (): Promise<MetadataResponse> => {
          const response = await api.get<BackendMetadataResponse>(
            `/metadata?color_by=${colorBy}`
          );
          let numericValues: number[];
          let labels: string[] | undefined;
          let min = 0;
          let max = 1;

          if (response.dtype === 'numeric') {
            numericValues = response.values as number[];
            min = Math.min(...numericValues);
            max = Math.max(...numericValues);
          } else if (response.dtype === 'boolean') {
            numericValues = (response.values as boolean[]).map(v => v ? 1 : 0);
            labels = ['false', 'true'];
          } else {
            const uniqueVals = [...new Set(response.values as string[])];
            labels = uniqueVals;
            numericValues = (response.values as string[]).map(v =>
              uniqueVals.indexOf(v)
            );
            max = uniqueVals.length - 1;
          }
          return { values: numericValues, labels, min, max };
        },
        staleTime: 1000 * 60 * 10, // 10 minutes
      });
    });

    // Prefetch adjacent layers in background
    const currentIdx = layers.indexOf(currentLayer);
    const adjacentLayers = [
      layers[currentIdx - 1],
      layers[currentIdx + 1],
    ].filter(l => l !== undefined);

    adjacentLayers.forEach((layer) => {
      queryClient.prefetchQuery({
        queryKey: ['embedding', layer, component, position, method],
        queryFn: async (): Promise<EmbeddingResponse> => {
          const response = await api.get<BackendEmbeddingResponse>(
            `/embedding/${layer}/${component}/${position}?method=${method.toLowerCase()}`
          );
          const positions: number[] = [];
          const indices: number[] = [];
          response.coordinates.forEach((coord, i) => {
            positions.push(coord.x, coord.y, coord.z);
            indices.push(i);
          });
          return { positions, indices, metrics: {} };
        },
        staleTime: 1000 * 60 * 5, // 5 minutes
      });
    });
  }, [queryClient, currentLayer, component, position, method, colorByOptions, layers]);
}
