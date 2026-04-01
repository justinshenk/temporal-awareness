import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useMemo, useState, useRef } from 'react';
import api, { getDataset } from '../lib/api';

// Re-export ApiError for use in components
export { ApiError } from '../lib/api';

// Logging helper
const log = (category: string, message: string, data?: Record<string, unknown>) => {
  const ts = new Date().toISOString().slice(11, 23);
  const dataStr = data ? ` | ${Object.entries(data).map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' ')}` : '';
  console.log(`[${ts}] [CLIENT] [${category}] ${message}${dataStr}`);
};

// Types matching the actual backend API responses

export interface PromptTemplateElement {
  name: string;
  label: string;
  type: 'marker' | 'variable' | 'static' | 'semantic';
  available: boolean;
}

export interface ExampleSample {
  sample_idx: number;
  format_texts: Record<string, string>;
}

export interface ConfigResponse {
  layers: number[];
  components: string[];
  positions: string[];
  color_options: string[];
  available_methods: string[];
  n_samples: number;
  model_name: string;
  position_labels: Record<string, string>;
  prompt_template: PromptTemplateElement[];
  semantic_to_positions: Record<string, string[]>;
  markers: Record<string, string>;
  rel_pos_counts: Record<string, number>;
  example_sample: ExampleSample | null;
}

// Transformed config for easier use in components
export interface TransformedConfig {
  layers: number[];
  components: string[];
  positions: string[];
  methods: string[];
  colorByOptions: string[];
  totalSamples: number;
  modelName: string;
  positionLabels: Record<string, string>;
  promptTemplate: PromptTemplateElement[];
  semanticToPositions: Record<string, string[]>;
  markers: Record<string, string>;
  relPosCounts: Record<string, number>;
  exampleSample: ExampleSample | null;
}

interface BackendEmbeddingResponse {
  layer: number;
  component: string;
  position: string;
  method: string;
  n_samples: number;
  coordinates_flat: number[];  // Flat array [x0,y0,z0,x1,y1,z1,...]
  sample_indices: number[];
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
  values: (number | string | boolean | null)[];
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
  response_label: string | null;
  response_term: string | null;
  response_text: string | null;
  choice_confidence: number | null;
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
  responseLabel: string | null;
  responseTerm: string | null;
  responseText: string | null;
  choiceConfidence: number | null;
  metadata: Record<string, unknown>;
}

// Hook to fetch app configuration
export function useConfig() {
  const dataset = getDataset();
  return useQuery({
    queryKey: ['config', dataset],
    queryFn: async (): Promise<TransformedConfig> => {
      log('useConfig', `Fetching config for dataset: ${dataset}`);
      const startTime = performance.now();
      const response = await api.get<ConfigResponse>('/config');
      const elapsed = performance.now() - startTime;
      log('useConfig', 'Config loaded', {
        n_layers: response.layers.length,
        n_positions: response.positions.length,
        n_samples: response.n_samples,
        elapsed_ms: elapsed.toFixed(1)
      });
      return {
        layers: response.layers,
        components: response.components,
        positions: response.positions,
        methods: response.available_methods || ['pca'],
        colorByOptions: response.color_options,
        totalSamples: response.n_samples,
        modelName: response.model_name || '',
        positionLabels: response.position_labels || {},
        promptTemplate: response.prompt_template || [],
        semanticToPositions: response.semantic_to_positions || {},
        markers: response.markers || {},
        relPosCounts: response.rel_pos_counts || {},
        exampleSample: response.example_sample || null,
      };
    },
    staleTime: 5 * 60 * 1000, // 5 minutes - allow refresh if something goes wrong
    refetchOnMount: true, // Always refetch when component mounts
    retry: 1, // Retry once on failure
    retryDelay: 500,
  });
}

// Hook to fetch 3D embedding coordinates
// CRASH BEHAVIOR: No retries - if embedding data is missing, app crashes immediately
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

      // Backend now returns flat array directly - no transformation needed
      return {
        positions: response.coordinates_flat,
        // Use actual sample indices from backend, not just 0,1,2,...
        indices: response.sample_indices,
        metrics: {},  // Backend doesn't return metrics in embedding endpoint
      };
    },
    enabled: layer >= 0 && !!component && !!position && !!method,
    staleTime: 1000 * 60 * 30, // 30 minutes - embeddings are expensive to compute
    gcTime: 1000 * 60 * 60, // 1 hour - keep in cache for a long time
    retry: false, // CRASH: No retries - fail immediately if data is missing
  });
}

// Streaming embedding response type
export interface StreamingEmbeddingState {
  positions: number[];
  indices: number[];
  totalPoints: number;
  loadedPoints: number;
  isStreaming: boolean;
  isComplete: boolean;
  error: Error | null;
  progress: number; // 0-100
}

// Global cache for streaming embedding data (persists across component re-renders)
const streamingCache = new Map<string, StreamingEmbeddingState>();

// Hook for streaming embedding data via SSE
export function useStreamingEmbedding(
  layer: number,
  component: string,
  position: string,
  method: string,
  enabled: boolean = true
) {
  const currentParams = `${layer}|${component}|${position}|${method}`;

  // Check global cache first for instant restore
  const cachedData = streamingCache.get(currentParams);

  const [state, setState] = useState<StreamingEmbeddingState>(() => {
    // Initialize from cache if available
    if (cachedData && cachedData.isComplete) {
      log('useStreamingEmbedding', `💾 CACHE HIT (init): L${layer}/${position}`, { loadedPoints: cachedData.loadedPoints });
      return cachedData;
    }
    return {
      positions: [],
      indices: [],
      totalPoints: 0,
      loadedPoints: 0,
      isStreaming: false,
      isComplete: false,
      error: null,
      progress: 0,
    };
  });

  const streamStartTime = useRef<number>(0);
  const activeEventSource = useRef<EventSource | null>(null);

  useEffect(() => {
    // Skip fetching if disabled or invalid params
    if (!enabled || layer < 0 || !component || !position || !method) return;

    // Check global cache - if we have complete data, use it immediately
    const cached = streamingCache.get(currentParams);
    if (cached && cached.isComplete) {
      log('useStreamingEmbedding', `💾 CACHE HIT: L${layer}/${position}`, { loadedPoints: cached.loadedPoints });
      setState(cached);
      return;
    }

    log('useStreamingEmbedding', `🚀 STARTING SSE: L${layer}/${component}/${position} (${method})`);
    streamStartTime.current = performance.now();

    // Close any existing connection
    if (activeEventSource.current) {
      activeEventSource.current.close();
      activeEventSource.current = null;
    }

    // Reset state for new request
    setState({
      positions: [],
      indices: [],
      totalPoints: 0,
      loadedPoints: 0,
      isStreaming: true,
      isComplete: false,
      error: null,
      progress: 0,
    });

    const dataset = getDataset();
    const url = `/api/${dataset}/embedding/${layer}/${component}/${encodeURIComponent(position)}/stream?method=${method}&chunk_size=500`;
    log('useStreamingEmbedding', `📡 Opening EventSource: ${url}`);
    const eventSource = new EventSource(url);
    activeEventSource.current = eventSource;

    let allPositions: number[] = [];
    let allIndices: number[] = [];
    let totalPoints = 0;

    let lastReportedProgress = 0;
    let chunkCount = 0;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'metadata') {
          totalPoints = data.total_points;
          log('useStreamingEmbedding', 'Received metadata', { totalPoints, chunk_size: data.chunk_size });
          // Pre-allocate arrays with zeros (NOT sparse arrays!)
          // Using Array(n).fill(0) ensures all slots have numeric values
          allPositions = new Array(totalPoints * 3).fill(0);
          allIndices = new Array(totalPoints).fill(0);
          setState(prev => ({
            ...prev,
            totalPoints,
          }));
        } else if (data.type === 'chunk') {
          chunkCount++;
          // Copy chunk data into pre-allocated arrays
          const coords = data.coordinates as number[];
          const indices = data.sample_indices as number[];
          const startIdx = data.start_idx as number;

          for (let i = 0; i < indices.length; i++) {
            allIndices[startIdx + i] = indices[i];
            allPositions[(startIdx + i) * 3] = coords[i * 3];
            allPositions[(startIdx + i) * 3 + 1] = coords[i * 3 + 1];
            allPositions[(startIdx + i) * 3 + 2] = coords[i * 3 + 2];
          }

          const loadedPoints = data.end_idx as number;
          const progress = Math.round((loadedPoints / totalPoints) * 100);

          // Batch updates: only update every 5% or on first chunk
          const shouldUpdate = progress === 100 || progress - lastReportedProgress >= 5 || lastReportedProgress === 0;
          if (shouldUpdate) {
            lastReportedProgress = progress;
            log('useStreamingEmbedding', `Chunk ${chunkCount}`, { progress: `${progress}%`, loadedPoints });
            // Update state with current progress - spread to trigger re-render
            setState(prev => ({
              ...prev,
              positions: [...allPositions.slice(0, loadedPoints * 3)],
              indices: [...allIndices.slice(0, loadedPoints)],
              loadedPoints,
              progress,
            }));
          }
        } else if (data.type === 'complete') {
          const elapsed = performance.now() - streamStartTime.current;
          log('useStreamingEmbedding', 'Stream complete', {
            totalPoints,
            totalChunks: chunkCount,
            elapsed_ms: elapsed.toFixed(1)
          });
          const completedState: StreamingEmbeddingState = {
            positions: allPositions,
            indices: allIndices,
            totalPoints,
            loadedPoints: totalPoints,
            isStreaming: false,
            isComplete: true,
            error: null,
            progress: 100,
          };
          // Store in global cache for instant restore later
          streamingCache.set(currentParams, completedState);
          log('useStreamingEmbedding', `💾 CACHED: ${currentParams}`, { cacheSize: streamingCache.size });
          setState(completedState);
          eventSource.close();
          activeEventSource.current = null;
        } else if (data.error) {
          log('useStreamingEmbedding', 'Stream error', { error: data.error });
          setState(prev => ({
            ...prev,
            isStreaming: false,
            error: new Error(data.error),
          }));
          eventSource.close();
        }
      } catch (e) {
        console.error('SSE parse error:', e);
      }
    };

    eventSource.onerror = () => {
      log('useStreamingEmbedding', 'SSE connection failed');
      setState(prev => ({
        ...prev,
        isStreaming: false,
        error: new Error('Stream connection failed'),
      }));
      eventSource.close();
      activeEventSource.current = null;
    };

    return () => {
      eventSource.close();
      activeEventSource.current = null;
    };
  }, [layer, component, position, method, enabled, currentParams]);

  return state;
}

// Hook to fetch metadata for coloring points
// CRASH BEHAVIOR: No retries - if metadata is missing, app crashes immediately
export function useMetadata(colorBy: string) {
  return useQuery({
    queryKey: ['metadata', colorBy],
    staleTime: 1000 * 60 * 60, // 1 hour - metadata doesn't change
    gcTime: 1000 * 60 * 120, // 2 hours
    retry: false, // CRASH: No retries - fail immediately if data is missing
    queryFn: async (): Promise<MetadataResponse> => {
      log('useMetadata', 'Fetching metadata', { colorBy });
      const startTime = performance.now();
      // Backend uses snake_case: color_by
      const response = await api.get<BackendMetadataResponse>(
        `/metadata?color_by=${colorBy}`
      );
      const elapsed = performance.now() - startTime;
      log('useMetadata', 'Metadata loaded', { colorBy, dtype: response.dtype, n_values: response.values.length, elapsed_ms: elapsed.toFixed(1) });

      // Transform based on dtype
      let numericValues: number[];
      let labels: string[] | undefined;
      let min = 0;
      let max = 1;

      if (response.dtype === 'numeric') {
        // For time_horizon fields, backend sends null for no-horizon samples
        // Convert null to -1 (sentinel value) so timeGradientColors can identify them
        const isTimeField = colorBy === 'time_horizon' ;
        numericValues = (response.values as (number | null)[]).map(v =>
          v === null ? -1 : v
        );


        // For min/max calculation, exclude sentinel values (-1)
        const validValues = isTimeField
          ? numericValues.filter(v => v >= 0)
          : numericValues;

        if (validValues.length > 0) {
          min = Math.min(...validValues);
          max = Math.max(...validValues);
        }
        // If empty, keep defaults min=0, max=1
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
        // Ensure max >= 1 to avoid division by zero in color calculations
        max = Math.max(1, uniqueVals.length - 1);
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
// CRASH BEHAVIOR: No retries - if sample data is missing, app crashes immediately
export function useSample(idx: number | null) {
  return useQuery({
    queryKey: ['sample', idx],
    queryFn: async (): Promise<SampleResponse> => {
      const response = await api.get<BackendSampleResponse>(`/sample/${idx}`);
      const metadata = response.metadata || {};
      return {
        idx: response.idx,
        text: response.text,
        timeHorizon: response.time_horizon_months,
        timeScale: response.time_scale,
        choiceType: response.choice_type,
        shortTermFirst: response.short_term_first,
        label: typeof metadata.label === 'string' ? metadata.label : undefined,
        responseLabel: response.response_label,
        responseTerm: response.response_term,
        responseText: response.response_text,
        choiceConfidence: response.choice_confidence,
        metadata,
      };
    },
    enabled: idx !== null,
    retry: false, // CRASH: No retries - fail immediately if data is missing
  });
}

// Hook to fetch all samples for filtering
// CRASH BEHAVIOR: No retries - if samples data is missing, app crashes immediately
export function useSamples() {
  return useQuery({
    queryKey: ['samples'],
    queryFn: () => api.get<SampleResponse[]>('/samples'),
    retry: false, // CRASH: No retries - fail immediately if data is missing
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
  // Accurate Viridis colormap from matplotlib
  // Interpolated from official viridis color stops
  const stops = [
    [0.267004, 0.004874, 0.329415],
    [0.282327, 0.140926, 0.457517],
    [0.253935, 0.265254, 0.529983],
    [0.206756, 0.371758, 0.553117],
    [0.163625, 0.471133, 0.558148],
    [0.127568, 0.566949, 0.550556],
    [0.134692, 0.658636, 0.517649],
    [0.266941, 0.748751, 0.440573],
    [0.477504, 0.821444, 0.318195],
    [0.741388, 0.873449, 0.149561],
    [0.993248, 0.906157, 0.143936],
  ];

  t = Math.max(0, Math.min(1, t));
  const idx = t * (stops.length - 1);
  const i = Math.floor(idx);
  const f = idx - i;

  if (i >= stops.length - 1) return stops[stops.length - 1] as [number, number, number];
  if (i < 0) return stops[0] as [number, number, number];

  const c0 = stops[i];
  const c1 = stops[i + 1];

  return [
    c0[0] + f * (c1[0] - c0[0]),
    c0[1] + f * (c1[1] - c0[1]),
    c0[2] + f * (c1[2] - c0[2]),
  ];
}

function plasma(t: number): [number, number, number] {
  // Accurate Plasma colormap from matplotlib
  // Interpolated from official plasma color stops
  const stops = [
    [0.050383, 0.029803, 0.527975],
    [0.254627, 0.013882, 0.615419],
    [0.417642, 0.000564, 0.658390],
    [0.562738, 0.051545, 0.641509],
    [0.692840, 0.165141, 0.564522],
    [0.798216, 0.280197, 0.469538],
    [0.881443, 0.392529, 0.383229],
    [0.949217, 0.517763, 0.295662],
    [0.988260, 0.652325, 0.211364],
    [0.988648, 0.809579, 0.145357],
    [0.940015, 0.975158, 0.131326],
  ];

  const idx = t * (stops.length - 1);
  const i = Math.floor(idx);
  const f = idx - i;

  if (i >= stops.length - 1) return stops[stops.length - 1] as [number, number, number];
  if (i < 0) return stops[0] as [number, number, number];

  const c0 = stops[i];
  const c1 = stops[i + 1];

  return [
    c0[0] + f * (c1[0] - c0[0]),
    c0[1] + f * (c1[1] - c0[1]),
    c0[2] + f * (c1[2] - c0[2]),
  ];
}

function turbo(t: number): [number, number, number] {
  // Accurate Turbo colormap from Google
  // Interpolated from official turbo color stops
  const stops = [
    [0.18995, 0.07176, 0.23217],
    [0.25107, 0.25237, 0.63374],
    [0.19097, 0.40774, 0.85766],
    [0.08552, 0.53310, 0.87085],
    [0.16354, 0.68323, 0.72642],
    [0.36110, 0.79945, 0.52558],
    [0.56532, 0.86851, 0.32241],
    [0.77377, 0.89730, 0.14590],
    [0.94290, 0.82507, 0.11454],
    [0.99218, 0.64896, 0.14537],
    [0.94890, 0.43070, 0.10159],
  ];

  t = Math.max(0, Math.min(1, t));
  const idx = t * (stops.length - 1);
  const i = Math.floor(idx);
  const f = idx - i;

  if (i >= stops.length - 1) return stops[stops.length - 1] as [number, number, number];
  if (i < 0) return stops[0] as [number, number, number];

  const c0 = stops[i];
  const c1 = stops[i + 1];

  return [
    c0[0] + f * (c1[0] - c0[0]),
    c0[1] + f * (c1[1] - c0[1]),
    c0[2] + f * (c1[2] - c0[2]),
  ];
}

// Generate categorical colors
export function categoricalColors(
  values: number[],
  _numCategories: number
): Float32Array {
  const colors = new Float32Array(values.length * 3);

  // Define a palette for categories - Anthropic-inspired with high contrast
  const palette = [
    [0.85, 0.47, 0.34], // Anthropic terracotta/coral #D97757
    [0.20, 0.51, 0.59], // Deep teal #348296 (high contrast with terracotta)
    [0.56, 0.44, 0.86], // Purple #8F70DB
    [0.95, 0.68, 0.26], // Warm amber #F2AD42
    [0.35, 0.65, 0.47], // Forest green #59A678
    [0.94, 0.5, 0.5],   // Light red
    [0.59, 0.59, 0.8],  // Light purple
    [1.0, 0.65, 0.4],   // Orange
  ];

  for (let i = 0; i < values.length; i++) {
    // Handle NaN, negative values, and ensure valid index
    const value = values[i];
    const safeValue = Number.isFinite(value) ? Math.max(0, Math.floor(value)) : 0;
    const categoryIdx = safeValue % palette.length;
    const color = palette[categoryIdx];
    colors[i * 3] = color[0];
    colors[i * 3 + 1] = color[1];
    colors[i * 3 + 2] = color[2];
  }

  return colors;
}

// Time scale types for transfer functions
export type TimeScaleType = 'linear' | 'log' | 'adaptive' | 'blend';

// Time tier breakpoints in months for scale-adaptive approach
// Each tier gets equal visual space regardless of absolute duration
const TIME_TIERS = [
  { name: 'seconds', maxMonths: 1 / 2592000 },      // 1 second in months
  { name: 'minutes', maxMonths: 1 / 43200 },        // 1 minute in months
  { name: 'hours', maxMonths: 1 / 720 },            // 1 hour in months
  { name: 'days', maxMonths: 1 / 30 },              // 1 day in months
  { name: 'weeks', maxMonths: 1 / 4.3 },            // 1 week in months
  { name: 'months', maxMonths: 1 },                 // 1 month
  { name: 'years', maxMonths: 12 },                 // 1 year in months
  { name: 'decades', maxMonths: 120 },              // 10 years
  { name: 'centuries', maxMonths: 1200 },           // 100 years
  { name: 'millennia', maxMonths: 12000 },          // 1000 years
  { name: 'deep_time', maxMonths: Infinity },       // Beyond millennia
];

// Transfer function: linear
function linearTransfer(value: number, min: number, max: number): number {
  // When all values are the same (min === max), return middle of range
  if (max === min) return 0.5;
  return (value - min) / (max - min);
}

// Transfer function: log
function logTransfer(value: number, min: number, max: number): number {
  // Handle edge cases: zero/negative values can't be log-transformed
  if (value <= 0) return 0;
  if (max <= 0) return 0;

  // Use a small epsilon for min to handle near-zero minimums
  const epsilon = 1e-10;
  const safeMin = Math.max(min, epsilon);
  const safeMax = Math.max(max, epsilon);
  const safeVal = Math.max(value, epsilon);

  const logMin = Math.log10(safeMin);
  const logMax = Math.log10(safeMax);
  const logVal = Math.log10(safeVal);

  // Avoid division by zero when min === max
  if (logMax === logMin) return 0.5;

  return (logVal - logMin) / (logMax - logMin);
}

// Transfer function: scale-adaptive (equal visual space per tier)
function adaptiveTransfer(valueMonths: number, _min: number, _max: number): number {
  if (valueMonths <= 0) return 0;

  // Find which tier this value belongs to
  let tierIdx = TIME_TIERS.length - 1; // Default to last tier
  for (let i = 0; i < TIME_TIERS.length; i++) {
    if (valueMonths <= TIME_TIERS[i].maxMonths) {
      tierIdx = i;
      break;
    }
  }

  // Calculate position within tier using log interpolation
  const tierStart = tierIdx === 0 ? 0 : TIME_TIERS[tierIdx - 1].maxMonths;
  const tierEnd = TIME_TIERS[tierIdx].maxMonths;

  // Position within this tier (0-1)
  let withinTier: number;
  if (!isFinite(tierEnd)) {
    // For the infinite tier (deep_time), use log position relative to tier start
    // Map values from tierStart to tierStart*1000 across the tier
    const logStart = Math.log10(Math.max(tierStart, 1e-10));
    const logEnd = logStart + 3; // 3 orders of magnitude above tier start
    const logVal = Math.log10(Math.max(valueMonths, tierStart));
    withinTier = Math.max(0, Math.min(1, (logVal - logStart) / (logEnd - logStart)));
  } else if (tierIdx === 0) {
    // First tier: value ranges from near-zero to tierEnd
    // Use linear interpolation since log of near-zero is problematic
    withinTier = Math.max(0, Math.min(1, valueMonths / tierEnd));
  } else {
    // Normal tier: use log interpolation within tier for smoothness
    const epsilon = 1e-10;
    const logStart = Math.log10(Math.max(tierStart, epsilon));
    const logEnd = Math.log10(tierEnd);
    const logVal = Math.log10(Math.max(valueMonths, tierStart));

    // Avoid division by zero when tier boundaries are equal (shouldn't happen normally)
    if (logEnd === logStart) {
      withinTier = 0.5;
    } else {
      withinTier = Math.max(0, Math.min(1, (logVal - logStart) / (logEnd - logStart)));
    }
  }

  // Each tier gets 1/N of the visual range
  const numTiers = TIME_TIERS.length;
  const tierWidth = 1 / numTiers;

  return tierIdx * tierWidth + withinTier * tierWidth;
}

// Transfer function: blend (mix between linear and log)
function blendTransfer(value: number, min: number, max: number, mix: number): number {
  const lin = linearTransfer(value, min, max);
  const log = logTransfer(value, min, max);
  return (1 - mix) * lin + mix * log;
}

// Time gradient coloring with configurable transfer function
// Values outside min-max range are shown as white (out of range)
// No-horizon values (< 0, i.e. sentinel -1) are shown as gray
export function timeGradientColors(
  values: number[],
  min: number,
  max: number,
  scaleType: TimeScaleType = 'linear',
  blendMix: number = 0.5
): Float32Array {
  const colors = new Float32Array(values.length * 3);


  for (let i = 0; i < values.length; i++) {
    const value = values[i];

    if (value < 0) {
      // Distinct blue-grey for no-horizon samples (sentinel value -1 from backend)
      // Using a cool grey that's clearly different from the warm plasma colormap
      colors[i * 3] = 0.35;      // R
      colors[i * 3 + 1] = 0.4;   // G
      colors[i * 3 + 2] = 0.5;   // B - slightly blue-tinted grey
    } else if (value < min || value > max) {
      // White/light gray for out-of-range samples (but NOT no-horizon)
      colors[i * 3] = 0.85;
      colors[i * 3 + 1] = 0.85;
      colors[i * 3 + 2] = 0.85;
    } else {
      // Apply transfer function based on scale type
      let t: number;
      switch (scaleType) {
        case 'log':
          t = logTransfer(value, min, max);
          break;
        case 'adaptive':
          t = adaptiveTransfer(value, min, max);
          break;
        case 'blend':
          t = blendTransfer(value, min, max, blendMix);
          break;
        case 'linear':
        default:
          t = linearTransfer(value, min, max);
      }

      t = Math.max(0, Math.min(1, t));
      const [r, g, b] = plasmaColor(t);
      colors[i * 3] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    }
  }

  return colors;
}

// Get tier labels for legend (adaptive scale)
export function getAdaptiveTierLabels(): { position: number; label: string }[] {
  const numTiers = TIME_TIERS.length;
  return TIME_TIERS.map((tier, i) => ({
    position: (i + 0.5) / numTiers,
    label: tier.name,
  }));
}

// Helper function for plasma color
function plasmaColor(t: number): [number, number, number] {
  const stops = [
    [0.050383, 0.029803, 0.527975],
    [0.254627, 0.013882, 0.615419],
    [0.417642, 0.000564, 0.658390],
    [0.562738, 0.051545, 0.641509],
    [0.692840, 0.165141, 0.564522],
    [0.798216, 0.280197, 0.469538],
    [0.881443, 0.392529, 0.383229],
    [0.949217, 0.517763, 0.295662],
    [0.988260, 0.652325, 0.211364],
    [0.988648, 0.809579, 0.145357],
    [0.940015, 0.975158, 0.131326],
  ];

  t = Math.max(0, Math.min(1, t));
  const idx = t * (stops.length - 1);
  const i = Math.floor(idx);
  const f = idx - i;

  if (i >= stops.length - 1) return stops[stops.length - 1] as [number, number, number];
  if (i < 0) return stops[0] as [number, number, number];

  const c0 = stops[i];
  const c1 = stops[i + 1];

  return [
    c0[0] + f * (c1[0] - c0[0]),
    c0[1] + f * (c1[1] - c0[1]),
    c0[2] + f * (c1[2] - c0[2]),
  ];
}

// Heatmap types
export interface HeatmapCell {
  layer: number;
  position: string;
  value: number | null;
}

export interface HeatmapData {
  metric: string;
  component: string;
  layers: number[];
  positions: string[];
  cells: HeatmapCell[];
  min_value: number | null;
  max_value: number | null;
}

// Hook to fetch heatmap data
// CRASH BEHAVIOR: No retries - if heatmap data is missing, app crashes immediately
export function useHeatmap(component: string, metric: string = 'r2') {
  return useQuery({
    queryKey: ['heatmap', component, metric],
    queryFn: async (): Promise<HeatmapData> => {
      return api.get<HeatmapData>(`/heatmap/${component}?metric=${metric}`);
    },
    enabled: !!component,
    staleTime: 1000 * 60 * 5, // 5 minutes
    retry: false, // CRASH: No retries - fail immediately if data is missing
  });
}

// Warmup status response type
interface WarmupStatusResponse {
  message: string;
  status: {
    is_running: boolean;
    progress: number;
    total: number;
    current_task: string | null;
    cached_pca: number;
    cached_umap: number;
    cached_tsne: number;
  };
}

// Hook to trigger backend prefetch for adjacent embeddings
export function useBackendPrefetch(
  layer: number,
  component: string,
  position: string,
  method: string,
  enabled: boolean = true
) {
  useEffect(() => {
    if (!enabled || layer < 0 || !component || !position || !method) return;

    // Fire immediately - no debounce for seamless navigation
    const prefetchUrl = `/warmup/prefetch?layer=${layer}&component=${encodeURIComponent(component)}&position=${encodeURIComponent(position)}&method=${method}`;

    api.post<WarmupStatusResponse>(prefetchUrl).catch(() => {
      // Silently ignore prefetch failures - it's just an optimization
    });
  }, [layer, component, position, method, enabled]);
}

// Hook to get warmup status
export function useWarmupStatus() {
  return useQuery({
    queryKey: ['warmup', 'status'],
    queryFn: async (): Promise<WarmupStatusResponse> => {
      return api.get<WarmupStatusResponse>('/warmup/status');
    },
    refetchInterval: (query) => {
      // Poll every 2 seconds while warmup is running
      return query.state.data?.status.is_running ? 2000 : false;
    },
    staleTime: 1000, // 1 second
  });
}

// Hook to trigger warmup
export function useWarmup() {
  const queryClient = useQueryClient();

  const startWarmup = async (
    methods: string[] = ['pca', 'umap'],
    components: string[] = ['resid_pre'],
    allPositions: boolean = true
  ) => {
    const methodsParam = methods.join(',');
    const componentsParam = components.join(',');

    const response = await api.post<WarmupStatusResponse>(
      `/warmup?methods=${methodsParam}&components=${componentsParam}&all_positions=${allPositions}`
    );

    // Invalidate the status query to get fresh data
    queryClient.invalidateQueries({ queryKey: ['warmup', 'status'] });

    return response;
  };

  return { startWarmup };
}

// Prefetching hook for background loading
export function usePrefetch(
  currentLayer: number,
  component: string,
  position: string,
  method: string,
  colorByOptions: string[],
  layers: number[],
  positions: string[] = []
) {
  const queryClient = useQueryClient();

  useEffect(() => {
    // Prefetch all color options in background
    colorByOptions.forEach((colorByOption) => {
      queryClient.prefetchQuery({
        queryKey: ['metadata', colorByOption],
        queryFn: async (): Promise<MetadataResponse> => {
          const response = await api.get<BackendMetadataResponse>(
            `/metadata?color_by=${colorByOption}`
          );
          let numericValues: number[];
          let labels: string[] | undefined;
          let min = 0;
          let max = 1;

          if (response.dtype === 'numeric') {
            // For time_horizon fields, backend sends null for no-horizon samples
            // Convert null to -1 (sentinel value) so timeGradientColors can identify them
            const isTimeField = colorByOption === 'time_horizon' ;
            numericValues = (response.values as (number | null)[]).map(v =>
              v === null ? -1 : v
            );

            // For min/max calculation, exclude sentinel values (-1)
            const validValues = isTimeField
              ? numericValues.filter(v => v >= 0)
              : numericValues;

            if (validValues.length > 0) {
              min = Math.min(...validValues);
              max = Math.max(...validValues);
            }
            // If empty, keep defaults min=0, max=1
          } else if (response.dtype === 'boolean') {
            numericValues = (response.values as boolean[]).map(v => v ? 1 : 0);
            labels = ['false', 'true'];
          } else {
            const uniqueVals = [...new Set(response.values as string[])];
            labels = uniqueVals;
            numericValues = (response.values as string[]).map(v =>
              uniqueVals.indexOf(v)
            );
            // Ensure max >= 1 to avoid division by zero in color calculations
            max = Math.max(1, uniqueVals.length - 1);
          }
          return { values: numericValues, labels, min, max };
        },
        staleTime: 1000 * 60 * 10, // 10 minutes
      });
    });

    // Prefetch adjacent layers in background
    const currentLayerIdx = layers.indexOf(currentLayer);
    const adjacentLayers = currentLayerIdx >= 0
      ? [layers[currentLayerIdx - 1], layers[currentLayerIdx + 1]].filter(l => l !== undefined)
      : [];

    // Prefetch adjacent positions in background
    const currentPosIdx = positions.indexOf(position);
    const adjacentPositions = currentPosIdx >= 0
      ? [positions[currentPosIdx - 1], positions[currentPosIdx + 1]].filter(p => p !== undefined)
      : [];

    // Prefetch embeddings for adjacent layers (same position)
    adjacentLayers.forEach((layer) => {
      queryClient.prefetchQuery({
        queryKey: ['embedding', layer, component, position, method],
        queryFn: async (): Promise<EmbeddingResponse> => {
          const response = await api.get<BackendEmbeddingResponse>(
            `/embedding/${layer}/${component}/${encodeURIComponent(position)}?method=${method.toLowerCase()}`
          );
          return {
            positions: response.coordinates_flat,
            indices: response.sample_indices,
            metrics: {},
          };
        },
        staleTime: 1000 * 60 * 5, // 5 minutes
      });
    });

    // Prefetch embeddings for adjacent positions (same layer)
    adjacentPositions.forEach((pos) => {
      queryClient.prefetchQuery({
        queryKey: ['embedding', currentLayer, component, pos, method],
        queryFn: async (): Promise<EmbeddingResponse> => {
          const response = await api.get<BackendEmbeddingResponse>(
            `/embedding/${currentLayer}/${component}/${encodeURIComponent(pos)}?method=${method.toLowerCase()}`
          );
          return {
            positions: response.coordinates_flat,
            indices: response.sample_indices,
            metrics: {},
          };
        },
        staleTime: 1000 * 60 * 5, // 5 minutes
      });
    });
  }, [queryClient, currentLayer, component, position, method, colorByOptions, layers, positions]);
}

// Types for trajectory data
interface TrajectoryPoint {
  x_value: string;
  values: number[];
  sample_indices: number[];
}

interface TrajectoryResponse {
  component: string;
  position?: string;
  layer?: number;
  method: string;
  x_axis: string;
  x_values: string[];
  n_samples: number;
  sample_indices: number[];  // Shared indices for layer trajectory
  data: TrajectoryPoint[];
}

export interface TrajectoryData {
  xValues: string[];
  trajectoryData: Map<string, Float32Array>;
  sampleIndices: number[];  // For layer trajectory (all x values share same indices)
  sampleIndicesMap: Map<string, number[]>;  // For position trajectory (per x value)
  nSamples: number;
  isLoading: boolean;
  error: Error | null;
}

// Hook to fetch PC1 trajectory across layers
// CRASH BEHAVIOR: No retries - if trajectory data is missing, app crashes immediately
export function useLayerTrajectory(
  component: string,
  position: string,
  enabled: boolean = true
): TrajectoryData {
  const { data, isLoading, error } = useQuery({
    queryKey: ['trajectory', 'layers', component, position],
    queryFn: async (): Promise<TrajectoryResponse> => {
      log('useLayerTrajectory', 'Fetching trajectory', { component, position });
      const startTime = performance.now();
      const response = await api.get<TrajectoryResponse>(
        `/trajectory/layers/${encodeURIComponent(component)}/${encodeURIComponent(position)}`
      );
      const elapsed = performance.now() - startTime;
      log('useLayerTrajectory', 'Trajectory loaded', { n_layers: response.x_values.length, n_samples: response.n_samples, elapsed_ms: elapsed.toFixed(1) });
      return response;
    },
    enabled: enabled && !!component && !!position,
    staleTime: 1000 * 60 * 5, // 5 minutes
    retry: false, // CRASH: No retries - fail immediately if data is missing
  });

  // Transform to Map<string, Float32Array>
  const trajectoryData = useMemo(() => {
    const map = new Map<string, Float32Array>();
    if (data?.data) {
      data.data.forEach((point) => {
        map.set(point.x_value, new Float32Array(point.values));
      });
    }
    return map;
  }, [data]);

  // For layer trajectory, sample indices are shared (from response level)
  const sampleIndices = data?.sample_indices || [];

  // Build per-point indices map (all same for layer trajectory)
  const sampleIndicesMap = useMemo(() => {
    const map = new Map<string, number[]>();
    if (data?.data) {
      data.data.forEach((point) => {
        map.set(point.x_value, point.sample_indices);
      });
    }
    return map;
  }, [data]);

  return {
    xValues: data?.x_values || [],
    trajectoryData,
    sampleIndices,
    sampleIndicesMap,
    nSamples: data?.n_samples || 0,
    isLoading,
    error: error as Error | null,
  };
}

// Hook to fetch PC1 trajectory across positions
// CRASH BEHAVIOR: No retries - if trajectory data is missing, app crashes immediately
export function usePositionTrajectory(
  layer: number,
  component: string,
  positions?: string[], // Optional filter
  enabled: boolean = true
): TrajectoryData {
  const positionsFilter = positions?.join(',') || '';

  const { data, isLoading, error } = useQuery({
    queryKey: ['trajectory', 'positions', layer, component, positionsFilter],
    queryFn: async (): Promise<TrajectoryResponse> => {
      log('usePositionTrajectory', 'Fetching trajectory', { layer, component });
      const startTime = performance.now();
      const url = `/trajectory/positions/${layer}/${encodeURIComponent(component)}${
        positionsFilter ? `?positions_filter=${encodeURIComponent(positionsFilter)}` : ''
      }`;
      const response = await api.get<TrajectoryResponse>(url);
      const elapsed = performance.now() - startTime;
      log('usePositionTrajectory', 'Trajectory loaded', { n_positions: response.x_values.length, n_samples: response.n_samples, elapsed_ms: elapsed.toFixed(1) });
      return response;
    },
    enabled: enabled && layer >= 0 && !!component,
    staleTime: 1000 * 60 * 5, // 5 minutes
    retry: false, // CRASH: No retries - fail immediately if data is missing
  });

  // Transform to Map<string, Float32Array>
  const trajectoryData = useMemo(() => {
    const map = new Map<string, Float32Array>();
    if (data?.data) {
      data.data.forEach((point) => {
        map.set(point.x_value, new Float32Array(point.values));
      });
    }
    return map;
  }, [data]);

  // For position trajectory, each position may have different sample indices
  const sampleIndicesMap = useMemo(() => {
    const map = new Map<string, number[]>();
    if (data?.data) {
      data.data.forEach((point) => {
        map.set(point.x_value, point.sample_indices);
      });
    }
    return map;
  }, [data]);

  return {
    xValues: data?.x_values || [],
    trajectoryData,
    sampleIndices: [],  // Position trajectory uses per-point indices
    sampleIndicesMap,
    nSamples: data?.n_samples || 0,
    isLoading,
    error: error as Error | null,
  };
}

// ==================== 2D TRAJECTORY HOOKS (PC1 + PC2) ====================

interface Trajectory2DPoint {
  x_value: string;
  pc1_values: number[];
  pc2_values: number[];
  sample_indices: number[];
}

interface Trajectory2DResponse {
  component: string;
  position?: string;
  layer?: number;
  method: string;
  x_axis: string;
  x_values: string[];
  n_samples: number;
  sample_indices: number[];
  data: Trajectory2DPoint[];
}

export interface Trajectory2DData {
  xValues: string[];
  pc1Data: Map<string, Float32Array>;  // x_value -> PC1 values
  pc2Data: Map<string, Float32Array>;  // x_value -> PC2 values
  sampleIndices: number[];
  sampleIndicesMap: Map<string, number[]>;
  nSamples: number;
  isLoading: boolean;
  error: Error | null;
}

// Hook to fetch PC1+PC2 trajectory across layers (for 3D line plot)
export function useLayerTrajectory2D(
  component: string,
  position: string,
  enabled: boolean = true
): Trajectory2DData {
  const { data, isLoading, error } = useQuery({
    queryKey: ['trajectory2d', 'layers', component, position],
    queryFn: async (): Promise<Trajectory2DResponse> => {
      log('useLayerTrajectory2D', 'Fetching 2D trajectory', { component, position });
      const startTime = performance.now();
      const response = await api.get<Trajectory2DResponse>(
        `/trajectory2d/layers/${encodeURIComponent(component)}/${encodeURIComponent(position)}`
      );
      const elapsed = performance.now() - startTime;
      log('useLayerTrajectory2D', '2D Trajectory loaded', { n_layers: response.x_values.length, n_samples: response.n_samples, elapsed_ms: elapsed.toFixed(1) });
      return response;
    },
    enabled: enabled && !!component && !!position,
    staleTime: 1000 * 60 * 5,
    retry: false,
  });

  const pc1Data = useMemo(() => {
    const map = new Map<string, Float32Array>();
    if (data?.data) {
      data.data.forEach((point) => {
        map.set(point.x_value, new Float32Array(point.pc1_values));
      });
    }
    return map;
  }, [data]);

  const pc2Data = useMemo(() => {
    const map = new Map<string, Float32Array>();
    if (data?.data) {
      data.data.forEach((point) => {
        map.set(point.x_value, new Float32Array(point.pc2_values));
      });
    }
    return map;
  }, [data]);

  const sampleIndicesMap = useMemo(() => {
    const map = new Map<string, number[]>();
    if (data?.data) {
      data.data.forEach((point) => {
        map.set(point.x_value, point.sample_indices);
      });
    }
    return map;
  }, [data]);

  return {
    xValues: data?.x_values || [],
    pc1Data,
    pc2Data,
    sampleIndices: data?.sample_indices || [],
    sampleIndicesMap,
    nSamples: data?.n_samples || 0,
    isLoading,
    error: error as Error | null,
  };
}

// Hook to fetch PC1+PC2 trajectory across positions (for 3D line plot)
export function usePositionTrajectory2D(
  layer: number,
  component: string,
  positions?: string[],
  enabled: boolean = true
): Trajectory2DData {
  const positionsFilter = positions?.join(',') || '';

  const { data, isLoading, error } = useQuery({
    queryKey: ['trajectory2d', 'positions', layer, component, positionsFilter],
    queryFn: async (): Promise<Trajectory2DResponse> => {
      log('usePositionTrajectory2D', 'Fetching 2D trajectory', { layer, component });
      const startTime = performance.now();
      const url = `/trajectory2d/positions/${layer}/${encodeURIComponent(component)}${
        positionsFilter ? `?positions_filter=${encodeURIComponent(positionsFilter)}` : ''
      }`;
      const response = await api.get<Trajectory2DResponse>(url);
      const elapsed = performance.now() - startTime;
      log('usePositionTrajectory2D', '2D Trajectory loaded', { n_positions: response.x_values.length, n_samples: response.n_samples, elapsed_ms: elapsed.toFixed(1) });
      return response;
    },
    enabled: enabled && layer >= 0 && !!component,
    staleTime: 1000 * 60 * 5,
    retry: false,
  });

  const pc1Data = useMemo(() => {
    const map = new Map<string, Float32Array>();
    if (data?.data) {
      data.data.forEach((point) => {
        map.set(point.x_value, new Float32Array(point.pc1_values));
      });
    }
    return map;
  }, [data]);

  const pc2Data = useMemo(() => {
    const map = new Map<string, Float32Array>();
    if (data?.data) {
      data.data.forEach((point) => {
        map.set(point.x_value, new Float32Array(point.pc2_values));
      });
    }
    return map;
  }, [data]);

  const sampleIndicesMap = useMemo(() => {
    const map = new Map<string, number[]>();
    if (data?.data) {
      data.data.forEach((point) => {
        map.set(point.x_value, point.sample_indices);
      });
    }
    return map;
  }, [data]);

  return {
    xValues: data?.x_values || [],
    pc1Data,
    pc2Data,
    sampleIndices: [],
    sampleIndicesMap,
    nSamples: data?.n_samples || 0,
    isLoading,
    error: error as Error | null,
  };
}

// Types for Scree plot data
interface ScreeSeries {
  label: string;
  layer: number;
  component: string;
  values: number[];
}

export interface ScreeData {
  series: ScreeSeries[];
}

// Hook to fetch Scree plot data (cumulative variance explained)
export function useScreeData(
  position: string,
  nComponents: number = 10,
  enabled: boolean = true
) {
  return useQuery({
    queryKey: ['scree', position, nComponents],
    queryFn: async (): Promise<ScreeData> => {
      log('useScreeData', 'Fetching scree data', { position, nComponents });
      const startTime = performance.now();
      const response = await api.get<ScreeData>(
        `/scree/${encodeURIComponent(position)}?n_components=${nComponents}`
      );
      const elapsed = performance.now() - startTime;
      log('useScreeData', 'Scree data loaded', { n_series: response.series?.length || 0, elapsed_ms: elapsed.toFixed(1) });
      return response;
    },
    enabled: enabled && !!position,
    staleTime: 1000 * 60 * 10, // 10 minutes
    retry: false,
  });
}

// Types for Direction Alignment data
export interface AlignmentData {
  labels: string[];
  matrix: number[][];
  n_targets: number;
  error?: string;
}

// Hook to fetch Direction Alignment data (cosine similarity matrix)
export function useAlignmentData(
  position: string,
  pcIndex: number = 0,
  enabled: boolean = true
) {
  return useQuery({
    queryKey: ['alignment', position, pcIndex],
    queryFn: async (): Promise<AlignmentData> => {
      log('useAlignmentData', 'Fetching alignment data', { position, pcIndex });
      const startTime = performance.now();
      const response = await api.get<AlignmentData>(
        `/alignment/${encodeURIComponent(position)}?pc_index=${pcIndex}`
      );
      const elapsed = performance.now() - startTime;
      log('useAlignmentData', 'Alignment data loaded', { n_targets: response.n_targets, elapsed_ms: elapsed.toFixed(1) });
      return response;
    },
    enabled: enabled && !!position,
    staleTime: 1000 * 60 * 10, // 10 minutes
    retry: false,
  });
}
