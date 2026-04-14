import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { Header, ControlPanel, InfoPanel, ScatterPlot3D, ScatterPlot2D, Legend, LegendItem, PositionSelector, TrajectoryPlot, TrajectoryPlot3D, FilterPanel, ScreePlot, AlignmentHeatmap } from './components';
import type { ScatterPlot2DExportHandle } from './components/ScatterPlot2D';
import { Toggle } from './components/ui/Toggle';
import { Select } from './components/ui/Select';
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/Card';
import { PointData } from './components/PointCloud';
import {
  useConfig,
  useStreamingEmbedding,
  useMetadata,
  useSample,
  usePrefetch,
  useBackendPrefetch,
  useLayerTrajectory,
  usePositionTrajectory,
  useLayerTrajectory2D,
  usePositionTrajectory2D,
  useScreeData,
  useAlignmentData,
  toFloat32Array,
  valuesToColors,
  categoricalColors,
  timeScaleCategoricalColors,
  timeGradientColors,
  getAdaptiveTierLabels,
  TimeScaleType,
  PCAMode,
} from './hooks/useEmbeddings';

// Logging helper
const log = (category: string, message: string, data?: Record<string, unknown>) => {
  const ts = new Date().toISOString().slice(11, 23);
  const dataStr = data ? ` | ${Object.entries(data).map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' ')}` : '';
  console.log(`[${ts}] [CLIENT] [${category}] ${message}${dataStr}`);
};

// Default values - these are fallbacks, actual values come from config
const DEFAULT_COMPONENT = 'resid_post';
const DEFAULT_METHOD = 'pca';
const DEFAULT_COLOR_BY = 'time_horizon';

type ViewMode = '2D' | '3D' | '1DxLayer' | '1DxPos' | '2DxLayer' | '2DxPos' | 'Scree' | 'Align';

// Helper to check if a view mode is a trajectory view
const isTrajectoryViewMode = (mode: ViewMode) =>
  mode === '1DxLayer' || mode === '1DxPos' || mode === '2DxLayer' || mode === '2DxPos';

function App() {
  // Get dataset name from URL path (e.g., /geometry -> "geometry")
  const dataset = window.location.pathname.split('/')[1] || 'geometry';

  // Render counter for debugging
  const renderCount = useRef(0);
  renderCount.current++;
  log('App', `Render #${renderCount.current}`);

  // Export ref for ScatterPlot2D
  const scatterPlot2DRef = useRef<ScatterPlot2DExportHandle>(null);
  // Ref for the plot container (for universal export)
  const plotContainerRef = useRef<HTMLDivElement>(null);

  // UI state
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('3D');
  const [pcaMode, setPcaMode] = useState<PCAMode>('aligned');
  const [legendCollapsed, setLegendCollapsed] = useState(false);

  // Control state - layer and position are initialized to -1/'', then set from config
  const [layer, setLayer] = useState<number>(-1);
  const [component, setComponent] = useState(DEFAULT_COMPONENT);
  const [position, setPosition] = useState<string>('');
  const [method, setMethod] = useState(DEFAULT_METHOD);
  const [colorBy, setColorBy] = useState(DEFAULT_COLOR_BY);
  const [showNoHorizon, setShowNoHorizon] = useState(true);
  const [showWithHorizon, setShowWithHorizon] = useState(true);

  // Color range controls for gradient coloring
  const [colorRangeMin, setColorRangeMin] = useState<number | null>(null);
  const [colorRangeMax, setColorRangeMax] = useState<number | null>(null);

  // Time scale transfer function controls
  const [timeScaleType, setTimeScaleType] = useState<TimeScaleType>('adaptive');
  const [blendMix, setBlendMix] = useState(0.5);

  // Filter state - separate short-term and long-term
  const [shortRewardFilter, setShortRewardFilter] = useState<{ min: number | null; max: number | null }>({ min: null, max: null });
  const [shortTimeFilter, setShortTimeFilter] = useState<{ min: number | null; max: number | null }>({ min: null, max: null });
  const [longRewardFilter, setLongRewardFilter] = useState<{ min: number | null; max: number | null }>({ min: null, max: null });
  const [longTimeFilter, setLongTimeFilter] = useState<{ min: number | null; max: number | null }>({ min: null, max: null });

  // Selection state
  const [selectedSampleIdx, setSelectedSampleIdx] = useState<number | null>(null);

  // Clear selection when embedding parameters change to avoid stale references
  // But only for parameters that affect the current view mode
  useEffect(() => {
    // In 1DxLayer view, layer changes don't affect the data (shows all layers)
    // In 1DxPos view, position changes don't affect the data (shows all positions)
    // So we only clear selection for parameters that matter
    if (isTrajectoryViewMode(viewMode)) {
      // For trajectory views, only component/method changes matter
      // (and position for layer trajectories, layer for position trajectories)
      return;
    }
    setSelectedSampleIdx(null);
  }, [layer, component, position, method, viewMode]);

  // Fetch config
  const { data: config, isLoading: configLoading } = useConfig();

  // Log loading states
  useEffect(() => {
    if (configLoading) {
      log('App', '⏳ WAITING FOR: Config from server');
    }
  }, [configLoading]);

  // Initialize layer and position from config when it first loads
  useEffect(() => {
    if (config) {
      // Set layer to LAST available if not yet set or invalid (highest layer = most processed)
      if (layer < 0 || !config.layers.includes(layer)) {
        const lastLayer = config.layers[config.layers.length - 1];
        if (lastLayer !== undefined) {
          log('App', 'Initializing layer from config', { layer: lastLayer });
          setLayer(lastLayer);
        }
      }
      // Set position to LAST available if not yet set or invalid
      // Check base position (without rel_pos suffix) against config.positions
      const basePosition = position.includes(':') ? position.split(':')[0] : position;
      if (!position || !config.positions.includes(basePosition)) {
        const lastPosition = config.positions[config.positions.length - 1];
        if (lastPosition) {
          log('App', 'Initializing position from config', { position: lastPosition });
          setPosition(lastPosition);
        }
      }
    }
  }, [config, layer, position]);

  // Log state changes
  useEffect(() => {
    log('App', 'State changed', { layer, component, position, method, colorBy, viewMode });
  }, [layer, component, position, method, colorBy, viewMode]);

  // Only fetch embedding data for scatter plots (2D/3D), not trajectory views
  const isScatterView = viewMode === '2D' || viewMode === '3D';
  const streamingEmbed = useStreamingEmbedding(layer, component, position, method, isScatterView);

  // Log when waiting for embedding
  useEffect(() => {
    if (isScatterView && streamingEmbed.isStreaming && streamingEmbed.loadedPoints === 0) {
      log('App', `⏳ WAITING FOR: Embedding stream L${layer}/${component}/${position} (${method})`);
    } else if (isScatterView && streamingEmbed.isStreaming) {
      log('App', `📥 STREAMING: ${streamingEmbed.progress}% loaded (${streamingEmbed.loadedPoints} points)`);
    }
  }, [isScatterView, streamingEmbed.isStreaming, streamingEmbed.loadedPoints, streamingEmbed.progress, layer, component, position, method]);

  // Convert streaming state to the format expected by the rest of the app
  const embedding = useMemo(() => {
    if (streamingEmbed.loadedPoints === 0) return undefined;
    return {
      positions: streamingEmbed.positions,
      indices: streamingEmbed.indices,
      metrics: {},
    };
  }, [streamingEmbed.positions, streamingEmbed.indices, streamingEmbed.loadedPoints]);

  const embeddingLoading = streamingEmbed.isStreaming && streamingEmbed.loadedPoints === 0;
  const embeddingError = streamingEmbed.error;
  const streamingProgress = streamingEmbed.progress;

  // Fetch metadata for coloring
  const { data: metadata, isLoading: metadataLoading, error: metadataError } = useMetadata(colorBy);

  // Log when waiting for metadata
  useEffect(() => {
    if (metadataLoading) {
      log('App', `⏳ WAITING FOR: Metadata (color_by=${colorBy})`);
    }
  }, [metadataLoading, colorBy]);

  // Fetch has_horizon metadata for filtering
  const { data: hasHorizonMeta, error: hasHorizonError } = useMetadata('has_horizon');

  // Fetch short/long term metadata for filtering
  const { data: shortRewardMeta } = useMetadata('short_term_reward');
  const { data: shortTimeMeta } = useMetadata('short_term_time');
  const { data: longRewardMeta } = useMetadata('long_term_reward');
  const { data: longTimeMeta } = useMetadata('long_term_time');

  // Fetch selected sample details
  const { data: selectedSample, isLoading: sampleLoading } = useSample(
    selectedSampleIdx
  );

  // Prefetch adjacent layers, positions, and all color options in background
  // Only prefetch for scatter views (trajectory views load all data anyway)
  usePrefetch(
    layer,
    component,
    position,
    method,
    config?.colorByOptions || [],
    config?.layers || [],
    config?.positions || []
  );

  // Trigger backend prefetch for adjacent embeddings (fire-and-forget)
  // This precomputes neighboring layers/positions server-side for seamless navigation
  // Only for scatter views since trajectory views don't use single-layer embeddings
  useBackendPrefetch(layer, component, position, method, isScatterView && !embeddingLoading);

  // Fetch trajectory data for 1DxLayer view (PC1 across all layers)
  const layerTrajectory = useLayerTrajectory(
    component,
    position,
    viewMode === '1DxLayer',
    pcaMode
  );

  // Fetch trajectory data for 1DxPos view (PC1 across positions)
  const positionTrajectory = usePositionTrajectory(
    layer,
    component,
    undefined, // Use default named positions
    viewMode === '1DxPos',
    pcaMode
  );

  // Fetch 2D trajectory data for 2DxLayer view (PC1+PC2 across all layers)
  const layerTrajectory2D = useLayerTrajectory2D(
    component,
    position,
    viewMode === '2DxLayer',
    pcaMode
  );

  // Fetch 2D trajectory data for 2DxPos view (PC1+PC2 across positions)
  const positionTrajectory2D = usePositionTrajectory2D(
    layer,
    component,
    undefined,
    viewMode === '2DxPos',
    pcaMode
  );

  // For Scree and Align views, use base position (strip rel_pos suffix)
  // These views don't have per-rel_pos data
  const basePosition = position.includes(':') ? position.split(':')[0] : position;

  // Fetch Scree data for Scree view
  const screeData = useScreeData(basePosition, 10, viewMode === 'Scree');

  // Fetch Alignment data for Align view
  const alignmentData = useAlignmentData(basePosition, 0, viewMode === 'Align');

  // Log when waiting for trajectory data
  useEffect(() => {
    if (viewMode === '1DxLayer' && layerTrajectory.isLoading) {
      log('App', `⏳ WAITING FOR: Layer trajectory (${position} @ ${component})`);
    }
  }, [viewMode, layerTrajectory.isLoading, position, component]);

  useEffect(() => {
    if (viewMode === '1DxPos' && positionTrajectory.isLoading) {
      log('App', `⏳ WAITING FOR: Position trajectory (L${layer} @ ${component})`);
    }
  }, [viewMode, positionTrajectory.isLoading, layer, component]);

  // Check if any filters are active
  const hasShortRewardFilter = shortRewardFilter.min !== null || shortRewardFilter.max !== null;
  const hasShortTimeFilter = shortTimeFilter.min !== null || shortTimeFilter.max !== null;
  const hasLongRewardFilter = longRewardFilter.min !== null || longRewardFilter.max !== null;
  const hasLongTimeFilter = longTimeFilter.min !== null || longTimeFilter.max !== null;
  const hasHorizonFilter = !showNoHorizon || !showWithHorizon;
  const hasAnyFilter = hasShortRewardFilter || hasShortTimeFilter || hasLongRewardFilter || hasLongTimeFilter || hasHorizonFilter;

  // Compute filter mask for 2D/3D views (aligned with embedding.indices)
  const scatterFilterMask = useMemo(() => {
    // No filtering if no filters active and data not loaded
    if (!embedding?.indices) return null;
    if (!hasAnyFilter) return null;

    // If both horizon toggles are off, filter everything
    if (!showNoHorizon && !showWithHorizon) {
      return embedding.indices.map(() => false);
    }

    // Apply all filters
    return embedding.indices.map(idx => {
      // Horizon filter
      if (hasHorizonMeta?.values) {
        const hasHorizon = Boolean(hasHorizonMeta.values[idx]);
        if (hasHorizon && !showWithHorizon) return false;
        if (!hasHorizon && !showNoHorizon) return false;
      }

      // Short-term reward filter
      if (hasShortRewardFilter && shortRewardMeta?.values) {
        const reward = shortRewardMeta.values[idx];
        if (shortRewardFilter.min !== null && reward < shortRewardFilter.min) return false;
        if (shortRewardFilter.max !== null && reward > shortRewardFilter.max) return false;
      }

      // Short-term time filter
      if (hasShortTimeFilter && shortTimeMeta?.values) {
        const time = shortTimeMeta.values[idx];
        if (shortTimeFilter.min !== null && time < shortTimeFilter.min) return false;
        if (shortTimeFilter.max !== null && time > shortTimeFilter.max) return false;
      }

      // Long-term reward filter
      if (hasLongRewardFilter && longRewardMeta?.values) {
        const reward = longRewardMeta.values[idx];
        if (longRewardFilter.min !== null && reward < longRewardFilter.min) return false;
        if (longRewardFilter.max !== null && reward > longRewardFilter.max) return false;
      }

      // Long-term time filter
      if (hasLongTimeFilter && longTimeMeta?.values) {
        const time = longTimeMeta.values[idx];
        if (longTimeFilter.min !== null && time < longTimeFilter.min) return false;
        if (longTimeFilter.max !== null && time > longTimeFilter.max) return false;
      }

      return true;
    });
  }, [
    hasHorizonMeta?.values, shortRewardMeta?.values, shortTimeMeta?.values, longRewardMeta?.values, longTimeMeta?.values,
    embedding?.indices, showNoHorizon, showWithHorizon,
    hasAnyFilter, hasShortRewardFilter, hasShortTimeFilter, hasLongRewardFilter, hasLongTimeFilter,
    shortRewardFilter, shortTimeFilter, longRewardFilter, longTimeFilter
  ]);

  // Get actual sample indices for trajectory views (needed for filter mask and colors)
  const trajectorySampleIndices = useMemo(() => {
    if (viewMode === '1DxLayer' || viewMode === '2DxLayer') {
      // Layer trajectory - all layers share same indices
      // Use 2D data if available, fallback to 1D
      return layerTrajectory2D.sampleIndices.length > 0
        ? layerTrajectory2D.sampleIndices
        : layerTrajectory.sampleIndices;
    } else if (viewMode === '1DxPos' || viewMode === '2DxPos') {
      // Position trajectory - UNION of all position indices
      // Each position may have different samples, so we need the union
      const sourceMap = positionTrajectory2D.sampleIndicesMap.size > 0
        ? positionTrajectory2D.sampleIndicesMap
        : positionTrajectory.sampleIndicesMap;
      const allIndices = new Set<number>();
      sourceMap.forEach((indices) => {
        indices.forEach(idx => allIndices.add(idx));
      });
      return Array.from(allIndices).sort((a, b) => a - b);
    }
    return [];
  }, [viewMode, layerTrajectory.sampleIndices, layerTrajectory2D.sampleIndices, positionTrajectory.sampleIndicesMap, positionTrajectory2D.sampleIndicesMap]);

  // Compute filter mask for trajectory views (aligned with trajectorySampleIndices)
  const trajectoryFilterMask = useMemo(() => {
    const isTrajectoryView = isTrajectoryViewMode(viewMode);
    if (!isTrajectoryView) return null;

    const sampleIndices = trajectorySampleIndices;
    if (!hasHorizonMeta?.values || sampleIndices.length === 0 || (showNoHorizon && showWithHorizon)) {
      return null;
    }
    if (!showNoHorizon && !showWithHorizon) {
      return Array(sampleIndices.length).fill(false);
    }
    // Use actual sample indices to look up metadata
    return sampleIndices.map(sampleIdx => {
      const hasHorizon = Boolean(hasHorizonMeta.values[sampleIdx]);
      return hasHorizon ? showWithHorizon : showNoHorizon;
    });
  }, [hasHorizonMeta?.values, viewMode, trajectorySampleIndices, showNoHorizon, showWithHorizon]);

  // Use appropriate filter mask based on view mode
  const filterMask = (isTrajectoryViewMode(viewMode)) ? trajectoryFilterMask : scatterFilterMask;

  // Compute positions Float32Array - NO filtering, positions stay fixed
  const positions = useMemo(() => {
    const startTime = performance.now();
    if (!embedding?.positions) {
      return new Float32Array(0);
    }
    const result = toFloat32Array(embedding.positions);
    const elapsed = performance.now() - startTime;
    log('App', 'Computed positions', { n_points: result.length / 3, elapsed_ms: elapsed.toFixed(2) });
    return result;
  }, [embedding?.positions]);

  // Compute visibility Float32Array from scatterFilterMask
  const visibility = useMemo(() => {
    const numPoints = positions.length / 3;
    if (numPoints === 0) {
      return new Float32Array(0);
    }
    // Default: all visible
    if (!scatterFilterMask) {
      const vis = new Float32Array(numPoints);
      for (let i = 0; i < numPoints; i++) vis[i] = 1.0;
      return vis;
    }
    // Apply filter mask
    const vis = new Float32Array(numPoints);
    for (let i = 0; i < numPoints; i++) {
      vis[i] = scatterFilterMask[i] ? 1.0 : 0.0;
    }
    return vis;
  }, [positions.length, scatterFilterMask]);

  // Fields that should always use gradient coloring (not categorical)
  const GRADIENT_FIELDS = [
    // Time horizon fields (all numeric, use gradient)
    'time_horizon', 'time_horizon_days', 'time_horizon_years', 'log_time_horizon',
    // Choice time fields
    'chosen_time', 'alt_time',
    // Option fields
    'short_term_reward', 'short_term_time', 'long_term_reward', 'long_term_time',
    'long_term_delay',
    // Reward fields
    'chosen_reward', 'alt_reward',
    // Delta/ratio fields
    'option_reward_delta', 'option_time_delta', 'option_confidence_delta',
    'reward_ratio', 'time_ratio',
    // Probability fields (0-1)
    'choice_confidence', 'choice_prob', 'alt_prob',
    // Index
    'sample_idx',
  ];

  // Compute effective color range (user-defined or data-derived)
  const effectiveColorRange = useMemo(() => {
    if (!metadata?.values) return { min: 0, max: 1, dataMin: 0, dataMax: 1 };

    // Probability fields always use 0-1 range
    const PROBABILITY_FIELDS = ['choice_confidence', 'choice_prob', 'alt_prob'];
    if (PROBABILITY_FIELDS.includes(colorBy)) {
      return { min: 0, max: 1, dataMin: 0, dataMax: 1 };
    }

    // For time-related fields, filter out no-horizon (sentinel -1) values for range calculation
    // TIME_FIELDS: fields that represent time durations in months (get special formatting)
    const TIME_FIELDS = ['time_horizon', 'chosen_time', 'alt_time', 'short_term_time', 'long_term_time'];
    const isTimeField = TIME_FIELDS.includes(colorBy);
    const valuesForRange = isTimeField
      ? metadata.values.filter(v => v >= 0)
      : metadata.values;

    const dataMin = valuesForRange.length > 0 ? Math.min(...valuesForRange) : 0;
    const dataMax = valuesForRange.length > 0 ? Math.max(...valuesForRange) : 1;

    // DEBUG: Log the color range calculation
    console.log(`[DEBUG] effectiveColorRange: colorBy=${colorBy}, valuesForRange.length=${valuesForRange.length}, dataMin=${dataMin}, dataMax=${dataMax} (${(dataMax/12).toFixed(2)} years), colorRangeMax=${colorRangeMax}`);

    return {
      min: colorRangeMin ?? dataMin,
      max: colorRangeMax ?? dataMax,
      dataMin,
      dataMax,
    };
  }, [metadata?.values, colorBy, colorRangeMin, colorRangeMax]);

  // Compute colors Float32Array - NO filtering, visibility handles hiding
  const colors = useMemo(() => {
    const startTime = performance.now();
    if (!metadata?.values || metadata.values.length === 0 || !embedding?.indices) {
      // Default gradient colors if no metadata
      const numPoints = positions.length / 3;
      const defaultColors = new Float32Array(numPoints * 3);
      for (let i = 0; i < numPoints; i++) {
        const t = i / (numPoints - 1 || 1);
        // Purple to pink gradient
        defaultColors[i * 3] = 0.78 + t * 0.22;
        defaultColors[i * 3 + 1] = 0.47 - t * 0.05;
        defaultColors[i * 3 + 2] = 0.87 - t * 0.25;
      }
      return defaultColors;
    }

    // Map metadata values through embedding indices to align with positions
    const embeddingValues = embedding.indices.map(idx => metadata.values[idx]);

    // Check if categorical or continuous (force gradient for certain fields)
    const uniqueValues = new Set(embeddingValues);
    const forceGradient = GRADIENT_FIELDS.includes(colorBy);
    const isCategorical = !forceGradient && uniqueValues.size <= 10;

    let result: Float32Array;
    if (isCategorical) {
      // Use special gradient colors for time_scale
      if (colorBy === 'time_scale') {
        result = timeScaleCategoricalColors(embeddingValues, uniqueValues.size);
      } else {
        result = categoricalColors(embeddingValues, uniqueValues.size);
      }
    } else {
      // For time-related fields, use special coloring with gray for no-horizon (value 0)
      const TIME_FIELDS = ['time_horizon', 'chosen_time', 'alt_time', 'short_term_time', 'long_term_time'];
      const isTimeField = TIME_FIELDS.includes(colorBy);
      if (isTimeField) {
        result = timeGradientColors(
          embeddingValues,
          effectiveColorRange.min,
          effectiveColorRange.max,
          timeScaleType,
          blendMix
        );
      } else {
        result = valuesToColors(embeddingValues, effectiveColorRange.min, effectiveColorRange.max, 'turbo');
      }
    }
    const elapsed = performance.now() - startTime;
    log('App', 'Computed colors', { n_points: result.length / 3, colorBy, isCategorical, elapsed_ms: elapsed.toFixed(2) });
    return result;
  }, [metadata, embedding?.indices, positions.length, colorBy, effectiveColorRange, timeScaleType, blendMix]);

  // Create point data array (with filtering)
  const pointData = useMemo<PointData[]>(() => {
    if (!embedding?.indices) return [];
    // NO filtering - visibility handles hiding
    return embedding.indices.map((idx) => ({
      sampleIdx: idx,
    }));
  }, [embedding?.indices]);

  // Helper to compute colors from values
  const computeColors = useCallback((values: number[], numPoints: number): Float32Array => {
    if (values.length === 0 || numPoints === 0) {
      const defaultColors = new Float32Array(numPoints * 3);
      for (let i = 0; i < numPoints; i++) {
        const t = i / (numPoints - 1 || 1);
        defaultColors[i * 3] = 0.78 + t * 0.22;
        defaultColors[i * 3 + 1] = 0.47 - t * 0.05;
        defaultColors[i * 3 + 2] = 0.87 - t * 0.25;
      }
      return defaultColors;
    }

    const uniqueValues = new Set(values);
    const forceGradient = GRADIENT_FIELDS.includes(colorBy);
    const isCategorical = !forceGradient && uniqueValues.size <= 10;

    if (isCategorical) {
      return categoricalColors(values, uniqueValues.size);
    } else {
      const TIME_FIELDS = ['time_horizon', 'chosen_time', 'alt_time', 'short_term_time', 'long_term_time'];
      const isTimeField = TIME_FIELDS.includes(colorBy);
      if (isTimeField) {
        return timeGradientColors(
          values,
          effectiveColorRange.min,
          effectiveColorRange.max,
          timeScaleType,
          blendMix
        );
      }
      return valuesToColors(values, effectiveColorRange.min, effectiveColorRange.max, 'turbo');
    }
  }, [colorBy, effectiveColorRange, timeScaleType, blendMix]);

  // Colors for scatter views (2D/3D) - only depends on embedding data
  const scatterColors = useMemo(() => {
    const numPoints = embedding?.indices?.length || 0;
    if (!metadata?.values || numPoints === 0) {
      return computeColors([], numPoints);
    }
    const values = embedding!.indices.map(idx => metadata.values[idx]);
    return computeColors(values, numPoints);
  }, [metadata, embedding?.indices, computeColors]);

  // Point data for scatter views
  const scatterPointData = useMemo<PointData[]>(() => {
    if (!embedding?.indices) return [];
    return embedding.indices.map((idx) => ({ sampleIdx: idx }));
  }, [embedding?.indices]);

  // Colors for trajectory views - only computed when needed
  const trajectoryColors = useMemo(() => {
    const isTrajectoryView = isTrajectoryViewMode(viewMode);
    if (!isTrajectoryView) return new Float32Array(0);

    const sampleIndices = trajectorySampleIndices;
    const numPoints = sampleIndices.length;
    if (!metadata?.values || numPoints === 0) {
      return computeColors([], numPoints);
    }
    // Get color values for actual sample indices
    const values = sampleIndices.map(idx => metadata.values[idx]);
    return computeColors(values, numPoints);
  }, [metadata, viewMode, trajectorySampleIndices, computeColors]);

  // Point data for trajectory views - use actual sample indices
  const trajectoryPointData = useMemo<PointData[]>(() => {
    const isTrajectoryView = isTrajectoryViewMode(viewMode);
    if (!isTrajectoryView) return [];

    return trajectorySampleIndices.map(sampleIdx => ({ sampleIdx }));
  }, [viewMode, trajectorySampleIndices]);

  // Select colors/pointData based on view mode
  const unfilteredColors = (isTrajectoryViewMode(viewMode)) ? trajectoryColors : scatterColors;
  const unfilteredPointData = (isTrajectoryViewMode(viewMode)) ? trajectoryPointData : scatterPointData;

  // Clear selection when the selected sample is filtered out
  useEffect(() => {
    if (selectedSampleIdx === null) return;
    // Check if the selected sample is still in the visible pointData
    const isStillVisible = pointData.some(p => p.sampleIdx === selectedSampleIdx);
    if (!isStillVisible && pointData.length > 0) {
      setSelectedSampleIdx(null);
    }
  }, [pointData, selectedSampleIdx]);

  // Human-readable labels for color options
  const COLOR_LABELS: Record<string, string> = {
    'time_horizon': 'Time Horizon',
    'chosen_time': 'Chosen Time',
    'chosen_reward': 'Chosen Reward',
    'matches_largest_reward': 'Chose Largest Reward',
    'matches_rational': 'Chose Rational',
    'matches_associated': 'Matches Associated',
    'has_horizon': 'Has Horizon',
    'short_term_first': 'Option Order',
    'context_id': 'Context',
    'sample_idx': 'Sample Index',
    'option_reward_delta': 'Option Reward Delta',
    'option_time_delta': 'Option Time Delta',
    'option_confidence_delta': 'Option Confidence Delta',
    'reward_ratio': 'Reward Ratio (Long/Short)',
    'time_ratio': 'Time Ratio (Long/Short)',
    'short_term_reward': 'Short-Term Reward',
    'short_term_time': 'Short-Term Time',
    'long_term_reward': 'Long-Term Reward',
    'long_term_time': 'Long-Term Time',
  };

  // Generate legend data
  // Helper to format time values for display
  const formatTimeValue = (value: number, isLog: boolean): string => {
    if (value < 0) return 'No horizon';  // Sentinel value -1 indicates no horizon

    // Convert from log if needed
    const months = isLog ? Math.pow(10, value) - 1 : value;

    if (months < 1) return `${Math.round(months * 30)} days`;
    if (months < 12) return `${months.toFixed(1)} mo`;
    if (months < 120) return `${(months / 12).toFixed(1)} yr`;
    if (months < 1200) return `${Math.round(months / 12)} yr`;
    if (months < 12000) return `${(months / 120).toFixed(0)} dec`;
    if (months < 120000) return `${(months / 1200).toFixed(0)} cent`;
    return `${(months / 12000).toFixed(0)} mill`;
  };

  const legendData = useMemo(() => {
    if (!metadata?.values || metadata.values.length === 0) {
      return null;
    }

    const uniqueValues = new Set(metadata.values);
    const forceGradient = GRADIENT_FIELDS.includes(colorBy);
    const isCategorical = !forceGradient && uniqueValues.size <= 10;

    if (isCategorical) {
      // Time scale colors by label - must match useEmbeddings.ts timeScaleCategoricalColors exactly!
      // Cool blue (Seconds) → warm red (Centuries), blue-grey for No Horizon
      const timeScaleColors: Record<string, string> = {
        'Seconds': '#3078ED',   // blue [0.19, 0.47, 0.93]
        'Minutes': '#199ED9',   // cyan-blue [0.10, 0.62, 0.85]
        'Hours': '#33BAB3',     // teal [0.20, 0.73, 0.70]
        'Days': '#66CC80',      // green [0.40, 0.80, 0.50]
        'Weeks': '#A6D959',     // yellow-green [0.65, 0.85, 0.35]
        'Months': '#E6CC4D',    // yellow [0.90, 0.80, 0.30]
        'Years': '#F29940',     // orange [0.95, 0.60, 0.25]
        'Decades': '#E66633',   // red-orange [0.90, 0.40, 0.20]
        'Centuries': '#CC4033', // red [0.80, 0.25, 0.20]
        'No Horizon': '#596680', // blue-grey [0.35, 0.40, 0.50]
      };

      // Default categorical palette - must match useEmbeddings.ts categoricalColors exactly!
      // Anthropic-inspired with high contrast
      // Values from useEmbeddings: [r, g, b] -> hex
      const defaultPalette = [
        '#D97757', // [0.85, 0.47, 0.34] Anthropic terracotta/coral
        '#348296', // [0.20, 0.51, 0.59] Deep teal (high contrast)
        '#8F70DB', // [0.56, 0.44, 0.86] Purple
        '#F2AD42', // [0.95, 0.68, 0.26] Warm amber
        '#59A678', // [0.35, 0.65, 0.47] Forest green
        '#F08080', // [0.94, 0.5, 0.5]   Light red
        '#9696CC', // [0.59, 0.59, 0.8]  Light purple
        '#FFA666', // [1.0, 0.65, 0.4]   Orange
      ];

      const items: LegendItem[] = [];
      // Filter out null/NaN/negative sentinel values and sort numerically
      const validValues = [...uniqueValues].filter(v =>
        v !== null && !Number.isNaN(v) && Number.isFinite(v) && v >= 0
      );
      const hasNullValues = uniqueValues.has(null as unknown as number) ||
        [...uniqueValues].some(v => !Number.isFinite(v) || v < 0);
      const sortedValues = validValues.sort((a, b) => Number(a) - Number(b));

      // Track labels we've already added to avoid duplicates
      const addedLabels = new Set<string>();

      sortedValues.forEach((value) => {
        let label: string;
        let color: string;
        const numValue = Number(value);
        const safeValue = Number.isFinite(numValue) ? Math.max(0, Math.floor(numValue)) : 0;

        // Special case for time_scale - use temporal gradient colors by label
        if (colorBy === 'time_scale') {
          // Get label from backend
          if (metadata.labels && metadata.labels[safeValue] !== undefined) {
            label = metadata.labels[safeValue];
          } else {
            label = String(numValue);
          }
          // Look up color by label (not index) so colors are consistent regardless of which categories are present
          color = timeScaleColors[label] || '#999999';
        // Special handling for known boolean fields (check FIRST, before backend labels)
        // These fields have specific semantic labels we want to enforce
        } else if (colorBy === 'has_horizon') {
          label = numValue > 0 ? 'Has horizon' : 'No horizon';
          const colorIdx = safeValue % defaultPalette.length;
          color = defaultPalette[colorIdx];
        } else if (colorBy === 'short_term_first') {
          label = numValue > 0 ? 'Short First' : 'Long First';
          const colorIdx = safeValue % defaultPalette.length;
          color = defaultPalette[colorIdx];
        } else if (colorBy === 'matches_largest_reward' || colorBy === 'matches_rational' || colorBy === 'matches_associated') {
          label = numValue > 0 ? 'Yes' : 'No';
          const colorIdx = safeValue % defaultPalette.length;
          color = defaultPalette[colorIdx];
        // Then check backend labels for other categorical fields (choice_type, etc.)
        } else if (metadata.labels && metadata.labels[safeValue] !== undefined) {
          label = metadata.labels[safeValue];
          const colorIdx = safeValue % defaultPalette.length;
          color = defaultPalette[colorIdx];
        } else {
          // Format numeric values nicely (avoid raw decimals)
          if (Number.isInteger(numValue)) {
            label = String(numValue);
          } else {
            // Round to reasonable precision for display
            label = numValue.toFixed(2).replace(/\.?0+$/, '');
          }
          const colorIdx = safeValue % defaultPalette.length;
          color = defaultPalette[colorIdx];
        }

        // Skip duplicate labels (can happen with rounding)
        if (addedLabels.has(label)) {
          return;
        }
        addedLabels.add(label);
        items.push({ label, color });
      });

      // Add N/A entry for null/NaN/sentinel values if present
      // For time_scale, "No Horizon" is handled as a regular category in the main loop
      if (hasNullValues && colorBy !== 'time_scale') {
        items.push({ label: 'N/A', color: '#999999' });
      }

      return { type: 'categorical' as const, items };
    } else {
      // Gradient legend for continuous values
      const TIME_FIELDS = ['time_horizon', 'chosen_time', 'alt_time', 'short_term_time', 'long_term_time'];
      const isTimeField = TIME_FIELDS.includes(colorBy);
      const isLogTime = false; // Removed log_time_horizon option

      // Add special indicators for time_horizon only (it has sentinel values)
      // Other time fields (chosen_time, alt_time, etc.) don't have sentinel values
      const legendItems: LegendItem[] = colorBy === 'time_horizon' ? [
        { label: 'No horizon', color: '#596680' },  // Blue-grey to match timeGradientColors
        { label: 'Out of range', color: '#d9d9d9' },
      ] : [];

      // Plasma colormap - MUST MATCH useEmbeddings.ts plasmaColor()
      // This is the actual colormap used for time gradient coloring
      const PLASMA_COLORS = [
        '#0d0887', // Dark blue-purple
        '#41049d', // Purple
        '#6a00a8', // Magenta
        '#8f0da4', // Pink-magenta
        '#b12a90', // Pink
        '#cc4778', // Salmon pink
        '#e16462', // Coral
        '#f2844b', // Orange
        '#fca636', // Yellow-orange
        '#fcce25', // Yellow
        '#f0f921', // Bright yellow
      ];

      // For adaptive scale, show tier labels instead of min/max
      if (isTimeField && timeScaleType === 'adaptive') {
        const tiers = getAdaptiveTierLabels();
        return {
          type: 'adaptive' as const,
          items: legendItems,
          tiers: tiers.map(t => ({
            ...t,
            label: t.label.charAt(0).toUpperCase() + t.label.slice(1), // Capitalize
          })),
          colors: PLASMA_COLORS,
        };
      }

      // For linear/log/blend, show min/max labels
      let minLabel: string;
      let maxLabel: string;

      if (isTimeField) {
        minLabel = formatTimeValue(effectiveColorRange.min, isLogTime);
        maxLabel = formatTimeValue(effectiveColorRange.max, isLogTime);
      } else if (colorBy === 'long_term_delay') {
        minLabel = `${Math.round(effectiveColorRange.min)} days`;
        maxLabel = `${Math.round(effectiveColorRange.max)} days`;
      } else {
        minLabel = String(Math.round(effectiveColorRange.min * 10) / 10);
        maxLabel = String(Math.round(effectiveColorRange.max * 10) / 10);
      }

      return {
        type: 'gradient' as const,
        items: legendItems.length > 0 ? legendItems : undefined,
        gradient: {
          minLabel,
          maxLabel,
          colors: PLASMA_COLORS,
        },
      };
    }
  }, [metadata, colorBy, effectiveColorRange, timeScaleType]);

  // Get human-readable color label
  const colorByLabel = COLOR_LABELS[colorBy] || colorBy.replace(/_/g, ' ');

  // Handle point selection
  const handlePointSelect = useCallback(
    (_index: number | null, data: PointData | null) => {
      if (data) {
        setSelectedSampleIdx(data.sampleIdx);
      } else {
        setSelectedSampleIdx(null);
      }
    },
    []
  );

  // Calculate legend panel width for export (publication-ready)
  const getLegendPanelWidth = useCallback((scale: number) => {
    if (!legendData || legendCollapsed) return 0;
    return 120 * scale; // Compact legend panel width
  }, [legendData, legendCollapsed]);

  // Helper to draw publication-ready legend panel on the right side of export canvas
  const drawLegendPanel = useCallback((ctx: CanvasRenderingContext2D, plotWidth: number, canvasHeight: number, scale: number) => {
    if (!legendData || legendCollapsed) return;

    // Compact, professional sizing
    const margin = 16 * scale;
    const itemHeight = 18 * scale;
    const fontSize = 10 * scale;
    const swatchSize = 10 * scale;
    const swatchGap = 6 * scale;
    const titleGap = 12 * scale;

    // Get items to display - use EXACT colors from legendData
    let items: { label: string; color: string }[] = [];
    if (legendData.type === 'categorical' && legendData.items) {
      // Categorical: use item colors directly (already correct)
      items = legendData.items;
    } else if (legendData.type === 'adaptive' && legendData.tiers) {
      // Adaptive: use legendData.colors if available (matches plot exactly)
      const plotColors = legendData.colors || ['#30123b', '#4777ef', '#1bd0d5', '#62fc6b', '#d2e935', '#fe9b2d', '#d23105'];
      items = legendData.tiers.map((tier) => {
        const colorIndex = Math.min(Math.floor(tier.position * (plotColors.length - 1)), plotColors.length - 1);
        return { label: tier.label, color: plotColors[colorIndex] };
      });
    }

    if (items.length === 0) return;

    // Calculate dimensions
    const titleHeight = fontSize + titleGap;
    const contentHeight = titleHeight + items.length * itemHeight;

    // Position: top-right of legend panel area, with margin
    const x = plotWidth + margin;
    const y = margin;

    // White background (already filled by parent)
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(plotWidth, 0, 200 * scale, canvasHeight);

    // Title - italic, smaller
    ctx.fillStyle = '#333333';
    ctx.font = `italic ${fontSize}px Arial, sans-serif`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    const title = colorBy.replace(/_/g, ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    ctx.fillText(title, x, y);

    // Items
    let currentY = y + titleHeight;
    ctx.font = `${fontSize}px Arial, sans-serif`;

    items.forEach((item) => {
      // Color swatch - filled rectangle, no border
      ctx.fillStyle = item.color;
      ctx.fillRect(x, currentY + (itemHeight - swatchSize) / 2, swatchSize, swatchSize);

      // Label
      ctx.fillStyle = '#000000';
      ctx.textBaseline = 'middle';
      ctx.fillText(item.label, x + swatchSize + swatchGap, currentY + itemHeight / 2);

      currentY += itemHeight;
    });
  }, [legendData, colorBy, legendCollapsed]);

  // Handle export - exports highest-res PNG of the current plot (universal for all plot types)
  const handleExport = useCallback(async () => {
    try {
      const container = plotContainerRef.current;
      if (!container) {
        console.error('Export: No plot container found');
        return;
      }

    const scale = 12; // 12x resolution for publication quality (300+ DPI at typical sizes)

    // Create filename: L{layer}__{position}__{colorBy}__{dataset}.png
    const positionClean = (position || 'all').replace(/:/g, '_');
    const filename = `L${layer}__${positionClean}__${colorBy}__${dataset}.png`;

    // Calculate legend panel width (0 if collapsed)
    const legendPanelWidth = getLegendPanelWidth(scale);

    // For 2D view, use the dedicated export ref for best quality
    if (viewMode === '2D' && scatterPlot2DRef.current) {
      const blob = await scatterPlot2DRef.current.exportPNG(scale);
      if (blob) {
        const img = new Image();
        const blobUrl = URL.createObjectURL(blob);
        img.onload = () => {
          // Create wider canvas to fit legend panel on right
          const plotWidth = img.width;
          const plotHeight = img.height;
          const exportCanvas = document.createElement('canvas');
          exportCanvas.width = plotWidth + legendPanelWidth;
          exportCanvas.height = plotHeight;

          const ctx = exportCanvas.getContext('2d');
          if (ctx) {
            // Fill with white background (publication-ready)
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

            // Draw the plot
            ctx.drawImage(img, 0, 0);

            // Replace beige background with white for publication
            const imageData = ctx.getImageData(0, 0, plotWidth, plotHeight);
            const data = imageData.data;
            for (let i = 0; i < data.length; i += 4) {
              const r = data[i], g = data[i + 1], b = data[i + 2];
              if ((r > 245 && g > 243 && b > 240) || (r < 30 && g < 26 && b < 23)) {
                data[i] = 255;
                data[i + 1] = 255;
                data[i + 2] = 255;
              }
            }
            ctx.putImageData(imageData, 0, 0);

            // Draw legend panel on the right (if not collapsed)
            drawLegendPanel(ctx, plotWidth, plotHeight, scale);

            exportCanvas.toBlob((finalBlob) => {
              if (finalBlob) {
                const url = URL.createObjectURL(finalBlob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
              }
            }, 'image/png', 1.0);
          }
          URL.revokeObjectURL(blobUrl);
        };
        img.src = blobUrl;
        console.log('Exported 2D plot PNG (publication-ready)');
      }
      return;
    }

    // Universal export: find canvas element in container
    const canvas = container.querySelector('canvas') as HTMLCanvasElement | null;

    if (canvas) {
      // For 2D canvases (TrajectoryPlot), the actual canvas dimensions include DPR scaling
      // For WebGL canvases (Three.js), they match display size
      // Use clientWidth/Height as the logical display size, or fall back to actual canvas size
      const dpr = window.devicePixelRatio || 1;
      const displayWidth = canvas.clientWidth || (canvas.width / dpr);
      const displayHeight = canvas.clientHeight || (canvas.height / dpr);

      if (displayWidth === 0 || displayHeight === 0) {
        console.error('Export: Canvas has zero dimensions');
        return;
      }

      const plotWidth = displayWidth * scale;
      const plotHeight = displayHeight * scale;

      const exportCanvas = document.createElement('canvas');
      exportCanvas.width = plotWidth + legendPanelWidth;
      exportCanvas.height = plotHeight;

      const ctx = exportCanvas.getContext('2d');
      if (ctx) {
        // Fill with white background (publication-ready)
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        // Draw the source canvas scaled to the export size
        ctx.drawImage(canvas, 0, 0, plotWidth, plotHeight);

        // Replace beige background (#faf8f5) with pure white for publication
        const imageData = ctx.getImageData(0, 0, plotWidth, plotHeight);
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
          const r = data[i], g = data[i + 1], b = data[i + 2];
          // Check if pixel is close to beige background color (250, 248, 245) or dark mode (26, 22, 19)
          if ((r > 245 && g > 243 && b > 240) || (r < 30 && g < 26 && b < 23)) {
            data[i] = 255;     // R
            data[i + 1] = 255; // G
            data[i + 2] = 255; // B
          }
        }
        ctx.putImageData(imageData, 0, 0);

        // Draw legend panel on the right (if not collapsed)
        drawLegendPanel(ctx, plotWidth, plotHeight, scale);

        exportCanvas.toBlob((blob) => {
          if (blob) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            console.log('Export: Downloaded successfully', { filename, size: blob.size });
          } else {
            console.error('Export: toBlob returned null');
          }
        }, 'image/png', 1.0);
      }
      return;
    }

    // For SVG plots (ScreePlot, AlignmentHeatmap)
    const svg = container.querySelector('svg');
    if (svg) {
      const svgClone = svg.cloneNode(true) as SVGSVGElement;
      const svgData = new XMLSerializer().serializeToString(svgClone);
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });

      const img = new Image();
      const url = URL.createObjectURL(svgBlob);

      img.onload = () => {
        const width = svg.clientWidth || 800;
        const height = svg.clientHeight || 600;
        const plotWidth = width * scale;
        const plotHeight = height * scale;

        const exportCanvas = document.createElement('canvas');
        exportCanvas.width = plotWidth + legendPanelWidth;
        exportCanvas.height = plotHeight;

        const ctx = exportCanvas.getContext('2d');
        if (ctx) {
          // Fill with white background (publication-ready)
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
          ctx.drawImage(img, 0, 0, plotWidth, plotHeight);

          // Draw legend panel on the right (if not collapsed)
          drawLegendPanel(ctx, plotWidth, plotHeight, scale);

          exportCanvas.toBlob((blob) => {
            if (blob) {
              const downloadUrl = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = downloadUrl;
              a.download = filename;
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
              URL.revokeObjectURL(downloadUrl);
              console.log('Exported SVG as PNG with legend successfully');
            }
          }, 'image/png', 1.0);
        }
        URL.revokeObjectURL(url);
      };
      img.src = url;
      return;
    }

    console.error('Export: No canvas or SVG found in plot container');
    } catch (error) {
      console.error('Export: Error during export', error);
    }
  }, [viewMode, colorBy, dataset, layer, position, drawLegendPanel, getLegendPanelWidth]);

  // Clear selection
  const handleClearSelection = useCallback(() => {
    setSelectedSampleIdx(null);
  }, []);

  // Random sample selection
  const handleRandomSelect = useCallback(() => {
    // Get visible sample indices based on view mode
    let visibleIndices: number[];

    if (isTrajectoryViewMode(viewMode)) {
      // For trajectory views, use trajectorySampleIndices with trajectoryFilterMask
      if (trajectoryFilterMask) {
        visibleIndices = trajectorySampleIndices.filter((_, i) => trajectoryFilterMask[i]);
      } else {
        visibleIndices = trajectorySampleIndices;
      }
    } else {
      // For scatter views (2D/3D), use embedding.indices with scatterFilterMask
      if (!embedding?.indices) return;
      if (scatterFilterMask) {
        visibleIndices = embedding.indices.filter((_, i) => scatterFilterMask[i]);
      } else {
        visibleIndices = embedding.indices;
      }
    }

    if (visibleIndices.length === 0) return;

    // Pick a random sample from visible indices
    const randomIndex = Math.floor(Math.random() * visibleIndices.length);
    setSelectedSampleIdx(visibleIndices[randomIndex]);
  }, [viewMode, trajectorySampleIndices, trajectoryFilterMask, embedding?.indices, scatterFilterMask]);

  // Derived state
  const isLoading = configLoading || embeddingLoading || metadataLoading;
  const layers = config?.layers || [];
  const components = config?.components || [DEFAULT_COMPONENT];
  const positions_options = config?.positions || [];
  const methods = config?.methods || [DEFAULT_METHOD];
  const colorByOptions = config?.colorByOptions || [DEFAULT_COLOR_BY];

  // Log overall loading state
  useEffect(() => {
    if (isLoading) {
      const waiting: string[] = [];
      if (configLoading) waiting.push('config');
      if (embeddingLoading) waiting.push('embedding');
      if (metadataLoading) waiting.push('metadata');
      log('App', `🔄 LOADING: Waiting for [${waiting.join(', ')}]`);
    } else {
      log('App', `✅ READY: All data loaded`);
    }
  }, [isLoading, configLoading, embeddingLoading, metadataLoading]);

  // Check if all samples have been filtered out
  const allSamplesFiltered = !isLoading && embedding?.positions && embedding.positions.length > 0 && positions.length === 0;

  return (
    <div className={`min-h-screen ${isDarkMode ? 'dark bg-[#1a1613]' : 'bg-gradient-main'}`}>
      {/* Header */}
      <Header
        datasetName={dataset}
        modelName={config?.modelName}
        totalSamples={config?.totalSamples || 0}
        totalLayers={config?.layers?.length || 0}
        totalPositions={config?.positions?.length || 0}
        isDarkMode={isDarkMode}
        onDarkModeChange={setIsDarkMode}
        onExport={handleExport}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
      />

      {/* Main Content */}
      <div className="flex h-[calc(100vh-80px)]">
        {/* Left Sidebar - Controls */}
        <aside className="w-72 flex-shrink-0 p-4 overflow-y-auto border-r border-white/40 dark:border-[#3a3633] bg-white/30 dark:bg-[#1a1613]/50 backdrop-blur-sm">
          <ControlPanel
            layer={layer}
            layers={layers}
            onLayerChange={setLayer}
            hideLayerSection={viewMode === '1DxLayer' || viewMode === '2DxLayer' || viewMode === 'Align'}
            component={component}
            components={components}
            onComponentChange={setComponent}
            hideComponentSection={viewMode === 'Scree' || viewMode === 'Align'}
            position={position}
            positions={positions_options}
            positionLabels={config?.positionLabels || {}}
            promptTemplate={config?.promptTemplate || []}
            onPositionChange={setPosition}
            hidePositionSection={true}
            method={method}
            methods={methods}
            onMethodChange={setMethod}
            hideMethodSection={isTrajectoryViewMode(viewMode) || viewMode === 'Scree' || viewMode === 'Align'}
            hideColorBySection={true}
            colorBy={colorBy}
            colorByOptions={colorByOptions}
            onColorByChange={setColorBy}
            colorRangeMin={colorRangeMin}
            colorRangeMax={colorRangeMax}
            colorRangeDataMin={effectiveColorRange.dataMin}
            colorRangeDataMax={effectiveColorRange.dataMax}
            onColorRangeMinChange={setColorRangeMin}
            onColorRangeMaxChange={setColorRangeMax}
            showColorRangeControls={GRADIENT_FIELDS.includes(colorBy) && viewMode !== 'Scree' && viewMode !== 'Align'}
            timeScaleType={timeScaleType}
            onTimeScaleTypeChange={setTimeScaleType}
            blendMix={blendMix}
            onBlendMixChange={setBlendMix}
            showTimeScaleControls={['time_horizon', 'chosen_time', 'alt_time', 'short_term_time', 'long_term_time'].includes(colorBy) && viewMode !== 'Scree' && viewMode !== 'Align'}
          />
        </aside>

        {/* Main Visualization Area */}
        <main className="flex-1 p-4 flex flex-col min-w-0 min-h-0 dark:bg-[#1a1613]">
          {/* Error State */}
          {(embeddingError || metadataError || hasHorizonError || layerTrajectory.error || positionTrajectory.error) && (
            <div className="mb-4 p-4 bg-rose-50 dark:bg-rose-900/30 border border-rose-200 dark:border-rose-800 rounded-xl text-rose-700 dark:text-rose-300">
              <strong>No data available:</strong>{' '}
              {(embeddingError as Error)?.message ||
               (metadataError as Error)?.message ||
               (hasHorizonError as Error)?.message ||
               layerTrajectory.error?.message ||
               positionTrajectory.error?.message ||
               'This position has no pre-computed embeddings'}
            </div>
          )}

          {/* Visualization Toolbar - above plot (hidden for Scree/Align) */}
          {viewMode !== 'Scree' && viewMode !== 'Align' && (
            <div className="flex items-center justify-between mb-2 px-2">
              {/* PCA mode toggle - only for trajectory views */}
              {isTrajectoryViewMode(viewMode) && (
                <div className="flex items-center gap-2 bg-white/95 dark:bg-[#2a2623] backdrop-blur-sm rounded-lg shadow-sm border border-white/60 dark:border-[#3a3633] px-3 py-1.5">
                  <span className="text-xs font-medium text-slate-600 dark:text-slate-300">PCA:</span>
                  <Select
                    value={pcaMode}
                    onChange={(value) => setPcaMode(value as PCAMode)}
                    options={[
                      { value: 'aligned', label: 'Per-target (aligned)' },
                      { value: 'shared', label: 'Shared subspace' },
                    ]}
                    className="w-40"
                  />
                </div>
              )}
              {/* Spacer when no PCA toggle */}
              {!isTrajectoryViewMode(viewMode) && <div />}
              {/* Horizon filter toggles */}
              <div className="flex items-center gap-3 bg-white/95 dark:bg-[#2a2623] backdrop-blur-sm rounded-lg shadow-sm border border-white/60 dark:border-[#3a3633] px-3 py-1.5">
                <Toggle
                  checked={showWithHorizon}
                  onChange={setShowWithHorizon}
                  label="With-Horizon"
                  size="sm"
                />
                <Toggle
                  checked={showNoHorizon}
                  onChange={setShowNoHorizon}
                  label="No-Horizon"
                  size="sm"
                />
              </div>
            </div>
          )}

          {/* Visualization Area - requires min-h-[400px] for Canvas to render properly */}
          <div ref={plotContainerRef} className="flex-1 relative rounded-2xl overflow-hidden shadow-2xl shadow-purple-500/10 border border-white/60 dark:border-[#3a3633] min-h-[400px] bg-[#faf8f5] dark:bg-[#1a1613]">
            {allSamplesFiltered ? (
              <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-[#faf8f5] to-[#f5f0eb] dark:from-[#1a1613] dark:to-[#252220]">
                <div className="text-center p-8">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-amber-100 to-amber-200 dark:from-amber-900/30 dark:to-amber-800/30 flex items-center justify-center">
                    <svg
                      className="w-8 h-8 text-amber-600 dark:text-amber-400"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                      />
                    </svg>
                  </div>
                  <p className="text-[#1a1613]/80 dark:text-white/80 font-medium mb-2">
                    No samples to display
                  </p>
                  <p className="text-[#1a1613]/50 dark:text-white/50 text-sm">
                    All samples have been filtered out. Try enabling "With-Horizon" or "No-Horizon" toggles.
                  </p>
                </div>
              </div>
            ) : isLoading && !positions.length ? (
              <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-[#faf8f5] to-[#f5f0eb] dark:from-[#1a1613] dark:to-[#252220]">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-[#D97757] to-[#348296] animate-pulse flex items-center justify-center">
                    <svg
                      className="w-8 h-8 text-white animate-spin"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                  </div>
                  <p className="text-[#1a1613]/60 dark:text-white/60 font-medium">
                    {streamingProgress > 0
                      ? `Streaming... ${streamingProgress}%`
                      : 'Loading embedding data...'}
                  </p>
                  {streamingProgress > 0 && (
                    <div className="w-32 h-1.5 mt-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-[#D97757] to-[#348296] transition-all duration-150"
                        style={{ width: `${streamingProgress}%` }}
                      />
                    </div>
                  )}
                </div>
              </div>
            ) : viewMode === '3D' ? (
              <ScatterPlot3D
                positions={positions}
                colors={colors}
                pointData={pointData}
                pointSize={5}
                showAxes={false}
                showGrid={true}
                onPointSelect={handlePointSelect}
                backgroundColor={isDarkMode ? '#1a1613' : '#faf8f5'}
                initialCameraPosition={[8, 6, 8]}
                className="absolute inset-0"
                selectedSampleIdx={selectedSampleIdx}
                visibility={visibility}
              />
            ) : viewMode === '2D' ? (
              <ScatterPlot2D
                ref={scatterPlot2DRef}
                positions={positions}
                colors={colors}
                pointData={pointData}
                pointSize={4}
                showAxes={true}
                showGrid={true}
                onPointSelect={handlePointSelect}
                backgroundColor={isDarkMode ? '#1a1613' : '#faf8f5'}
                className="absolute inset-0"
                selectedSampleIdx={selectedSampleIdx}
                xAxis={0}
                yAxis={1}
                visibility={visibility}
              />
            ) : viewMode === '1DxLayer' ? (
              <TrajectoryPlot
                trajectoryData={layerTrajectory.trajectoryData}
                xValues={layerTrajectory.xValues}
                xAxisLabel="Layer"
                yAxisLabel="PC1 Projection"
                title={`PC1 Trajectory Across Layers (${position} @ ${component})`}
                colors={unfilteredColors}
                pointData={unfilteredPointData}
                filterMask={filterMask}
                backgroundColor={isDarkMode ? '#1a1613' : '#faf8f5'}
                showGrid={true}
                onPointSelect={handlePointSelect}
                selectedSampleIdx={selectedSampleIdx}
                lineOpacity={0.5}
                isLoading={layerTrajectory.isLoading}
                loadingProgress={0}
                className="absolute inset-0"
              />
            ) : viewMode === '1DxPos' ? (
              <TrajectoryPlot
                trajectoryData={positionTrajectory.trajectoryData}
                xValues={positionTrajectory.xValues}
                xAxisLabel="Position"
                yAxisLabel="PC1 Projection"
                title={`PC1 Trajectory Across Positions (L${layer} @ ${component})`}
                colors={unfilteredColors}
                pointData={unfilteredPointData}
                filterMask={filterMask}
                backgroundColor={isDarkMode ? '#1a1613' : '#faf8f5'}
                showGrid={true}
                onPointSelect={handlePointSelect}
                selectedSampleIdx={selectedSampleIdx}
                lineOpacity={0.5}
                isLoading={positionTrajectory.isLoading}
                loadingProgress={0}
                className="absolute inset-0"
              />
            ) : viewMode === '2DxLayer' ? (
              <TrajectoryPlot3D
                pc1Data={layerTrajectory2D.pc1Data}
                pc2Data={layerTrajectory2D.pc2Data}
                xValues={layerTrajectory2D.xValues}
                xAxisLabel="Layer"
                title={`PC1 vs PC2 Across Layers (${position} @ ${component})`}
                colors={unfilteredColors}
                pointData={unfilteredPointData}
                filterMask={filterMask}
                backgroundColor={isDarkMode ? '#1a1613' : '#faf8f5'}
                onPointSelect={handlePointSelect}
                selectedSampleIdx={selectedSampleIdx}
                lineOpacity={0.5}
                isLoading={layerTrajectory2D.isLoading}
                className="absolute inset-0"
              />
            ) : viewMode === '2DxPos' ? (
              <TrajectoryPlot3D
                pc1Data={positionTrajectory2D.pc1Data}
                pc2Data={positionTrajectory2D.pc2Data}
                xValues={positionTrajectory2D.xValues}
                xAxisLabel="Position"
                title={`PC1 vs PC2 Across Positions (L${layer} @ ${component})`}
                colors={unfilteredColors}
                pointData={unfilteredPointData}
                filterMask={filterMask}
                backgroundColor={isDarkMode ? '#1a1613' : '#faf8f5'}
                onPointSelect={handlePointSelect}
                selectedSampleIdx={selectedSampleIdx}
                lineOpacity={0.5}
                isLoading={positionTrajectory2D.isLoading}
                className="absolute inset-0"
              />
            ) : viewMode === 'Scree' ? (
              <ScreePlot
                data={screeData.data || null}
                isLoading={screeData.isLoading}
                selectedLayer={layer}
                className="w-full h-full"
              />
            ) : viewMode === 'Align' ? (
              <AlignmentHeatmap
                data={alignmentData.data || null}
                isLoading={alignmentData.isLoading}
                position={position}
                className="w-full h-full"
              />
            ) : null}

            {/* Loading overlay when updating */}
            {isLoading && positions.length > 0 && (
              <div className="absolute inset-0 bg-white/50 dark:bg-black/50 backdrop-blur-sm flex items-center justify-center pointer-events-none z-20">
                <div className="px-4 py-2 bg-white/90 dark:bg-[#2a2623]/90 rounded-full shadow-lg border border-purple-100/50 dark:border-purple-900/50 flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full border-2 border-[#D97757] border-t-transparent animate-spin" />
                  <span className="text-sm text-[#1a1613]/70 dark:text-white/70">Updating...</span>
                </div>
              </div>
            )}

            {/* Legend - hidden for Scree/Align views */}
            {legendData && viewMode !== 'Scree' && viewMode !== 'Align' && (
              <Legend
                title={colorByLabel}
                items={legendData.items}
                gradient={legendData.type === 'gradient' ? legendData.gradient : undefined}
                tiers={legendData.type === 'adaptive' ? legendData.tiers : undefined}
                tierColors={legendData.type === 'adaptive' ? legendData.colors : undefined}
                collapsed={legendCollapsed}
                onCollapsedChange={setLegendCollapsed}
              />
            )}

          </div>
        </main>

        {/* Right Sidebar - Position, Color By, and Info Panel */}
        <aside className="w-80 flex-shrink-0 p-4 overflow-y-auto border-l border-white/40 dark:border-[#3a3633] bg-white/30 dark:bg-[#1a1613]/30 backdrop-blur-sm">
          <div className="flex flex-col gap-4">
            {/* Position Selector - hidden for position trajectory views */}
            {viewMode !== '1DxPos' && viewMode !== '2DxPos' && (config?.promptTemplate?.length ?? 0) > 0 && (
              <Card padding="sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Position</CardTitle>
                </CardHeader>
                <CardContent className="py-2">
                  <PositionSelector
                    position={position}
                    promptTemplate={config?.promptTemplate || []}
                    positionLabels={config?.positionLabels || {}}
                    onPositionChange={setPosition}
                    exampleSample={config?.exampleSample}
                    relPosCounts={config?.relPosCounts}
                    isDarkMode={isDarkMode}
                  />
                </CardContent>
              </Card>
            )}

            {/* Color By Control - hidden for Scree/Align views */}
            {viewMode !== 'Scree' && viewMode !== 'Align' && (
              <Card padding="sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Color By</CardTitle>
                </CardHeader>
                <CardContent className="py-2">
                  <Select
                    options={colorByOptions.map((c) => ({
                      value: c,
                      label: COLOR_LABELS[c] || c.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
                    }))}
                    value={colorBy}
                    onChange={setColorBy}
                    placeholder="Select attribute..."
                  />
                </CardContent>
              </Card>
            )}

            {/* Filter Panel - hidden for Scree/Align views */}
            {viewMode !== 'Scree' && viewMode !== 'Align' && (
            <FilterPanel
              shortRewardFilter={shortRewardFilter}
              shortTimeFilter={shortTimeFilter}
              longRewardFilter={longRewardFilter}
              longTimeFilter={longTimeFilter}
              onShortRewardFilterChange={setShortRewardFilter}
              onShortTimeFilterChange={setShortTimeFilter}
              onLongRewardFilterChange={setLongRewardFilter}
              onLongTimeFilterChange={setLongTimeFilter}
            />
            )}

            {/* Selected Sample Info - hidden for Scree/Align views */}
            {viewMode !== 'Scree' && viewMode !== 'Align' && (
            <InfoPanel
              selectedSample={
                selectedSample
                  ? {
                      idx: selectedSample.idx,
                      text: selectedSample.text,
                      timeHorizon: selectedSample.timeHorizon,
                      timeScale: selectedSample.timeScale,
                      choiceType: selectedSample.choiceType,
                      shortTermFirst: selectedSample.shortTermFirst,
                      label: selectedSample.label,
                      responseLabel: selectedSample.responseLabel,
                      responseTerm: selectedSample.responseTerm,
                      responseText: selectedSample.responseText,
                      choiceConfidence: selectedSample.choiceConfidence,
                    }
                  : null
              }
              isLoading={sampleLoading}
              onClose={handleClearSelection}
              onRandomSelect={handleRandomSelect}
              markers={config?.markers}
            />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
