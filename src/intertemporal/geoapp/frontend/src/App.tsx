import { useState, useMemo, useCallback, useEffect } from 'react';
import { Header, ControlPanel, InfoPanel, ScatterPlot3D, ScatterPlot2D, Legend, LegendItem, PositionSelector, TrajectoryPlot } from './components';
import { Toggle } from './components/ui/Toggle';
import { Select } from './components/ui/Select';
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/Card';
import { PointData } from './components/PointCloud';
import {
  useConfig,
  useEmbedding,
  useMetadata,
  useSample,
  usePrefetch,
  useBackendPrefetch,
  useLayerTrajectory,
  usePositionTrajectory,
  toFloat32Array,
  valuesToColors,
  categoricalColors,
  timeGradientColors,
  getAdaptiveTierLabels,
  TimeScaleType,
} from './hooks/useEmbeddings';

// Default values - these are fallbacks, actual values come from config
const DEFAULT_COMPONENT = 'resid_post';
const DEFAULT_METHOD = 'pca';
const DEFAULT_COLOR_BY = 'time_horizon';

type ViewMode = '2D' | '3D' | '1DxLayer' | '1DxPos';

function App() {
  // UI state
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('3D');

  // Control state - layer and position are initialized to -1/'', then set from config
  const [layer, setLayer] = useState<number>(-1);
  const [component, setComponent] = useState(DEFAULT_COMPONENT);
  const [position, setPosition] = useState<string>('');
  const [method, setMethod] = useState(DEFAULT_METHOD);
  const [colorBy, setColorBy] = useState(DEFAULT_COLOR_BY);
  const [showNoHorizon, setShowNoHorizon] = useState(true);

  // Color range controls for gradient coloring
  const [colorRangeMin, setColorRangeMin] = useState<number | null>(null);
  const [colorRangeMax, setColorRangeMax] = useState<number | null>(null);

  // Time scale transfer function controls
  const [timeScaleType, setTimeScaleType] = useState<TimeScaleType>('linear');
  const [blendMix, setBlendMix] = useState(0.5);

  // Selection state
  const [selectedSampleIdx, setSelectedSampleIdx] = useState<number | null>(null);

  // Clear selection when embedding parameters change to avoid stale references
  useEffect(() => {
    setSelectedSampleIdx(null);
  }, [layer, component, position, method]);

  // Fetch config
  const { data: config, isLoading: configLoading } = useConfig();

  // Initialize layer and position from config when it first loads
  useEffect(() => {
    if (config) {
      // Set layer to first available if not yet set or invalid
      if (layer < 0 || !config.layers.includes(layer)) {
        const firstLayer = config.layers[0];
        if (firstLayer !== undefined) {
          setLayer(firstLayer);
        }
      }
      // Set position to first available if not yet set or invalid
      if (!position || !config.positions.includes(position)) {
        const firstPosition = config.positions[0];
        if (firstPosition) {
          setPosition(firstPosition);
        }
      }
    }
  }, [config, layer, position]);

  // Fetch embedding data
  const {
    data: embedding,
    isLoading: embeddingLoading,
    error: embeddingError,
  } = useEmbedding(layer, component, position, method);

  // Fetch metadata for coloring
  const { data: metadata, isLoading: metadataLoading, error: metadataError } = useMetadata(colorBy);

  // Fetch has_horizon metadata for filtering
  const { data: hasHorizonMeta, error: hasHorizonError } = useMetadata('has_horizon');

  // Fetch selected sample details
  const { data: selectedSample, isLoading: sampleLoading } = useSample(
    selectedSampleIdx
  );

  // Prefetch adjacent layers and all color options in background
  usePrefetch(
    layer,
    component,
    position,
    method,
    config?.colorByOptions || [],
    config?.layers || []
  );

  // Trigger backend prefetch for adjacent embeddings (fire-and-forget)
  // This precomputes neighboring layers/positions server-side for seamless navigation
  useBackendPrefetch(layer, component, position, method, !embeddingLoading);

  // Fetch trajectory data for 1DxLayer view (PC1 across all layers)
  const layerTrajectory = useLayerTrajectory(
    component,
    position,
    viewMode === '1DxLayer'
  );

  // Fetch trajectory data for 1DxPos view (PC1 across positions)
  const positionTrajectory = usePositionTrajectory(
    layer,
    component,
    undefined, // Use default named positions
    viewMode === '1DxPos'
  );

  // Compute filter mask for no-horizon samples
  const filterMask = useMemo(() => {
    if (!hasHorizonMeta?.values || showNoHorizon) {
      return null; // No filtering needed
    }
    // Filter out samples where has_horizon = 0 (no horizon)
    return hasHorizonMeta.values.map(v => v === 1);
  }, [hasHorizonMeta?.values, showNoHorizon]);

  // Compute positions Float32Array (with filtering)
  const positions = useMemo(() => {
    if (!embedding?.positions) {
      return new Float32Array(0);
    }
    if (!filterMask) {
      return toFloat32Array(embedding.positions);
    }
    // Filter positions
    const filtered: number[] = [];
    for (let i = 0; i < filterMask.length; i++) {
      if (filterMask[i]) {
        filtered.push(
          embedding.positions[i * 3],
          embedding.positions[i * 3 + 1],
          embedding.positions[i * 3 + 2]
        );
      }
    }
    return new Float32Array(filtered);
  }, [embedding?.positions, filterMask]);

  // Fields that should always use gradient coloring (not categorical)
  const GRADIENT_FIELDS = ['time_horizon', 'long_term_delay', 'sample_idx'];

  // Compute effective color range (user-defined or data-derived)
  const effectiveColorRange = useMemo(() => {
    if (!metadata?.values) return { min: 0, max: 1 };

    // For time-related fields, filter out no-horizon (sentinel -1) values for range calculation
    const isTimeField = colorBy === 'time_horizon' ;
    const valuesForRange = isTimeField
      ? metadata.values.filter(v => v >= 0)
      : metadata.values;

    const dataMin = valuesForRange.length > 0 ? Math.min(...valuesForRange) : 0;
    const dataMax = valuesForRange.length > 0 ? Math.max(...valuesForRange) : 1;

    return {
      min: colorRangeMin ?? dataMin,
      max: colorRangeMax ?? dataMax,
      dataMin,
      dataMax,
    };
  }, [metadata?.values, colorBy, colorRangeMin, colorRangeMax]);

  // Compute colors Float32Array (with filtering)
  const colors = useMemo(() => {
    if (!metadata?.values || metadata.values.length === 0) {
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

    // Apply filter mask if needed
    const filteredValues = filterMask
      ? metadata.values.filter((_, i) => filterMask[i])
      : metadata.values;

    // Check if categorical or continuous (force gradient for certain fields)
    const uniqueValues = new Set(filteredValues);
    const forceGradient = GRADIENT_FIELDS.includes(colorBy);
    const isCategorical = !forceGradient && uniqueValues.size <= 10;

    if (isCategorical) {
      return categoricalColors(filteredValues, uniqueValues.size);
    } else {
      // For time-related fields, use special coloring with gray for no-horizon (value 0)
      const isTimeField = colorBy === 'time_horizon' ;
      if (isTimeField) {
        return timeGradientColors(
          filteredValues,
          effectiveColorRange.min,
          effectiveColorRange.max,
          timeScaleType,
          blendMix
        );
      }
      return valuesToColors(filteredValues, effectiveColorRange.min, effectiveColorRange.max, 'turbo');
    }
  }, [metadata, filterMask, positions.length, colorBy, effectiveColorRange, timeScaleType, blendMix]);

  // Create point data array (with filtering)
  const pointData = useMemo<PointData[]>(() => {
    if (!embedding?.indices) return [];
    if (!filterMask) {
      return embedding.indices.map((idx) => ({
        sampleIdx: idx,
      }));
    }
    // Filter point data
    return embedding.indices
      .filter((_, i) => filterMask[i])
      .map((idx) => ({
        sampleIdx: idx,
      }));
  }, [embedding?.indices, filterMask]);

  // Unfiltered colors for trajectory views (all samples)
  const unfilteredColors = useMemo(() => {
    // Use embedding indices length as canonical sample count
    const numPoints = embedding?.indices?.length || 0;

    if (!metadata?.values || metadata.values.length === 0) {
      // Default gradient colors when no metadata
      const defaultColors = new Float32Array(numPoints * 3);
      for (let i = 0; i < numPoints; i++) {
        const t = i / (numPoints - 1 || 1);
        defaultColors[i * 3] = 0.78 + t * 0.22;
        defaultColors[i * 3 + 1] = 0.47 - t * 0.05;
        defaultColors[i * 3 + 2] = 0.87 - t * 0.25;
      }
      return defaultColors;
    }

    const uniqueValues = new Set(metadata.values);
    const forceGradient = GRADIENT_FIELDS.includes(colorBy);
    const isCategorical = !forceGradient && uniqueValues.size <= 10;

    if (isCategorical) {
      return categoricalColors(metadata.values, uniqueValues.size);
    } else {
      const isTimeField = colorBy === 'time_horizon';
      if (isTimeField) {
        return timeGradientColors(
          metadata.values,
          effectiveColorRange.min,
          effectiveColorRange.max,
          timeScaleType,
          blendMix
        );
      }
      return valuesToColors(metadata.values, effectiveColorRange.min, effectiveColorRange.max, 'turbo');
    }
  }, [metadata, embedding?.indices?.length, colorBy, effectiveColorRange, timeScaleType, blendMix]);

  // Unfiltered point data for trajectory views
  const unfilteredPointData = useMemo<PointData[]>(() => {
    if (!embedding?.indices) return [];
    return embedding.indices.map((idx) => ({
      sampleIdx: idx,
    }));
  }, [embedding?.indices]);

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
    'chosen_delivery': 'Chosen Delivery',
    'has_horizon': 'Has Horizon',
    'short_term_first': 'Option Order',
    'context_id': 'Context',
    'sample_idx': 'Sample Index',
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
      // Default categorical palette - must match useEmbeddings.ts categoricalColors
      // Anthropic-inspired with high contrast
      const defaultPalette = [
        '#D97857', // Anthropic terracotta/coral
        '#348296', // Deep teal (high contrast)
        '#8F70DB', // Purple
        '#F2AD42', // Warm amber
        '#59A678', // Forest green
        '#E85D75', // Coral red
        '#6B8E9F', // Slate blue
        '#C4963A', // Bronze
      ];

      const items: LegendItem[] = [];
      const sortedValues = [...uniqueValues].sort((a, b) => Number(a) - Number(b));

      sortedValues.forEach((value) => {
        let label: string;
        // Color must match the actual point coloring logic in categoricalColors()
        // which uses Math.floor(value) % palette.length as the index
        const colorIdx = Math.floor(Number(value)) % defaultPalette.length;
        const color = defaultPalette[colorIdx];

        if (colorBy === 'has_horizon') {
          label = value === 1 ? 'Has horizon' : 'No horizon';
        } else if (colorBy === 'short_term_first') {
          label = value === 1 ? 'Short-term first' : 'Long-term first';
        } else if (colorBy === 'chosen_delivery') {
          label = value === 1 ? 'Long' : 'Short';
        } else {
          label = String(value);
        }
        items.push({ label, color });
      });

      return { type: 'categorical' as const, items };
    } else {
      // Gradient legend for continuous values
      const isTimeField = colorBy === 'time_horizon' ;
      const isLogTime = false; // Removed log_time_horizon option

      // Add special indicators for time fields
      const legendItems: LegendItem[] = isTimeField ? [
        { label: 'No horizon', color: '#596680' },  // Blue-grey to match timeGradientColors
        { label: 'Out of range', color: '#d9d9d9' },
      ] : [];

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
          colors: ['#30123b', '#4777ef', '#1bd0d5', '#62fc6b', '#d2e935', '#fe9b2d', '#d23105'], // Turbo
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
          colors: ['#30123b', '#4777ef', '#1bd0d5', '#62fc6b', '#d2e935', '#fe9b2d', '#d23105'], // Turbo
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

  // Handle export
  const handleExport = useCallback(() => {
    // TODO: Implement export functionality
    console.log('Export clicked');
  }, []);

  // Clear selection
  const handleClearSelection = useCallback(() => {
    setSelectedSampleIdx(null);
  }, []);

  // Derived state
  const isLoading = configLoading || embeddingLoading || metadataLoading;
  const layers = config?.layers || [];
  const components = config?.components || [DEFAULT_COMPONENT];
  const positions_options = config?.positions || [];
  const methods = config?.methods || [DEFAULT_METHOD];
  const colorByOptions = config?.colorByOptions || [DEFAULT_COLOR_BY];

  // Check if all samples have been filtered out
  const allSamplesFiltered = !isLoading && embedding?.positions && embedding.positions.length > 0 && positions.length === 0;

  return (
    <div className={`min-h-screen bg-gradient-main ${isDarkMode ? 'dark' : ''}`}>
      {/* Header */}
      <Header
        modelName={config?.modelName}
        totalSamples={config?.totalSamples || 0}
        totalLayers={config?.layers?.length || 0}
        totalPositions={config?.positions?.length || 0}
        isDarkMode={isDarkMode}
        onDarkModeChange={setIsDarkMode}
        onExport={handleExport}
      />

      {/* Main Content */}
      <div className="flex h-[calc(100vh-80px)]">
        {/* Left Sidebar - Controls */}
        <aside className="w-72 flex-shrink-0 p-4 overflow-y-auto border-r border-white/40 bg-white/30 backdrop-blur-sm">
          <ControlPanel
            layer={layer}
            layers={layers}
            onLayerChange={setLayer}
            component={component}
            components={components}
            onComponentChange={setComponent}
            position={position}
            positions={positions_options}
            positionLabels={config?.positionLabels || {}}
            promptTemplate={config?.promptTemplate || []}
            onPositionChange={setPosition}
            hidePositionSection={true}
            method={method}
            methods={methods}
            onMethodChange={setMethod}
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
            showColorRangeControls={GRADIENT_FIELDS.includes(colorBy)}
            timeScaleType={timeScaleType}
            onTimeScaleTypeChange={setTimeScaleType}
            blendMix={blendMix}
            onBlendMixChange={setBlendMix}
            showTimeScaleControls={colorBy === 'time_horizon' }
          />
        </aside>

        {/* Main Visualization Area */}
        <main className="flex-1 p-4 flex flex-col min-w-0">
          {/* Error State */}
          {(embeddingError || metadataError || hasHorizonError || layerTrajectory.error || positionTrajectory.error) && (
            <div className="mb-4 p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-700">
              <strong>Error loading data:</strong>{' '}
              {(embeddingError as Error)?.message ||
               (metadataError as Error)?.message ||
               (hasHorizonError as Error)?.message ||
               layerTrajectory.error?.message ||
               positionTrajectory.error?.message ||
               'Unknown error'}
            </div>
          )}

          {/* Visualization Toolbar - above plot */}
          <div className="flex items-center justify-between mb-2 px-2">
            {/* View mode toggle */}
            <div className="flex items-center gap-1 bg-white/95 backdrop-blur-sm rounded-lg shadow-sm border border-white/60 p-1">
              {(['2D', '3D', '1DxLayer', '1DxPos'] as ViewMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                    viewMode === mode
                      ? 'bg-gradient-to-r from-[#D97757] to-[#348296] text-white shadow-sm'
                      : 'text-[#1a1613]/70 hover:bg-gray-100'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>

            {/* No-horizon toggle */}
            <div className="bg-white/95 backdrop-blur-sm rounded-lg shadow-sm border border-white/60 px-3 py-1.5">
              <Toggle
                checked={showNoHorizon}
                onChange={setShowNoHorizon}
                label="Show no-horizon"
                size="sm"
              />
            </div>
          </div>

          {/* Visualization Area */}
          <div className="flex-1 relative rounded-2xl overflow-hidden shadow-2xl shadow-purple-500/10 border border-white/60">
            {allSamplesFiltered ? (
              <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-[#faf8f5] to-[#f5f0eb]">
                <div className="text-center p-8">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-amber-100 to-amber-200 flex items-center justify-center">
                    <svg
                      className="w-8 h-8 text-amber-600"
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
                  <p className="text-[#1a1613]/80 font-medium mb-2">
                    No samples to display
                  </p>
                  <p className="text-[#1a1613]/50 text-sm">
                    All samples have been filtered out. Try enabling "Show no-horizon samples" in the Filters section.
                  </p>
                </div>
              </div>
            ) : isLoading && !positions.length ? (
              <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-[#faf8f5] to-[#f5f0eb]">
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
                  <p className="text-[#1a1613]/60 font-medium">
                    Loading embedding data...
                  </p>
                </div>
              </div>
            ) : viewMode === '3D' ? (
              <ScatterPlot3D
                positions={positions}
                colors={colors}
                pointData={pointData}
                pointSize={5}
                showAxes={true}
                showGrid={true}
                onPointSelect={handlePointSelect}
                backgroundColor="#faf8f5"
                initialCameraPosition={[8, 6, 8]}
                className="w-full h-full"
                selectedSampleIdx={selectedSampleIdx}
              />
            ) : viewMode === '2D' ? (
              <ScatterPlot2D
                positions={positions}
                colors={colors}
                pointData={pointData}
                pointSize={4}
                showAxes={true}
                showGrid={true}
                onPointSelect={handlePointSelect}
                backgroundColor="#faf8f5"
                className="w-full h-full"
                selectedSampleIdx={selectedSampleIdx}
                xAxis={0}
                yAxis={1}
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
                filterMask={showNoHorizon ? null : filterMask}
                backgroundColor="#faf8f5"
                showGrid={true}
                onPointSelect={handlePointSelect}
                selectedSampleIdx={selectedSampleIdx}
                lineOpacity={0.5}
                isLoading={layerTrajectory.isLoading}
                loadingProgress={0}
                className="absolute inset-0"
              />
            ) : (
              <TrajectoryPlot
                trajectoryData={positionTrajectory.trajectoryData}
                xValues={positionTrajectory.xValues}
                xAxisLabel="Position"
                yAxisLabel="PC1 Projection"
                title={`PC1 Trajectory Across Positions (L${layer} @ ${component})`}
                colors={unfilteredColors}
                pointData={unfilteredPointData}
                filterMask={showNoHorizon ? null : filterMask}
                backgroundColor="#faf8f5"
                showGrid={true}
                onPointSelect={handlePointSelect}
                selectedSampleIdx={selectedSampleIdx}
                lineOpacity={0.5}
                isLoading={positionTrajectory.isLoading}
                loadingProgress={0}
                className="absolute inset-0"
              />
            )}

            {/* Loading overlay when updating */}
            {isLoading && positions.length > 0 && (
              <div className="absolute inset-0 bg-white/50 backdrop-blur-sm flex items-center justify-center pointer-events-none z-20">
                <div className="px-4 py-2 bg-white/90 rounded-full shadow-lg border border-purple-100/50 flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full border-2 border-[#D97757] border-t-transparent animate-spin" />
                  <span className="text-sm text-[#1a1613]/70">Updating...</span>
                </div>
              </div>
            )}

            {/* Legend */}
            {legendData && (
              <Legend
                title={colorByLabel}
                items={legendData.type === 'categorical' ? legendData.items : legendData.items}
                gradient={legendData.type === 'gradient' ? legendData.gradient : undefined}
                tiers={legendData.type === 'adaptive' ? legendData.tiers : undefined}
                tierColors={legendData.type === 'adaptive' ? legendData.colors : undefined}
              />
            )}

          </div>
        </main>

        {/* Right Sidebar - Position, Color By, and Info Panel */}
        <aside className="w-80 flex-shrink-0 p-4 overflow-y-auto border-l border-white/40 bg-white/30 backdrop-blur-sm">
          <div className="flex flex-col gap-4">
            {/* Position Selector */}
            {(config?.promptTemplate?.length ?? 0) > 0 && (
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
                  />
                </CardContent>
              </Card>
            )}

            {/* Color By Control */}
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

            {/* Selected Sample Info */}
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
                    }
                  : null
              }
              isLoading={sampleLoading}
              onClose={handleClearSelection}
              markers={config?.markers}
            />
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
