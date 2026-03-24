import { useState, useMemo, useCallback } from 'react';
import { Header, ControlPanel, InfoPanel, ScatterPlot3D, Legend, LegendItem } from './components';
import { PointData } from './components/PointCloud';
import {
  useConfig,
  useEmbedding,
  useMetadata,
  useSample,
  usePrefetch,
  toFloat32Array,
  valuesToColors,
  categoricalColors,
  timeHorizonColors,
  logTimeHorizonColors,
} from './hooks/useEmbeddings';

// Default values matching available data
const DEFAULT_LAYER = 13;
const DEFAULT_COMPONENT = 'resid_post';
const DEFAULT_POSITION = 'response';
const DEFAULT_METHOD = 'pca';
const DEFAULT_COLOR_BY = 'time_horizon';

function App() {
  // UI state
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Control state
  const [layer, setLayer] = useState(DEFAULT_LAYER);
  const [component, setComponent] = useState(DEFAULT_COMPONENT);
  const [position, setPosition] = useState(DEFAULT_POSITION);
  const [method, setMethod] = useState(DEFAULT_METHOD);
  const [colorBy, setColorBy] = useState(DEFAULT_COLOR_BY);
  const [showNoHorizon, setShowNoHorizon] = useState(true);

  // Selection state
  const [selectedSampleIdx, setSelectedSampleIdx] = useState<number | null>(null);

  // Fetch config
  const { data: config, isLoading: configLoading } = useConfig();

  // Fetch embedding data
  const {
    data: embedding,
    isLoading: embeddingLoading,
    error: embeddingError,
  } = useEmbedding(layer, component, position, method);

  // Fetch metadata for coloring
  const { data: metadata, isLoading: metadataLoading } = useMetadata(colorBy);

  // Fetch has_horizon metadata for filtering
  const { data: hasHorizonMeta } = useMetadata('has_horizon');

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

  // Fields that should always use gradient coloring
  const GRADIENT_FIELDS = ['log_time_horizon', 'long_term_delay', 'sample_idx'];

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
      // Special handling for time_horizon: use gray for no-horizon (value 0)
      if (colorBy === 'time_horizon') {
        return timeHorizonColors(filteredValues);
      }
      return categoricalColors(filteredValues, uniqueValues.size);
    } else {
      // For log_time_horizon, use special coloring with gray for no-horizon (value 0)
      if (colorBy === 'log_time_horizon') {
        return logTimeHorizonColors(filteredValues, metadata.min, metadata.max);
      }
      return valuesToColors(filteredValues, metadata.min, metadata.max, 'plasma');
    }
  }, [metadata, filterMask, positions.length, colorBy]);

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

  // Human-readable labels for color options
  const COLOR_LABELS: Record<string, string> = {
    'time_horizon': 'Time Horizon',
    'log_time_horizon': 'Time Horizon (Gradient)',
    'long_term_delay': 'Long-term Delay',
    'has_horizon': 'Has Horizon',
    'short_term_first': 'Order',
    'context_id': 'Context',
    'formatting_id': 'Formatting',
    'sample_idx': 'Sample Index',
  };

  // Generate legend data
  const legendData = useMemo(() => {
    if (!metadata?.values || metadata.values.length === 0) {
      return null;
    }

    const uniqueValues = new Set(metadata.values);
    const forceGradient = GRADIENT_FIELDS.includes(colorBy);
    const isCategorical = !forceGradient && uniqueValues.size <= 10;

    if (isCategorical) {
      // Time horizon uses special plasma-inspired palette with gray for no-horizon
      const timeHorizonPalette: Record<number, string> = {
        0: '#666666',  // Gray - no horizon
        1: '#0d0887',  // Deep purple - 1 month
        2: '#6a00a8',  // Purple - 2 months
        3: '#b12a90',  // Magenta - 3 months
        4: '#e16462',  // Salmon - 4 months
        5: '#f38d27',  // Orange - 5 months
        8: '#fccf25',  // Yellow - 8 months
        10: '#f0fa21', // Bright yellow - 10 months
      };

      // Default categorical palette
      const defaultPalette = [
        '#c778de', // Primary purple
        '#ff6b9e', // Primary pink
        '#57b5c2', // Primary cyan
        '#faba02', // Yellow
        '#4db04f', // Green
        '#f08080', // Light red
        '#9696cc', // Light purple
        '#ffa666', // Orange
      ];

      const items: LegendItem[] = [];
      const sortedValues = [...uniqueValues].sort((a, b) => Number(a) - Number(b));

      sortedValues.forEach((value, idx) => {
        let label: string;
        let color: string;

        if (colorBy === 'has_horizon') {
          label = value === 1 ? 'Has horizon' : 'No horizon';
          color = defaultPalette[idx % defaultPalette.length];
        } else if (colorBy === 'short_term_first') {
          label = value === 1 ? 'Short-term first' : 'Long-term first';
          color = defaultPalette[idx % defaultPalette.length];
        } else if (colorBy === 'time_horizon') {
          label = value === 0 ? 'No horizon' : `${value} month${Number(value) !== 1 ? 's' : ''}`;
          color = timeHorizonPalette[Number(value)] || '#888888';
        } else {
          label = String(value);
          color = defaultPalette[idx % defaultPalette.length];
        }
        items.push({ label, color });
      });

      return { type: 'categorical' as const, items };
    } else {
      // Gradient legend for continuous values
      let minLabel = String(Math.round(metadata.min * 10) / 10);
      let maxLabel = String(Math.round(metadata.max * 10) / 10);

      // Better labels for specific fields
      if (colorBy === 'log_time_horizon') {
        // Convert log10 values back to months: 10^min - 1 and 10^max - 1
        // min ~0.3 -> 10^0.3 - 1 = 1 month, max ~1.04 -> 10^1.04 - 1 = 10 months
        // But we also have no-horizon at 0 -> show as "No horizon" to actual range
        const actualMinMonths = Math.round(Math.pow(10, metadata.min) - 1);
        const actualMaxMonths = Math.round(Math.pow(10, metadata.max) - 1);
        minLabel = actualMinMonths <= 0 ? 'No horizon' : `${actualMinMonths} mo`;
        maxLabel = `${actualMaxMonths} mo`;
      } else if (colorBy === 'long_term_delay') {
        minLabel = `${Math.round(metadata.min)} days`;
        maxLabel = `${Math.round(metadata.max)} days`;
      }

      return {
        type: 'gradient' as const,
        gradient: {
          minLabel,
          maxLabel,
          colors: ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921'], // Plasma
        },
      };
    }
  }, [metadata, colorBy]);

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
  const layers = config?.layers || [DEFAULT_LAYER];
  const components = config?.components || [DEFAULT_COMPONENT];
  const positions_options = config?.positions || [DEFAULT_POSITION];
  const methods = config?.methods || [DEFAULT_METHOD];
  const colorByOptions = config?.colorByOptions || [DEFAULT_COLOR_BY];

  return (
    <div className={`min-h-screen bg-gradient-main ${isDarkMode ? 'dark' : ''}`}>
      {/* Header */}
      <Header
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
            onPositionChange={setPosition}
            method={method}
            methods={methods}
            onMethodChange={setMethod}
            colorBy={colorBy}
            colorByOptions={colorByOptions}
            onColorByChange={setColorBy}
            showNoHorizon={showNoHorizon}
            onShowNoHorizonChange={setShowNoHorizon}
          />
        </aside>

        {/* Main Visualization Area */}
        <main className="flex-1 p-4 flex flex-col min-w-0">
          {/* Error State */}
          {embeddingError && (
            <div className="mb-4 p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-700">
              <strong>Error loading data:</strong>{' '}
              {(embeddingError as Error).message || 'Unknown error'}
            </div>
          )}

          {/* 3D Visualization */}
          <div className="flex-1 relative rounded-2xl overflow-hidden shadow-2xl shadow-purple-500/10 border border-white/60">
            {isLoading && !positions.length ? (
              <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-[#fef6f9] to-[#f8f4fc]">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-[#C678DD] to-[#FF6B9D] animate-pulse flex items-center justify-center">
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
                  <p className="text-[#4a3f5c]/60 font-medium">
                    Loading embedding data...
                  </p>
                </div>
              </div>
            ) : (
              <ScatterPlot3D
                positions={positions}
                colors={colors}
                pointData={pointData}
                pointSize={3}
                showAxes={true}
                showGrid={true}
                onPointSelect={handlePointSelect}
                backgroundColor="#fef6f9"
                initialCameraPosition={[8, 6, 8]}
                className="w-full h-full"
              />
            )}

            {/* Loading overlay when updating */}
            {isLoading && positions.length > 0 && (
              <div className="absolute inset-0 bg-white/50 backdrop-blur-sm flex items-center justify-center pointer-events-none">
                <div className="px-4 py-2 bg-white/90 rounded-full shadow-lg border border-purple-100/50 flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full border-2 border-[#C678DD] border-t-transparent animate-spin" />
                  <span className="text-sm text-[#4a3f5c]/70">Updating...</span>
                </div>
              </div>
            )}

            {/* Legend */}
            {legendData && (
              <Legend
                title={colorByLabel}
                items={legendData.type === 'categorical' ? legendData.items : undefined}
                gradient={legendData.type === 'gradient' ? legendData.gradient : undefined}
              />
            )}
          </div>
        </main>

        {/* Right Sidebar - Info Panel */}
        <aside className="w-80 flex-shrink-0 p-4 overflow-y-auto border-l border-white/40 bg-white/30 backdrop-blur-sm">
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
            metrics={embedding?.metrics || null}
            isLoading={sampleLoading}
            layer={layer}
            component={component}
            method={method}
            onClose={handleClearSelection}
          />
        </aside>
      </div>
    </div>
  );
}

export default App;
