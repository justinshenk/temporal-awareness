import { useState, useMemo, useCallback } from 'react';
import { Header, ControlPanel, InfoPanel, ScatterPlot3D } from './components';
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

  // Compute positions Float32Array
  const positions = useMemo(() => {
    if (!embedding?.positions) {
      return new Float32Array(0);
    }
    return toFloat32Array(embedding.positions);
  }, [embedding?.positions]);

  // Compute colors Float32Array
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

    // Check if categorical or continuous
    const uniqueValues = new Set(metadata.values);
    const isCategorical = uniqueValues.size <= 10;

    if (isCategorical) {
      return categoricalColors(metadata.values, uniqueValues.size);
    } else {
      return valuesToColors(metadata.values, metadata.min, metadata.max, 'viridis');
    }
  }, [metadata, positions.length]);

  // Create point data array
  const pointData = useMemo<PointData[]>(() => {
    if (!embedding?.indices) return [];
    return embedding.indices.map((idx) => ({
      sampleIdx: idx,
      // Additional data can be added from metadata
    }));
  }, [embedding?.indices]);

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
  const maxLayer = config?.layers ? Math.max(...config.layers) : 35;
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
            maxLayer={maxLayer}
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
