import React, { useState, useMemo, useEffect } from 'react';
import { Select } from './ui/Select';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Tabs } from './ui/Tabs';
import { PositionSelector } from './PositionSelector';

// Time scale transfer function types
type TimeScaleType = 'linear' | 'log' | 'adaptive' | 'blend';

// Time units for range input
type TimeUnit = 'seconds' | 'minutes' | 'hours' | 'days' | 'weeks' | 'months' | 'years' | 'decades' | 'centuries' | 'millennia';

// Using 30 days/month as base for consistency
const TIME_UNIT_TO_MONTHS: Record<TimeUnit, number> = {
  seconds: 1 / (30 * 24 * 60 * 60),  // 1 second in months
  minutes: 1 / (30 * 24 * 60),        // 1 minute in months
  hours: 1 / (30 * 24),               // 1 hour in months
  days: 1 / 30,                       // 1 day in months
  weeks: 7 / 30,                      // 7 days in months
  months: 1,
  years: 12,
  decades: 120,
  centuries: 1200,
  millennia: 12000,
};

const TIME_UNITS: TimeUnit[] = ['seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years', 'decades', 'centuries', 'millennia'];

// Convert months to best display unit
function monthsToBestUnit(months: number): { value: number; unit: TimeUnit } {
  if (months >= 12000) return { value: months / 12000, unit: 'millennia' };
  if (months >= 1200) return { value: months / 1200, unit: 'centuries' };
  if (months >= 120) return { value: months / 120, unit: 'decades' };
  if (months >= 12) return { value: months / 12, unit: 'years' };
  if (months >= 1) return { value: months, unit: 'months' };
  if (months >= 7/30) return { value: months * 30 / 7, unit: 'weeks' };
  if (months >= 1/30) return { value: months * 30, unit: 'days' };
  if (months >= 1/(30*24)) return { value: months * 30 * 24, unit: 'hours' };
  if (months >= 1/(30*24*60)) return { value: months * 30 * 24 * 60, unit: 'minutes' };
  return { value: months * 30 * 24 * 60 * 60, unit: 'seconds' };
}

interface ControlPanelProps {
  // Layer controls
  layer: number;
  layers: number[];
  onLayerChange: (layer: number) => void;
  hideLayerSection?: boolean;

  // Component controls
  component: string;
  components: string[];
  onComponentChange: (component: string) => void;

  // Position controls
  position: string;
  positions: string[];
  positionLabels?: Record<string, string>;
  promptTemplate?: Array<{
    name: string;
    label: string;
    type: 'marker' | 'variable' | 'static' | 'semantic';
    available: boolean;
  }>;
  onPositionChange: (position: string) => void;
  hidePositionSection?: boolean;

  // Method controls
  method: string;
  methods: string[];
  onMethodChange: (method: string) => void;
  hideMethodSection?: boolean;

  // Color controls
  colorBy: string;
  colorByOptions: string[];
  onColorByChange: (colorBy: string) => void;

  // Color range controls (for gradient coloring)
  colorRangeMin: number | null;
  colorRangeMax: number | null;
  colorRangeDataMin?: number;
  colorRangeDataMax?: number;
  onColorRangeMinChange: (val: number | null) => void;
  onColorRangeMaxChange: (val: number | null) => void;
  showColorRangeControls?: boolean;

  // Time scale controls
  timeScaleType?: TimeScaleType;
  onTimeScaleTypeChange?: (scaleType: TimeScaleType) => void;
  blendMix?: number;
  onBlendMixChange?: (mix: number) => void;
  showTimeScaleControls?: boolean;

  // Hide color by section (when moved to right sidebar)
  hideColorBySection?: boolean;

  className?: string;
}

// Time Range Controls subcomponent with unit selection
interface TimeRangeControlsProps {
  colorRangeMin: number | null;
  colorRangeMax: number | null;
  colorRangeDataMin?: number;
  colorRangeDataMax?: number;
  onColorRangeMinChange: (val: number | null) => void;
  onColorRangeMaxChange: (val: number | null) => void;
}

const TimeRangeControls: React.FC<TimeRangeControlsProps> = ({
  colorRangeMin,
  colorRangeMax,
  colorRangeDataMin,
  colorRangeDataMax,
  onColorRangeMinChange,
  onColorRangeMaxChange,
}) => {
  // Determine best unit for data range display
  // Use geometric mean to choose a unit that works well for both min and max
  const dataRangeDisplay = useMemo(() => {
    if (colorRangeDataMin === undefined || colorRangeDataMax === undefined) {
      return { minDisplay: '?', maxDisplay: '?', unit: 'months' as TimeUnit };
    }
    // Handle edge cases: zero, negative, or equal values
    const safeMin = Math.max(colorRangeDataMin, 0.0001);
    const safeMax = Math.max(colorRangeDataMax, 0.0001);
    // Use geometric mean to find a balanced unit
    const geoMean = Math.sqrt(safeMin * safeMax);
    const unitInfo = monthsToBestUnit(geoMean);
    const conversionFactor = TIME_UNIT_TO_MONTHS[unitInfo.unit];

    return {
      minDisplay: (colorRangeDataMin / conversionFactor).toFixed(2),
      maxDisplay: (colorRangeDataMax / conversionFactor).toFixed(2),
      unit: unitInfo.unit,
    };
  }, [colorRangeDataMin, colorRangeDataMax]);

  // Local state for input units (default to best unit for data)
  const [minUnit, setMinUnit] = useState<TimeUnit>('months');
  const [maxUnit, setMaxUnit] = useState<TimeUnit>('months');

  // Sync units when data range changes (e.g., on initial load)
  useEffect(() => {
    if (dataRangeDisplay.unit !== 'months' || colorRangeDataMin !== undefined) {
      setMinUnit(dataRangeDisplay.unit);
      setMaxUnit(dataRangeDisplay.unit);
    }
  }, [dataRangeDisplay.unit, colorRangeDataMin]);

  // Convert stored months value to display value in current unit
  const minDisplayValue = useMemo(() => {
    if (colorRangeMin === null) return '';
    return (colorRangeMin / TIME_UNIT_TO_MONTHS[minUnit]).toFixed(2);
  }, [colorRangeMin, minUnit]);

  const maxDisplayValue = useMemo(() => {
    if (colorRangeMax === null) return '';
    return (colorRangeMax / TIME_UNIT_TO_MONTHS[maxUnit]).toFixed(2);
  }, [colorRangeMax, maxUnit]);

  // Handle input changes - convert from display unit to months
  // Validates that values are non-negative (time can't be negative)
  const handleMinChange = (inputValue: string) => {
    if (inputValue === '') {
      onColorRangeMinChange(null);
    } else {
      const numValue = parseFloat(inputValue);
      if (!isNaN(numValue) && numValue >= 0) {
        onColorRangeMinChange(numValue * TIME_UNIT_TO_MONTHS[minUnit]);
      }
    }
  };

  const handleMaxChange = (inputValue: string) => {
    if (inputValue === '') {
      onColorRangeMaxChange(null);
    } else {
      const numValue = parseFloat(inputValue);
      if (!isNaN(numValue) && numValue >= 0) {
        onColorRangeMaxChange(numValue * TIME_UNIT_TO_MONTHS[maxUnit]);
      }
    }
  };

  // Unit selector dropdown - abbreviated labels for compact display
  const unitAbbreviations: Record<TimeUnit, string> = {
    seconds: 'sec',
    minutes: 'min',
    hours: 'hr',
    days: 'd',
    weeks: 'wk',
    months: 'mo',
    years: 'yr',
    decades: 'dec',
    centuries: 'cent',
    millennia: 'mill',
  };

  const UnitSelect: React.FC<{ value: TimeUnit; onChange: (u: TimeUnit) => void; label: string }> = ({ value, onChange, label }) => (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as TimeUnit)}
      aria-label={label}
      className="w-16 shrink-0 px-1 py-1.5 text-xs bg-white/70 border border-[#f5f0eb] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#D97757]/50 cursor-pointer"
    >
      {TIME_UNITS.map((unit) => (
        <option key={unit} value={unit}>
          {unitAbbreviations[unit]}
        </option>
      ))}
    </select>
  );

  return (
    <Card padding="sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">Time Range</CardTitle>
      </CardHeader>
      <CardContent className="py-2 space-y-3">
        {/* Min input with unit */}
        <div className="space-y-1">
          <label htmlFor="time-range-min" className="text-xs text-[#1a1613]/70 block">Min:</label>
          <div className="flex items-center gap-2">
            <input
              id="time-range-min"
              type="number"
              value={minDisplayValue}
              onChange={(e) => handleMinChange(e.target.value)}
              placeholder="auto"
              step="any"
              min="0"
              className="w-full min-w-0 flex-1 px-2 py-1.5 text-sm bg-white/50 border border-[#f5f0eb] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#D97757]/50"
            />
            <UnitSelect value={minUnit} onChange={setMinUnit} label="Minimum time unit" />
          </div>
        </div>

        {/* Max input with unit */}
        <div className="space-y-1">
          <label htmlFor="time-range-max" className="text-xs text-[#1a1613]/70 block">Max:</label>
          <div className="flex items-center gap-2">
            <input
              id="time-range-max"
              type="number"
              value={maxDisplayValue}
              onChange={(e) => handleMaxChange(e.target.value)}
              placeholder="auto"
              step="any"
              min="0"
              className="w-full min-w-0 flex-1 px-2 py-1.5 text-sm bg-white/50 border border-[#f5f0eb] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#D97757]/50"
            />
            <UnitSelect value={maxUnit} onChange={setMaxUnit} label="Maximum time unit" />
          </div>
        </div>

        {/* Warning when min > max */}
        {colorRangeMin !== null && colorRangeMax !== null && colorRangeMin > colorRangeMax && (
          <div className="text-[10px] text-amber-600 bg-amber-50 border border-amber-200 rounded px-2 py-1">
            Warning: Min is greater than Max. Colors may appear inverted.
          </div>
        )}

        {/* Reset button */}
        <button
          onClick={() => {
            onColorRangeMinChange(null);
            onColorRangeMaxChange(null);
          }}
          className="w-full px-2 py-1.5 text-xs text-[#1a1613]/70 bg-white/30 border border-[#f5f0eb]/50 rounded-lg hover:bg-white/50 focus:outline-none focus:ring-2 focus:ring-[#D97757]/50 transition-colors"
        >
          Reset to Auto
        </button>

        {/* Data range display */}
        <div className="text-[10px] text-[#1a1613]/50">
          Data: {dataRangeDisplay.minDisplay} - {dataRangeDisplay.maxDisplay} {dataRangeDisplay.unit}
        </div>
      </CardContent>
    </Card>
  );
};

const componentLabels: Record<string, string> = {
  resid_pre: 'Residual Pre',
  attn_out: 'Attention Out',
  mlp_out: 'MLP Out',
  resid_post: 'Residual Post',
};

const methodLabels: Record<string, string> = {
  pca: 'PCA',
  umap: 'UMAP',
  tsne: 't-SNE',
  PCA: 'PCA',
  UMAP: 'UMAP',
  'T-SNE': 't-SNE',
  TSNE: 't-SNE',
};

export const ControlPanel: React.FC<ControlPanelProps> = ({
  layer,
  layers,
  onLayerChange,
  hideLayerSection = false,
  component,
  components,
  onComponentChange,
  position,
  positions,
  positionLabels = {},
  promptTemplate = [],
  onPositionChange,
  hidePositionSection = false,
  method,
  methods,
  onMethodChange,
  hideMethodSection = false,
  colorBy,
  colorByOptions,
  onColorByChange,
  colorRangeMin,
  colorRangeMax,
  colorRangeDataMin,
  colorRangeDataMax,
  onColorRangeMinChange,
  onColorRangeMaxChange,
  showColorRangeControls = false,
  timeScaleType = 'linear',
  onTimeScaleTypeChange,
  blendMix = 0.5,
  onBlendMixChange,
  showTimeScaleControls = false,
  hideColorBySection = false,
  className = '',
}) => {
  // Convert components to tabs format
  const componentTabs = components.map((c) => ({
    id: c,
    label: componentLabels[c] || c,
  }));

  // Convert methods to tabs format
  const methodTabs = methods.map((m) => ({
    id: m,
    label: methodLabels[m] || m,
  }));

  // Convert positions to select options using labels from backend (DefaultPromptFormat)
  const positionOptions = positions.map((p) => {
    // Use label from backend if available
    if (positionLabels[p]) {
      return { value: p, label: positionLabels[p] };
    }
    // Numeric positions
    if (/^\d+$/.test(p)) {
      return { value: p, label: `Token #${p}` };
    }
    // Fallback: format the key as label
    return {
      value: p,
      label: p.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
    };
  });

  // Human-readable labels for color options
  const colorByLabels: Record<string, string> = {
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
  };

  // Convert colorBy to select options
  const colorBySelectOptions = colorByOptions.map((c) => ({
    value: c,
    label: colorByLabels[c] || c.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
  }));

  // Convert layers to select options
  const layerOptions = layers.map((l) => ({
    value: l.toString(),
    label: `Layer ${l}`,
  }));

  return (
    <div className={`flex flex-col gap-4 ${className}`}>
      {/* Layer Control - hidden for 1DxLayer view */}
      {!hideLayerSection && (
        <Card padding="sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Layer</CardTitle>
          </CardHeader>
          <CardContent className="py-2">
            <Select
              options={layerOptions}
              value={layer.toString()}
              onChange={(val) => onLayerChange(parseInt(val, 10))}
              placeholder="Select layer..."
            />
          </CardContent>
        </Card>
      )}

      {/* Component Control */}
      <Card padding="sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Component</CardTitle>
        </CardHeader>
        <CardContent className="py-2">
          {componentTabs.length > 0 && (
            <div className="space-y-1.5">
              {componentTabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => onComponentChange(tab.id)}
                  aria-pressed={component === tab.id}
                  className={`
                    w-full px-3 py-2 text-left text-sm rounded-lg
                    transition-all duration-200
                    focus:outline-none focus:ring-2 focus:ring-[#D97757]/50
                    ${component === tab.id
                      ? 'bg-gradient-to-r from-[#D97757]/20 to-[#348296]/20 text-[#1a1613] font-medium border border-[#D97757]/30'
                      : 'bg-white/50 text-[#1a1613]/70 hover:bg-white/80 hover:text-[#1a1613] border border-transparent'
                    }
                  `}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Position Control - hidden when using floating position selector */}
      {!hidePositionSection && (
        <Card padding="sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Position</CardTitle>
          </CardHeader>
          <CardContent className="py-2">
            {promptTemplate.length > 0 ? (
              <PositionSelector
                position={position}
                promptTemplate={promptTemplate}
                positionLabels={positionLabels}
                onPositionChange={onPositionChange}
              />
            ) : (
              <Select
                options={positionOptions}
                value={position}
                onChange={onPositionChange}
                placeholder="Select position..."
                searchable={positions.length > 5}
              />
            )}
          </CardContent>
        </Card>
      )}

      {/* Method Control - hidden for 1D views (they only use PCA) */}
      {!hideMethodSection && (
        <Card padding="sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Reduction Method</CardTitle>
          </CardHeader>
          <CardContent className="py-2">
            {methodTabs.length > 0 && (
              <Tabs
                tabs={methodTabs}
                activeTab={method}
                onChange={onMethodChange}
              />
            )}
          </CardContent>
        </Card>
      )}

      {/* Color By Control */}
      {!hideColorBySection && (
        <Card padding="sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Color By</CardTitle>
          </CardHeader>
          <CardContent className="py-2">
            <Select
              options={colorBySelectOptions}
              value={colorBy}
              onChange={onColorByChange}
              placeholder="Select attribute..."
            />
          </CardContent>
        </Card>
      )}

      {/* Time Scale Controls (for time-related gradient fields) */}
      {showTimeScaleControls && (
        <Card padding="sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Time Scale Transfer</CardTitle>
          </CardHeader>
          <CardContent className="py-2 space-y-3">
            {/* Scale type selector */}
            <div className="grid grid-cols-2 gap-1">
              {(['linear', 'log', 'adaptive', 'blend'] as TimeScaleType[]).map((type) => (
                <button
                  key={type}
                  onClick={() => onTimeScaleTypeChange?.(type)}
                  aria-pressed={timeScaleType === type}
                  className={`
                    px-2 py-1.5 text-xs rounded-lg transition-all duration-200
                    focus:outline-none focus:ring-2 focus:ring-[#D97757]/50
                    ${timeScaleType === type
                      ? 'bg-gradient-to-r from-[#D97757]/20 to-[#348296]/20 text-[#1a1613] font-medium border border-[#D97757]/30'
                      : 'bg-white/50 text-[#1a1613]/70 hover:bg-white/80 border border-transparent'
                    }
                  `}
                >
                  {type === 'linear' ? 'Linear' :
                   type === 'log' ? 'Log' :
                   type === 'adaptive' ? 'Adaptive' : 'Blend'}
                </button>
              ))}
            </div>

            {/* Blend mix slider (only show when blend is selected) */}
            {timeScaleType === 'blend' && (
              <div className="space-y-1">
                <div className="flex justify-between text-[10px] text-[#1a1613]/50">
                  <span id="blend-min-label">Linear</span>
                  <span id="blend-max-label">Log</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={blendMix}
                  onChange={(e) => onBlendMixChange?.(parseFloat(e.target.value))}
                  aria-label="Linear to log blend mix"
                  aria-valuetext={`${(blendMix * 100).toFixed(0)}% log`}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gradient-to-r from-[#D97757]/30 to-[#348296]/30 focus:outline-none focus:ring-2 focus:ring-[#D97757]/50"
                />
                <div className="text-center text-[10px] text-[#1a1613]/50">
                  Mix: {(blendMix * 100).toFixed(0)}%
                </div>
              </div>
            )}

            {/* Description */}
            <div className="text-[10px] text-[#1a1613]/50 leading-relaxed">
              {timeScaleType === 'linear' && 'Linear scale - equal spacing for equal time differences'}
              {timeScaleType === 'log' && 'Log scale - equal spacing for equal ratios (10× gets same space)'}
              {timeScaleType === 'adaptive' && 'Scale-adaptive - equal visual space per tier (sec, min, hr, day, wk, mo, yr, dec, cent, mill)'}
              {timeScaleType === 'blend' && 'Blend between linear and log - adjust the mix slider'}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Color Range Controls (for gradient fields, only for linear/log/blend) */}
      {showColorRangeControls && timeScaleType !== 'adaptive' && (
        <TimeRangeControls
          colorRangeMin={colorRangeMin}
          colorRangeMax={colorRangeMax}
          colorRangeDataMin={colorRangeDataMin}
          colorRangeDataMax={colorRangeDataMax}
          onColorRangeMinChange={onColorRangeMinChange}
          onColorRangeMaxChange={onColorRangeMaxChange}
        />
      )}
    </div>
  );
};

export default ControlPanel;
