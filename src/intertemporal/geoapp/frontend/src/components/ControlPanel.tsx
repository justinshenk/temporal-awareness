import React from 'react';
import { Select } from './ui/Select';
import { Toggle } from './ui/Toggle';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Tabs } from './ui/Tabs';

interface ControlPanelProps {
  // Layer controls
  layer: number;
  layers: number[];
  onLayerChange: (layer: number) => void;

  // Component controls
  component: string;
  components: string[];
  onComponentChange: (component: string) => void;

  // Position controls
  position: string;
  positions: string[];
  onPositionChange: (position: string) => void;

  // Method controls
  method: string;
  methods: string[];
  onMethodChange: (method: string) => void;

  // Color controls
  colorBy: string;
  colorByOptions: string[];
  onColorByChange: (colorBy: string) => void;

  // Filters
  showNoHorizon: boolean;
  onShowNoHorizonChange: (show: boolean) => void;

  className?: string;
}

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
  component,
  components,
  onComponentChange,
  position,
  positions,
  onPositionChange,
  method,
  methods,
  onMethodChange,
  colorBy,
  colorByOptions,
  onColorByChange,
  showNoHorizon,
  onShowNoHorizonChange,
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

  // Convert positions to select options with better labels
  const positionOptions = positions.map((p) => {
    // Check if it's a numeric position like "P121"
    if (p.startsWith('P') && /^P\d+$/.test(p)) {
      return { value: p, label: `Token ${p.slice(1)}` };
    }
    // Named positions - format nicely
    return {
      value: p,
      label: p.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
    };
  });

  // Human-readable labels for color options
  const colorByLabels: Record<string, string> = {
    'time_horizon': 'Time Horizon',
    'log_time_horizon': 'Time Horizon (Gradient)',
    'long_term_delay': 'Long-term Delay',
    'has_horizon': 'Has Horizon',
    'short_term_first': 'Order (ST/LT First)',
    'context_id': 'Context',
    'formatting_id': 'Formatting',
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
      {/* Layer Control */}
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
                  className={`
                    w-full px-3 py-2 text-left text-sm rounded-lg
                    transition-all duration-200
                    ${component === tab.id
                      ? 'bg-gradient-to-r from-[#C678DD]/20 to-[#FF6B9D]/20 text-[#4a3f5c] font-medium border border-[#C678DD]/30'
                      : 'bg-white/50 text-[#4a3f5c]/70 hover:bg-white/80 hover:text-[#4a3f5c] border border-transparent'
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

      {/* Position Control */}
      <Card padding="sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Position</CardTitle>
        </CardHeader>
        <CardContent className="py-2">
          <Select
            options={positionOptions}
            value={position}
            onChange={onPositionChange}
            placeholder="Select position..."
            searchable={positions.length > 5}
          />
        </CardContent>
      </Card>

      {/* Method Control */}
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

      {/* Color By Control */}
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

      {/* Filters */}
      <Card padding="sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Filters</CardTitle>
        </CardHeader>
        <CardContent className="py-2">
          <Toggle
            checked={showNoHorizon}
            onChange={onShowNoHorizonChange}
            label="Show no-horizon samples"
            size="sm"
          />
        </CardContent>
      </Card>
    </div>
  );
};

export default ControlPanel;
