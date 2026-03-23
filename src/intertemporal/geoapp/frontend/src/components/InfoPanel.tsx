import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from './ui/Card';
import { Button } from './ui/Button';
import { Badge, BadgeGroup } from './ui/Badge';

interface SampleInfo {
  idx: number;
  text: string;
  timeHorizon: number | null;
  timeScale: string | null;
  choiceType: string | null;
  shortTermFirst: boolean | null;
  label?: string;
}

interface Metrics {
  varianceExplained?: number[];
  r2Score?: number;
}

interface InfoPanelProps {
  selectedSample: SampleInfo | null;
  metrics: Metrics | null;
  isLoading?: boolean;
  layer?: number;
  component?: string;
  method?: string;
  onClose?: () => void;
  className?: string;
}

const CopyIcon = () => (
  <svg
    className="w-4 h-4"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
    />
  </svg>
);

const CheckIcon = () => (
  <svg
    className="w-4 h-4"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M5 13l4 4L19 7"
    />
  </svg>
);

const CloseIcon = () => (
  <svg
    className="w-4 h-4"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M6 18L18 6M6 6l12 12"
    />
  </svg>
);

export const InfoPanel: React.FC<InfoPanelProps> = ({
  selectedSample,
  metrics,
  isLoading = false,
  layer,
  component,
  method,
  onClose,
  className = '',
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (selectedSample?.text) {
      await navigator.clipboard.writeText(selectedSample.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className={`flex flex-col gap-4 ${className}`}>
      {/* Metrics Card */}
      <Card padding="sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Metrics</CardTitle>
        </CardHeader>
        <CardContent className="py-2">
          {metrics ? (
            <div className="space-y-3">
              {metrics.r2Score !== undefined && (
                <div className="flex justify-between items-center">
                  <span className="text-sm text-[#4a3f5c]/70">R2 Score</span>
                  <Badge variant="primary" size="sm">
                    {(metrics.r2Score * 100).toFixed(1)}%
                  </Badge>
                </div>
              )}
              {metrics.varianceExplained && metrics.varianceExplained.length > 0 && (
                <div className="space-y-2">
                  <span className="text-sm text-[#4a3f5c]/70 block">
                    Variance Explained
                  </span>
                  <div className="space-y-1.5">
                    {metrics.varianceExplained.slice(0, 3).map((v, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <span className="text-xs text-[#4a3f5c]/50 w-12">
                          PC{i + 1}
                        </span>
                        <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-[#C678DD] to-[#FF6B9D] rounded-full transition-all duration-500"
                            style={{ width: `${v * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-[#4a3f5c] font-medium w-12 text-right">
                          {(v * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                  <div className="text-xs text-[#4a3f5c]/50 mt-2">
                    Total:{' '}
                    {(
                      metrics.varianceExplained.slice(0, 3).reduce((a, b) => a + b, 0) * 100
                    ).toFixed(1)}
                    %
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-[#4a3f5c]/50 text-center py-4">
              {isLoading ? 'Loading metrics...' : 'No metrics available'}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Current View Info */}
      {(layer !== undefined || component || method) && (
        <Card padding="sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Current View</CardTitle>
          </CardHeader>
          <CardContent className="py-2">
            <BadgeGroup>
              {layer !== undefined && (
                <Badge variant="secondary" size="sm">
                  Layer {layer}
                </Badge>
              )}
              {component && (
                <Badge variant="info" size="sm">
                  {component}
                </Badge>
              )}
              {method && (
                <Badge variant="primary" size="sm">
                  {method}
                </Badge>
              )}
            </BadgeGroup>
          </CardContent>
        </Card>
      )}

      {/* Selected Sample Card */}
      <Card padding="sm" className="flex-1">
        <CardHeader className="pb-3 flex items-start justify-between">
          <CardTitle className="text-sm">
            {selectedSample ? `Sample #${selectedSample.idx}` : 'Selected Sample'}
          </CardTitle>
          {onClose && selectedSample && (
            <Button
              variant="ghost"
              size="sm"
              icon={<CloseIcon />}
              onClick={onClose}
              className="!p-1"
            />
          )}
        </CardHeader>
        <CardContent className="py-2">
          {isLoading ? (
            <div className="space-y-3 animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-3/4" />
              <div className="h-4 bg-gray-200 rounded w-1/2" />
              <div className="h-20 bg-gray-200 rounded" />
            </div>
          ) : selectedSample ? (
            <div className="space-y-3">
              {/* Sample attributes */}
              <BadgeGroup>
                {selectedSample.choiceType && (
                  <Badge
                    variant={selectedSample.choiceType === 'short_term' ? 'warning' : 'success'}
                    size="sm"
                    dot
                  >
                    {selectedSample.choiceType === 'short_term'
                      ? 'Short-term'
                      : 'Long-term'}
                  </Badge>
                )}
                {selectedSample.timeHorizon !== null && (
                  <Badge variant="info" size="sm">
                    {selectedSample.timeHorizon} {selectedSample.timeScale}
                  </Badge>
                )}
                {selectedSample.shortTermFirst === true && (
                  <Badge variant="secondary" size="sm">
                    ST First
                  </Badge>
                )}
              </BadgeGroup>

              {/* Sample text */}
              <div className="relative group">
                <div className="p-3 bg-white/50 rounded-lg border border-purple-100/50 max-h-40 overflow-y-auto">
                  <p className="text-sm text-[#4a3f5c] whitespace-pre-wrap leading-relaxed">
                    {selectedSample.text}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  icon={copied ? <CheckIcon /> : <CopyIcon />}
                  onClick={handleCopy}
                  className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                />
              </div>

              {/* Label if available */}
              {selectedSample.label && (
                <div className="text-xs text-[#4a3f5c]/50">
                  Label: {selectedSample.label}
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-gradient-to-br from-[#C678DD]/20 to-[#FF6B9D]/20 flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-[#C678DD]"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"
                  />
                </svg>
              </div>
              <p className="text-sm text-[#4a3f5c]/50">
                Click on a point to see details
              </p>
            </div>
          )}
        </CardContent>
        {selectedSample && (
          <CardFooter className="pt-2">
            <Button variant="ghost" size="sm" onClick={handleCopy}>
              {copied ? 'Copied!' : 'Copy Text'}
            </Button>
          </CardFooter>
        )}
      </Card>
    </div>
  );
};

export default InfoPanel;
