import React, { useState } from 'react';
import { Card, CardHeader, CardContent } from './ui/Card';

interface FilterRange {
  min: number | null;
  max: number | null;
}

interface FilterPanelProps {
  // Short-term filters
  shortRewardFilter: FilterRange;
  shortTimeFilter: FilterRange;
  // Long-term filters
  longRewardFilter: FilterRange;
  longTimeFilter: FilterRange;
  // Callbacks
  onShortRewardFilterChange: (filter: FilterRange) => void;
  onShortTimeFilterChange: (filter: FilterRange) => void;
  onLongRewardFilterChange: (filter: FilterRange) => void;
  onLongTimeFilterChange: (filter: FilterRange) => void;
  className?: string;
}

function RangeInput({
  value,
  onChange,
  label,
}: {
  value: FilterRange;
  onChange: (range: FilterRange) => void;
  label: string;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[8px] text-gray-400 dark:text-gray-500">{label}</span>
      <div className="flex items-center gap-0.5">
        <input
          type="number"
          placeholder="min"
          value={value.min ?? ''}
          onChange={(e) => onChange({ ...value, min: e.target.value === '' ? null : parseFloat(e.target.value) })}
          className="w-12 px-1 py-0.5 text-[9px] border border-gray-200 dark:border-[#3a3633] rounded bg-white dark:bg-[#2a2623] text-[#1a1613] dark:text-white text-center"
        />
        <span className="text-[8px] text-gray-300">–</span>
        <input
          type="number"
          placeholder="max"
          value={value.max ?? ''}
          onChange={(e) => onChange({ ...value, max: e.target.value === '' ? null : parseFloat(e.target.value) })}
          className="w-12 px-1 py-0.5 text-[9px] border border-gray-200 dark:border-[#3a3633] rounded bg-white dark:bg-[#2a2623] text-[#1a1613] dark:text-white text-center"
        />
      </div>
    </div>
  );
}

export const FilterPanel: React.FC<FilterPanelProps> = ({
  shortRewardFilter,
  shortTimeFilter,
  longRewardFilter,
  longTimeFilter,
  onShortRewardFilterChange,
  onShortTimeFilterChange,
  onLongRewardFilterChange,
  onLongTimeFilterChange,
  className = '',
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const hasFilters =
    shortRewardFilter.min !== null || shortRewardFilter.max !== null ||
    shortTimeFilter.min !== null || shortTimeFilter.max !== null ||
    longRewardFilter.min !== null || longRewardFilter.max !== null ||
    longTimeFilter.min !== null || longTimeFilter.max !== null;

  const clearAll = () => {
    onShortRewardFilterChange({ min: null, max: null });
    onShortTimeFilterChange({ min: null, max: null });
    onLongRewardFilterChange({ min: null, max: null });
    onLongTimeFilterChange({ min: null, max: null });
  };

  return (
    <Card padding="sm" className={className}>
      <CardHeader className="pb-1">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1 text-sm font-semibold text-[#1a1613] dark:text-white hover:text-[#D97757]"
          >
            <svg
              className={`w-3 h-3 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            Filters
            {hasFilters && <span className="w-1.5 h-1.5 rounded-full bg-[#D97757]" />}
          </button>
          {hasFilters && isExpanded && (
            <button
              onClick={clearAll}
              className="text-[9px] px-1.5 py-0.5 rounded bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 hover:bg-red-200"
            >
              Clear
            </button>
          )}
        </div>
      </CardHeader>

      {isExpanded && (
        <CardContent className="pt-2 pb-1">
          <div className="space-y-3">
            {/* Short-term section */}
            <div>
              <div className="text-[10px] font-semibold text-[#7CB342] mb-1.5">Short-term Option</div>
              <div className="flex gap-3">
                <RangeInput label="Reward ($)" value={shortRewardFilter} onChange={onShortRewardFilterChange} />
                <RangeInput label="Time (mo)" value={shortTimeFilter} onChange={onShortTimeFilterChange} />
              </div>
            </div>

            {/* Long-term section */}
            <div>
              <div className="text-[10px] font-semibold text-[#00ACC1] mb-1.5">Long-term Option</div>
              <div className="flex gap-3">
                <RangeInput label="Reward ($)" value={longRewardFilter} onChange={onLongRewardFilterChange} />
                <RangeInput label="Time (mo)" value={longTimeFilter} onChange={onLongTimeFilterChange} />
              </div>
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  );
};

export default FilterPanel;
