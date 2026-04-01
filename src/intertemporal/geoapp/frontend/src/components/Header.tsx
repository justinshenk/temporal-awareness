import React from 'react';
import { Button } from './ui/Button';

type ViewMode = '2D' | '3D' | '1DxLayer' | '1DxPos' | 'Scree' | 'Align';

interface HeaderProps {
  datasetName?: string;
  modelName?: string;
  totalSamples?: number;
  totalLayers?: number;
  totalPositions?: number;
  isDarkMode?: boolean;
  onDarkModeChange?: (dark: boolean) => void;
  onExport?: () => void;
  isExporting?: boolean;
  viewMode?: ViewMode;
  onViewModeChange?: (mode: ViewMode) => void;
  className?: string;
}

const ExportIcon = () => (
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
      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
    />
  </svg>
);

const SunIcon = () => (
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
      d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
    />
  </svg>
);

const MoonIcon = () => (
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
      d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
    />
  </svg>
);

const GlobeIcon = () => (
  <svg
    className="w-6 h-6"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={1.5}
      d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
    />
  </svg>
);

export const Header: React.FC<HeaderProps> = ({
  datasetName = '',
  modelName = '',
  totalSamples = 0,
  totalLayers = 0,
  totalPositions = 0,
  isDarkMode = false,
  onDarkModeChange,
  onExport,
  isExporting = false,
  viewMode = '3D',
  onViewModeChange,
  className = '',
}) => {
  // Build title: "{datasetName} {modelName}" or just one if the other is missing
  const title = [datasetName, modelName].filter(Boolean).join(' ') || 'Geometry Explorer';

  return (
    <header
      className={`
        bg-white/80 backdrop-blur-md
        border-b border-white/60
        sticky top-0 z-50
        dark:bg-[#1a1613]/90 dark:border-[#1a1613]/60
        ${className}
      `}
    >
      <div className="px-6 py-3">
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center gap-3">
            <div className="relative">
              {/* Glow effect */}
              <div className="absolute inset-0 bg-[#D97757] rounded-xl blur-lg opacity-30" />
              {/* Logo container */}
              <div className="relative w-9 h-9 rounded-xl bg-[#D97757] flex items-center justify-center shadow-lg shadow-[#D97757]/25">
                <span className="text-white">
                  <GlobeIcon />
                </span>
              </div>
            </div>
            <div className="flex flex-col">
              <h1 className="text-lg font-bold text-[#1a1613] dark:text-white leading-tight">
                {title}
              </h1>
              {/* Stats - small text below title */}
              <div className="hidden md:flex items-center gap-2 text-xs text-[#1a1613]/60 dark:text-white/60">
                <span>{totalSamples.toLocaleString()} samples</span>
                <span>·</span>
                <span>{totalLayers} layers</span>
                <span>·</span>
                <span>{totalPositions} positions</span>
              </div>
            </div>
          </div>

          {/* View Mode Tabs - center/prominent position */}
          <div className="flex items-center gap-1 bg-white/95 dark:bg-[#2a2623] backdrop-blur-sm rounded-lg shadow-sm border border-white/60 dark:border-[#3a3633] p-1">
            {(['2D', '3D', '1DxLayer', '1DxPos', 'Scree', 'Align'] as ViewMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => onViewModeChange?.(mode)}
                className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${
                  viewMode === mode
                    ? 'bg-gradient-to-r from-[#D97757] to-[#348296] text-white shadow-sm'
                    : 'text-[#1a1613]/70 dark:text-white/70 hover:bg-gray-100 dark:hover:bg-[#3a3633]'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2">
            {/* Theme Toggle */}
            <button
              onClick={() => onDarkModeChange?.(!isDarkMode)}
              className="
                p-2 rounded-lg
                bg-white/50 border border-white/60
                text-[#1a1613]
                dark:bg-[#2a2623] dark:border-[#3a3633] dark:text-white
                hover:bg-white/80 hover:border-[#D97757]/30 hover:text-[#348296]
                dark:hover:bg-[#3a3633]
                transition-all duration-200
                focus:outline-none focus:ring-2 focus:ring-[#D97757]/50
              "
              title={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {isDarkMode ? <SunIcon /> : <MoonIcon />}
            </button>

            {/* Export Button */}
            <Button
              variant="secondary"
              size="sm"
              icon={<ExportIcon />}
              onClick={onExport}
              loading={isExporting}
            >
              <span className="hidden sm:inline">Export</span>
            </Button>
          </div>
        </div>

        {/* Mobile Stats */}
        <div className="flex md:hidden items-center gap-2 mt-2 text-xs text-[#1a1613]/60 dark:text-white/60">
          <span>{totalSamples.toLocaleString()} samples</span>
          <span>·</span>
          <span>{totalLayers} layers</span>
          <span>·</span>
          <span>{totalPositions} positions</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
