import React from 'react';

export interface LegendItem {
  label: string;
  color: string;
}

export interface LegendProps {
  title: string;
  items?: LegendItem[];
  gradient?: {
    minLabel: string;
    maxLabel: string;
    colors: string[];
  };
  className?: string;
}

export const Legend: React.FC<LegendProps> = ({
  title,
  items,
  gradient,
  className = '',
}) => {
  return (
    <div
      className={`absolute bottom-4 right-4 bg-white/90 backdrop-blur-md rounded-xl shadow-lg border border-white/60 p-3 min-w-[140px] ${className}`}
    >
      <h4 className="text-xs font-semibold text-[#4a3f5c] mb-2 uppercase tracking-wide">
        {title}
      </h4>

      {/* Categorical legend */}
      {items && items.length > 0 && (
        <div className="space-y-1.5">
          {items.map((item, idx) => (
            <div key={idx} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full flex-shrink-0"
                style={{ backgroundColor: item.color }}
              />
              <span className="text-xs text-[#4a3f5c]/80 truncate">
                {item.label}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Gradient legend */}
      {gradient && (
        <div className="space-y-1">
          <div
            className="h-3 rounded-full"
            style={{
              background: `linear-gradient(to right, ${gradient.colors.join(', ')})`,
            }}
          />
          <div className="flex justify-between text-xs text-[#4a3f5c]/70">
            <span>{gradient.minLabel}</span>
            <span>{gradient.maxLabel}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default Legend;
