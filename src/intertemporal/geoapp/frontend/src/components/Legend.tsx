import React, { useState, useRef, useCallback, useEffect } from 'react';

export interface LegendItem {
  label: string;
  color: string;
}

export interface TierLabel {
  position: number;
  label: string;
}

export interface LegendProps {
  title: string;
  items?: LegendItem[];
  gradient?: {
    minLabel: string;
    maxLabel: string;
    colors: string[];
  };
  tiers?: TierLabel[];
  tierColors?: string[];
  className?: string;
  /** Controlled collapsed state */
  collapsed?: boolean;
  /** Callback when collapsed state changes */
  onCollapsedChange?: (collapsed: boolean) => void;
}

export const Legend: React.FC<LegendProps> = ({
  title,
  items,
  gradient,
  tiers,
  tierColors = ['#30123b', '#4777ef', '#1bd0d5', '#62fc6b', '#d2e935', '#fe9b2d', '#d23105'],
  className = '',
  collapsed,
  onCollapsedChange,
}) => {
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  const isCollapsed = collapsed !== undefined ? collapsed : internalCollapsed;
  const setIsCollapsed = (value: boolean) => {
    if (onCollapsedChange) {
      onCollapsedChange(value);
    } else {
      setInternalCollapsed(value);
    }
  };
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef<{ startX: number; startY: number; startPosX: number; startPosY: number } | null>(null);
  const legendRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button')) return;
    e.preventDefault();
    setIsDragging(true);
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startPosX: position.x,
      startPosY: position.y,
    };
  }, [position]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !dragRef.current) return;
      const dx = e.clientX - dragRef.current.startX;
      const dy = e.clientY - dragRef.current.startY;
      setPosition({
        x: dragRef.current.startPosX + dx,
        y: dragRef.current.startPosY + dy,
      });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      dragRef.current = null;
    };

    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);

  return (
    <div
      ref={legendRef}
      onMouseDown={handleMouseDown}
      style={{
        transform: `translate(${position.x}px, ${position.y}px)`,
        cursor: isDragging ? 'grabbing' : 'grab',
      }}
      className={`absolute ${isCollapsed ? 'bottom-16' : 'bottom-4'} right-4 bg-white/90 backdrop-blur-md rounded-xl shadow-lg border border-white/60 ${isCollapsed ? 'p-2' : 'p-3'} min-w-[40px] ${isCollapsed ? '' : 'max-w-[200px]'} select-none ${className}`}
    >
      <div className="flex items-center justify-between gap-2">
        <h4 className={`text-xs font-semibold text-[#1a1613] uppercase tracking-wide ${isCollapsed ? 'hidden' : ''}`}>
          {title}
        </h4>
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="w-5 h-5 flex items-center justify-center rounded hover:bg-[#1a1613]/10 transition-colors flex-shrink-0"
          title={isCollapsed ? 'Expand legend' : 'Collapse legend'}
        >
          <svg
            className={`w-3 h-3 text-[#1a1613]/60 transition-transform ${isCollapsed ? 'rotate-180' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={isCollapsed ? "M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" : "M19 9l-7 7-7-7"} />
          </svg>
        </button>
      </div>

      {!isCollapsed && (
        <div className="mt-2 space-y-2">
          {/* Categorical legend items */}
          {items && items.length > 0 && (
            <div className="space-y-1.5">
              {items.map((item, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full flex-shrink-0"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-xs text-[#1a1613]/80 truncate">
                    {item.label}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Separator */}
          {items && items.length > 0 && (gradient || tiers) && (
            <div className="border-t border-[#1a1613]/10" />
          )}

          {/* Adaptive tier legend */}
          {tiers && tiers.length > 0 && (
            <div className="space-y-1">
              <div
                className="h-3 rounded-full"
                style={{ background: `linear-gradient(to right, ${tierColors.join(', ')})` }}
              />
              <div className="flex flex-wrap gap-1 mt-1">
                {tiers
                  .filter((_, i) => i % 2 === 0 || i === tiers.length - 1)
                  .map((tier, idx) => (
                    <span
                      key={idx}
                      className="text-[9px] text-[#1a1613]/60 px-1 py-0.5 bg-[#1a1613]/5 rounded"
                      title={tier.label}
                    >
                      {tier.label.slice(0, 3)}
                    </span>
                  ))}
              </div>
            </div>
          )}

          {/* Standard gradient legend */}
          {gradient && !tiers && (
            <div className="space-y-1">
              <div
                className="h-3 rounded-full"
                style={{ background: `linear-gradient(to right, ${gradient.colors.join(', ')})` }}
              />
              <div className="flex justify-between text-xs text-[#1a1613]/70">
                <span>{gradient.minLabel}</span>
                <span>{gradient.maxLabel}</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Legend;
