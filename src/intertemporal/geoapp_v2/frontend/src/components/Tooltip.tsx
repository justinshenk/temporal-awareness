import { useEffect, useRef, useState } from 'react';

export interface TooltipData {
  sampleIdx: number;
  position: { x: number; y: number; z: number };
  timeHorizon?: number;
  timeScale?: string;
  choiceType?: string;
  shortTermFirst?: boolean;
  [key: string]: unknown;
}

interface TooltipProps {
  data: TooltipData | null;
  mousePosition: { x: number; y: number };
  visible: boolean;
}

export function Tooltip({ data, mousePosition, visible }: TooltipProps) {
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [adjustedPosition, setAdjustedPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!tooltipRef.current || !visible) return;

    const tooltip = tooltipRef.current;
    const rect = tooltip.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    // Offset from cursor
    const offsetX = 16;
    const offsetY = 16;

    let x = mousePosition.x + offsetX;
    let y = mousePosition.y + offsetY;

    // Prevent tooltip from going off-screen
    if (x + rect.width > viewportWidth) {
      x = mousePosition.x - rect.width - offsetX;
    }
    if (y + rect.height > viewportHeight) {
      y = mousePosition.y - rect.height - offsetY;
    }

    // Clamp to viewport bounds
    x = Math.max(8, Math.min(x, viewportWidth - rect.width - 8));
    y = Math.max(8, Math.min(y, viewportHeight - rect.height - 8));

    setAdjustedPosition({ x, y });
  }, [mousePosition, visible]);

  if (!visible || !data) return null;

  const formatValue = (key: string, value: unknown): string => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'boolean') return value ? 'Yes' : 'No';
    if (typeof value === 'number') {
      if (key.includes('time') || key.includes('horizon')) {
        return `${value.toFixed(1)} months`;
      }
      return value.toFixed(3);
    }
    return String(value);
  };

  const displayKeys: Array<{ key: string; label: string }> = [
    { key: 'sampleIdx', label: 'Sample' },
    { key: 'timeHorizon', label: 'Time Horizon' },
    { key: 'timeScale', label: 'Time Scale' },
    { key: 'choiceType', label: 'Choice Type' },
    { key: 'shortTermFirst', label: 'Short-Term First' },
  ];

  return (
    <div
      ref={tooltipRef}
      className="tooltip-container"
      style={{
        position: 'fixed',
        left: adjustedPosition.x,
        top: adjustedPosition.y,
        zIndex: 1000,
        pointerEvents: 'none',
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(4px)',
        transition: 'opacity 150ms ease, transform 150ms ease',
      }}
    >
      <div
        style={{
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 244, 255, 0.95) 100%)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          border: '1px solid rgba(180, 160, 200, 0.3)',
          borderRadius: '12px',
          padding: '12px 16px',
          boxShadow: '0 8px 32px rgba(100, 80, 120, 0.15), 0 4px 12px rgba(0, 0, 0, 0.05)',
          minWidth: '180px',
          maxWidth: '280px',
        }}
      >
        <div
          style={{
            fontSize: '11px',
            fontWeight: 600,
            color: '#C678DD',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            marginBottom: '8px',
            paddingBottom: '6px',
            borderBottom: '1px solid rgba(198, 120, 221, 0.2)',
          }}
        >
          Point Info
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {displayKeys.map(({ key, label }) => {
            const value = data[key as keyof TooltipData];
            if (value === undefined) return null;
            return (
              <div
                key={key}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  gap: '12px',
                }}
              >
                <span
                  style={{
                    fontSize: '12px',
                    color: '#7a6b8a',
                    fontWeight: 500,
                  }}
                >
                  {label}
                </span>
                <span
                  style={{
                    fontSize: '12px',
                    color: '#4a3f5c',
                    fontWeight: 600,
                    fontFamily: 'monospace',
                  }}
                >
                  {formatValue(key, value)}
                </span>
              </div>
            );
          })}
        </div>
        <div
          style={{
            marginTop: '8px',
            paddingTop: '6px',
            borderTop: '1px solid rgba(180, 160, 200, 0.2)',
            fontSize: '10px',
            color: '#9a8baa',
            fontFamily: 'monospace',
          }}
        >
          [{data.position.x.toFixed(2)}, {data.position.y.toFixed(2)}, {data.position.z.toFixed(2)}]
        </div>
      </div>
    </div>
  );
}

export default Tooltip;
