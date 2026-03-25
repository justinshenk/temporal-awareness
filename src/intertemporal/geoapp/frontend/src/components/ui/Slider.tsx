import React, { useState, useRef, useCallback, useEffect } from 'react';

interface SliderProps {
  min?: number;
  max?: number;
  step?: number;
  value?: number;
  defaultValue?: number;
  onChange?: (value: number) => void;
  onChangeEnd?: (value: number) => void;
  disabled?: boolean;
  showTooltip?: boolean;
  formatTooltip?: (value: number) => string;
  className?: string;
}

export const Slider: React.FC<SliderProps> = ({
  min = 0,
  max = 100,
  step = 1,
  value: controlledValue,
  defaultValue = 50,
  onChange,
  onChangeEnd,
  disabled = false,
  showTooltip = true,
  formatTooltip = (v) => String(v),
  className = '',
}) => {
  const [internalValue, setInternalValue] = useState(defaultValue);
  const [isDragging, setIsDragging] = useState(false);
  const [isHovering, setIsHovering] = useState(false);
  const trackRef = useRef<HTMLDivElement>(null);

  const value = controlledValue !== undefined ? controlledValue : internalValue;
  const percentage = ((value - min) / (max - min)) * 100;

  const updateValue = useCallback(
    (clientX: number) => {
      if (!trackRef.current || disabled) return;

      const rect = trackRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const percent = Math.max(0, Math.min(1, x / rect.width));
      const rawValue = min + percent * (max - min);
      const steppedValue = Math.round(rawValue / step) * step;
      const clampedValue = Math.max(min, Math.min(max, steppedValue));

      setInternalValue(clampedValue);
      onChange?.(clampedValue);
    },
    [min, max, step, disabled, onChange]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (disabled) return;
      e.preventDefault();
      setIsDragging(true);
      updateValue(e.clientX);
    },
    [disabled, updateValue]
  );

  const handleTouchStart = useCallback(
    (e: React.TouchEvent) => {
      if (disabled) return;
      setIsDragging(true);
      updateValue(e.touches[0].clientX);
    },
    [disabled, updateValue]
  );

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      updateValue(e.clientX);
    };

    const handleTouchMove = (e: TouchEvent) => {
      updateValue(e.touches[0].clientX);
    };

    const handleEnd = () => {
      setIsDragging(false);
      onChangeEnd?.(value);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleEnd);
    document.addEventListener('touchmove', handleTouchMove);
    document.addEventListener('touchend', handleEnd);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleEnd);
      document.removeEventListener('touchmove', handleTouchMove);
      document.removeEventListener('touchend', handleEnd);
    };
  }, [isDragging, updateValue, onChangeEnd, value]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (disabled) return;

      let newValue = value;
      switch (e.key) {
        case 'ArrowRight':
        case 'ArrowUp':
          e.preventDefault();
          newValue = Math.min(max, value + step);
          break;
        case 'ArrowLeft':
        case 'ArrowDown':
          e.preventDefault();
          newValue = Math.max(min, value - step);
          break;
        case 'Home':
          e.preventDefault();
          newValue = min;
          break;
        case 'End':
          e.preventDefault();
          newValue = max;
          break;
        default:
          return;
      }

      setInternalValue(newValue);
      onChange?.(newValue);
      onChangeEnd?.(newValue);
    },
    [disabled, value, min, max, step, onChange, onChangeEnd]
  );

  const showTooltipNow = showTooltip && (isDragging || isHovering);

  return (
    <div className={`relative py-2 ${className}`}>
      <div
        ref={trackRef}
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        aria-disabled={disabled}
        onMouseDown={handleMouseDown}
        onTouchStart={handleTouchStart}
        onKeyDown={handleKeyDown}
        onMouseEnter={() => setIsHovering(true)}
        onMouseLeave={() => setIsHovering(false)}
        className={`
          relative h-2 rounded-full
          bg-gradient-to-r from-[#f5f0eb] to-pink-100
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          focus:outline-none focus:ring-2 focus:ring-[#D97757]/50 focus:ring-offset-2 focus:ring-offset-transparent
          transition-all duration-200
        `}
      >
        {/* Filled track */}
        <div
          className="
            absolute inset-y-0 left-0 rounded-full
            bg-gradient-to-r from-[#D97757] to-[#348296]
            transition-all duration-75 ease-out
          "
          style={{ width: `${percentage}%` }}
        />

        {/* Thumb */}
        <div
          className={`
            absolute top-1/2 -translate-y-1/2 -translate-x-1/2
            w-5 h-5 rounded-full
            bg-white
            border-2 border-[#348296]
            shadow-lg shadow-[#D97757]/30
            transition-all duration-150
            ${isDragging ? 'scale-125 shadow-xl shadow-[#D97757]/40' : ''}
            ${!disabled && !isDragging ? 'hover:scale-110' : ''}
          `}
          style={{ left: `${percentage}%` }}
        >
          {/* Inner glow */}
          <div
            className={`
              absolute inset-1 rounded-full
              bg-gradient-to-br from-[#348296] to-[#D97757]
              transition-opacity duration-200
              ${isDragging ? 'opacity-100' : 'opacity-60'}
            `}
          />
        </div>

        {/* Tooltip */}
        <div
          className={`
            absolute bottom-full mb-3 -translate-x-1/2
            px-2.5 py-1 rounded-lg
            bg-gradient-to-r from-[#D97757] to-[#348296]
            text-white text-sm font-medium
            shadow-lg shadow-[#faf8f5]0/20
            transition-all duration-200
            ${showTooltipNow
              ? 'opacity-100 translate-y-0'
              : 'opacity-0 translate-y-1 pointer-events-none'
            }
          `}
          style={{ left: `${percentage}%` }}
        >
          {formatTooltip(value)}
          {/* Arrow */}
          <div
            className="
              absolute top-full left-1/2 -translate-x-1/2
              border-4 border-transparent border-t-[#348296]
            "
          />
        </div>
      </div>
    </div>
  );
};

export default Slider;
