import React from 'react';

interface ToggleProps {
  checked?: boolean;
  defaultChecked?: boolean;
  onChange?: (checked: boolean) => void;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  labelPosition?: 'left' | 'right';
  className?: string;
}

const sizeConfig = {
  sm: {
    track: 'w-8 h-4',
    thumb: 'w-3 h-3',
    translate: 'translate-x-4',
    labelGap: 'gap-2',
    labelText: 'text-sm',
  },
  md: {
    track: 'w-11 h-6',
    thumb: 'w-5 h-5',
    translate: 'translate-x-5',
    labelGap: 'gap-3',
    labelText: 'text-base',
  },
  lg: {
    track: 'w-14 h-8',
    thumb: 'w-7 h-7',
    translate: 'translate-x-6',
    labelGap: 'gap-3',
    labelText: 'text-lg',
  },
};

export const Toggle: React.FC<ToggleProps> = ({
  checked: controlledChecked,
  defaultChecked = false,
  onChange,
  disabled = false,
  size = 'md',
  label,
  labelPosition = 'right',
  className = '',
}) => {
  const [internalChecked, setInternalChecked] = React.useState(defaultChecked);

  const isChecked = controlledChecked !== undefined ? controlledChecked : internalChecked;
  const config = sizeConfig[size];

  const handleToggle = () => {
    if (disabled) return;
    const newValue = !isChecked;
    setInternalChecked(newValue);
    onChange?.(newValue);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleToggle();
    }
  };

  const toggleSwitch = (
    <button
      type="button"
      role="switch"
      aria-checked={isChecked}
      disabled={disabled}
      onClick={handleToggle}
      onKeyDown={handleKeyDown}
      className={`
        relative inline-flex items-center shrink-0
        ${config.track} rounded-full
        transition-all duration-300 ease-out
        focus:outline-none focus:ring-2 focus:ring-pink-400/50 focus:ring-offset-2 focus:ring-offset-transparent
        ${disabled
          ? 'opacity-50 cursor-not-allowed'
          : 'cursor-pointer'
        }
        ${isChecked
          ? 'bg-gradient-to-r from-[#C678DD] to-[#FF6B9D] shadow-lg shadow-pink-500/30'
          : 'bg-gray-200 shadow-inner'
        }
      `}
    >
      {/* Track glow when active */}
      <span
        className={`
          absolute inset-0 rounded-full
          bg-gradient-to-r from-[#C678DD]/20 to-[#FF6B9D]/20
          blur-md
          transition-opacity duration-300
          ${isChecked ? 'opacity-100' : 'opacity-0'}
        `}
      />

      {/* Thumb */}
      <span
        className={`
          absolute left-0.5
          ${config.thumb} rounded-full
          bg-white
          shadow-md
          transition-all duration-300 ease-out
          ${isChecked ? config.translate : 'translate-x-0'}
          ${isChecked ? 'shadow-lg shadow-pink-500/20' : 'shadow-gray-400/20'}
        `}
      >
        {/* Inner gradient when active */}
        <span
          className={`
            absolute inset-0.5 rounded-full
            bg-gradient-to-br from-white to-pink-50
            transition-opacity duration-300
            ${isChecked ? 'opacity-100' : 'opacity-0'}
          `}
        />
      </span>
    </button>
  );

  if (!label) {
    return <div className={className}>{toggleSwitch}</div>;
  }

  return (
    <label
      className={`
        inline-flex items-center ${config.labelGap}
        ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'}
        ${className}
      `}
    >
      {labelPosition === 'left' && (
        <span
          className={`
            ${config.labelText} text-[#4a3f5c]
            ${disabled ? 'opacity-50' : ''}
          `}
        >
          {label}
        </span>
      )}
      {toggleSwitch}
      {labelPosition === 'right' && (
        <span
          className={`
            ${config.labelText} text-[#4a3f5c]
            ${disabled ? 'opacity-50' : ''}
          `}
        >
          {label}
        </span>
      )}
    </label>
  );
};

export default Toggle;
