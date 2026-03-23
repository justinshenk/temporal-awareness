import React from 'react';

type BadgeVariant =
  | 'default'
  | 'primary'
  | 'secondary'
  | 'success'
  | 'warning'
  | 'error'
  | 'info';

type BadgeSize = 'sm' | 'md' | 'lg';

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  size?: BadgeSize;
  dot?: boolean;
  removable?: boolean;
  onRemove?: () => void;
  className?: string;
}

const variantClasses: Record<BadgeVariant, string> = {
  default: `
    bg-gray-100 text-gray-700
    border border-gray-200
  `,
  primary: `
    bg-gradient-to-r from-[#C678DD]/20 to-[#FF6B9D]/20
    text-[#4a3f5c]
    border border-[#C678DD]/30
  `,
  secondary: `
    bg-purple-50 text-purple-700
    border border-purple-200
  `,
  success: `
    bg-emerald-50 text-emerald-700
    border border-emerald-200
  `,
  warning: `
    bg-amber-50 text-amber-700
    border border-amber-200
  `,
  error: `
    bg-rose-50 text-rose-700
    border border-rose-200
  `,
  info: `
    bg-[#56B6C2]/10 text-[#56B6C2]
    border border-[#56B6C2]/30
  `,
};

const dotColors: Record<BadgeVariant, string> = {
  default: 'bg-gray-500',
  primary: 'bg-gradient-to-r from-[#C678DD] to-[#FF6B9D]',
  secondary: 'bg-purple-500',
  success: 'bg-emerald-500',
  warning: 'bg-amber-500',
  error: 'bg-rose-500',
  info: 'bg-[#56B6C2]',
};

const sizeClasses: Record<BadgeSize, string> = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-2.5 py-1 text-sm',
  lg: 'px-3 py-1.5 text-base',
};

const dotSizeClasses: Record<BadgeSize, string> = {
  sm: 'w-1.5 h-1.5',
  md: 'w-2 h-2',
  lg: 'w-2.5 h-2.5',
};

const removeBtnSizeClasses: Record<BadgeSize, string> = {
  sm: 'w-3 h-3 ml-1',
  md: 'w-4 h-4 ml-1.5',
  lg: 'w-5 h-5 ml-2',
};

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'default',
  size = 'md',
  dot = false,
  removable = false,
  onRemove,
  className = '',
}) => {
  return (
    <span
      className={`
        inline-flex items-center
        font-medium rounded-full
        transition-all duration-200
        ${variantClasses[variant]}
        ${sizeClasses[size]}
        ${className}
      `}
    >
      {dot && (
        <span
          className={`
            ${dotSizeClasses[size]}
            rounded-full mr-1.5
            ${dotColors[variant]}
            ${variant === 'primary' ? '' : ''}
          `}
        />
      )}

      {children}

      {removable && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRemove?.();
          }}
          className={`
            ${removeBtnSizeClasses[size]}
            rounded-full
            flex items-center justify-center
            opacity-60 hover:opacity-100
            hover:bg-black/10
            transition-all duration-150
            focus:outline-none focus:ring-1 focus:ring-current
          `}
          aria-label="Remove"
        >
          <svg
            className="w-full h-full"
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
        </button>
      )}
    </span>
  );
};

export const BadgeGroup: React.FC<{
  children: React.ReactNode;
  className?: string;
}> = ({ children, className = '' }) => {
  return (
    <div className={`flex flex-wrap gap-2 ${className}`}>
      {children}
    </div>
  );
};

export default Badge;
