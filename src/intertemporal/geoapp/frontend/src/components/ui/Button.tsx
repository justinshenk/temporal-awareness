import React from 'react';

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'icon';
type ButtonSize = 'sm' | 'md' | 'lg';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
}

const variantClasses: Record<ButtonVariant, string> = {
  primary: `
    bg-gradient-to-r from-[#D97757] to-[#348296]
    text-white font-medium
    shadow-lg shadow-[#D97757]/25
    hover:shadow-xl hover:shadow-[#D97757]/30
    hover:from-[#b86cd0] hover:to-[#ff5a8f]
    active:shadow-md active:scale-[0.98]
    disabled:from-gray-300 disabled:to-gray-400 disabled:shadow-none
  `,
  secondary: `
    bg-white/70 backdrop-blur-md
    border-2 border-[#D97757]/30
    text-[#1a1613] font-medium
    shadow-md shadow-[#faf8f5]0/10
    hover:border-[#348296]/50 hover:bg-white/90
    hover:shadow-lg hover:shadow-[#D97757]/15
    active:scale-[0.98]
    disabled:border-gray-200 disabled:text-gray-400 disabled:bg-gray-50
  `,
  ghost: `
    bg-transparent
    text-[#1a1613]
    hover:bg-[#faf8f5]/50 hover:text-[#D97757]
    active:bg-[#f5f0eb]/50
    disabled:text-gray-300 disabled:bg-transparent
  `,
  icon: `
    bg-white/70 backdrop-blur-md
    border border-white/60
    text-[#1a1613]
    shadow-md shadow-[#faf8f5]0/5
    hover:bg-white/90 hover:border-[#D97757]/30/50 hover:text-[#348296]
    hover:shadow-lg hover:shadow-[#D97757]/10
    active:scale-[0.95]
    disabled:text-gray-300 disabled:bg-gray-50 disabled:border-gray-200
  `,
};

const sizeClasses: Record<ButtonVariant, Record<ButtonSize, string>> = {
  primary: {
    sm: 'px-3 py-1.5 text-sm rounded-lg',
    md: 'px-5 py-2.5 text-base rounded-xl',
    lg: 'px-7 py-3.5 text-lg rounded-xl',
  },
  secondary: {
    sm: 'px-3 py-1.5 text-sm rounded-lg',
    md: 'px-5 py-2.5 text-base rounded-xl',
    lg: 'px-7 py-3.5 text-lg rounded-xl',
  },
  ghost: {
    sm: 'px-2 py-1 text-sm rounded-lg',
    md: 'px-4 py-2 text-base rounded-xl',
    lg: 'px-6 py-3 text-lg rounded-xl',
  },
  icon: {
    sm: 'p-1.5 rounded-lg',
    md: 'p-2.5 rounded-xl',
    lg: 'p-3.5 rounded-xl',
  },
};

const iconSizeClasses: Record<ButtonSize, string> = {
  sm: 'w-4 h-4',
  md: 'w-5 h-5',
  lg: 'w-6 h-6',
};

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  loading = false,
  icon,
  iconPosition = 'left',
  children,
  disabled,
  className = '',
  ...props
}) => {
  const isDisabled = disabled || loading;
  const isIconOnly = variant === 'icon' || (!children && icon);

  const LoadingSpinner = () => (
    <svg
      className={`animate-spin ${iconSizeClasses[size]} ${children ? (iconPosition === 'left' ? 'mr-2' : 'ml-2') : ''}`}
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );

  const IconWrapper = ({ children: iconChildren }: { children: React.ReactNode }) => (
    <span className={`${iconSizeClasses[size]} flex items-center justify-center`}>
      {iconChildren}
    </span>
  );

  return (
    <button
      disabled={isDisabled}
      className={`
        inline-flex items-center justify-center
        transition-all duration-200 ease-out
        focus:outline-none focus:ring-2 focus:ring-[#D97757]/50 focus:ring-offset-2 focus:ring-offset-transparent
        disabled:cursor-not-allowed disabled:opacity-60
        ${variantClasses[variant]}
        ${sizeClasses[variant][size]}
        ${className}
      `}
      {...props}
    >
      {loading ? (
        <>
          <LoadingSpinner />
          {!isIconOnly && children}
        </>
      ) : (
        <>
          {icon && iconPosition === 'left' && (
            <span className={isIconOnly ? '' : 'mr-2'}>
              <IconWrapper>{icon}</IconWrapper>
            </span>
          )}
          {children}
          {icon && iconPosition === 'right' && (
            <span className={isIconOnly ? '' : 'ml-2'}>
              <IconWrapper>{icon}</IconWrapper>
            </span>
          )}
        </>
      )}
    </button>
  );
};

export default Button;
