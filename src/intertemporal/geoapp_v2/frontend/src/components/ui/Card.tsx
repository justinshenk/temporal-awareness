import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  onClick?: () => void;
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

interface CardHeaderProps {
  children: React.ReactNode;
  className?: string;
}

interface CardTitleProps {
  children: React.ReactNode;
  className?: string;
}

interface CardDescriptionProps {
  children: React.ReactNode;
  className?: string;
}

interface CardContentProps {
  children: React.ReactNode;
  className?: string;
}

interface CardFooterProps {
  children: React.ReactNode;
  className?: string;
}

const paddingClasses = {
  none: '',
  sm: 'p-3',
  md: 'p-5',
  lg: 'p-7',
};

export const Card: React.FC<CardProps> = ({
  children,
  className = '',
  hover = false,
  onClick,
  padding = 'md',
}) => {
  const isClickable = onClick !== undefined;

  return (
    <div
      onClick={onClick}
      role={isClickable ? 'button' : undefined}
      tabIndex={isClickable ? 0 : undefined}
      onKeyDown={
        isClickable
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onClick();
              }
            }
          : undefined
      }
      className={`
        bg-white/70 backdrop-blur-xl
        border border-white/60
        rounded-2xl
        shadow-xl shadow-purple-500/5
        overflow-visible
        ${paddingClasses[padding]}
        transition-all duration-300 ease-out
        ${hover || isClickable
          ? 'hover:-translate-y-1 hover:shadow-2xl hover:shadow-purple-500/10 hover:bg-white/80 hover:border-pink-200/50'
          : ''
        }
        ${isClickable ? 'cursor-pointer focus:outline-none focus:ring-2 focus:ring-pink-400/50' : ''}
        ${className}
      `}
    >
      {children}
    </div>
  );
};

export const CardHeader: React.FC<CardHeaderProps> = ({
  children,
  className = '',
}) => {
  return (
    <div
      className={`
        pb-4 border-b border-purple-100/50
        ${className}
      `}
    >
      {children}
    </div>
  );
};

export const CardTitle: React.FC<CardTitleProps> = ({
  children,
  className = '',
}) => {
  return (
    <h3
      className={`
        text-lg font-semibold text-[#4a3f5c]
        ${className}
      `}
    >
      {children}
    </h3>
  );
};

export const CardDescription: React.FC<CardDescriptionProps> = ({
  children,
  className = '',
}) => {
  return (
    <p
      className={`
        mt-1 text-sm text-[#4a3f5c]/60
        ${className}
      `}
    >
      {children}
    </p>
  );
};

export const CardContent: React.FC<CardContentProps> = ({
  children,
  className = '',
}) => {
  return (
    <div
      className={`
        py-4
        ${className}
      `}
    >
      {children}
    </div>
  );
};

export const CardFooter: React.FC<CardFooterProps> = ({
  children,
  className = '',
}) => {
  return (
    <div
      className={`
        pt-4 border-t border-purple-100/50
        flex items-center gap-3
        ${className}
      `}
    >
      {children}
    </div>
  );
};

export default Card;
