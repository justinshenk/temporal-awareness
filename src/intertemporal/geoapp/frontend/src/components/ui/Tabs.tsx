import React, { useState, useRef, useEffect, useCallback } from 'react';

interface Tab {
  id: string;
  label: string;
  disabled?: boolean;
}

interface TabsProps {
  tabs: Tab[];
  activeTab?: string;
  onChange?: (tabId: string) => void;
  className?: string;
}

interface TabsContentProps {
  tabId: string;
  activeTab: string;
  children: React.ReactNode;
  className?: string;
}

export const Tabs: React.FC<TabsProps> = ({
  tabs,
  activeTab: controlledActiveTab,
  onChange,
  className = '',
}) => {
  const [internalActiveTab, setInternalActiveTab] = useState(tabs[0]?.id || '');
  const [indicatorStyle, setIndicatorStyle] = useState({ left: 0, width: 0 });
  const tabRefs = useRef<Map<string, HTMLButtonElement>>(new Map());
  const containerRef = useRef<HTMLDivElement>(null);

  const activeTab = controlledActiveTab !== undefined ? controlledActiveTab : internalActiveTab;

  const updateIndicator = useCallback(() => {
    const activeButton = tabRefs.current.get(activeTab);
    const container = containerRef.current;
    if (activeButton && container) {
      const containerRect = container.getBoundingClientRect();
      const buttonRect = activeButton.getBoundingClientRect();
      setIndicatorStyle({
        left: buttonRect.left - containerRect.left,
        width: buttonRect.width,
      });
    }
  }, [activeTab]);

  useEffect(() => {
    updateIndicator();
    window.addEventListener('resize', updateIndicator);
    return () => window.removeEventListener('resize', updateIndicator);
  }, [updateIndicator]);

  const handleTabClick = (tabId: string) => {
    setInternalActiveTab(tabId);
    onChange?.(tabId);
  };

  const handleKeyDown = (e: React.KeyboardEvent, currentIndex: number) => {
    const enabledTabs = tabs.filter((t) => !t.disabled);
    const currentEnabledIndex = enabledTabs.findIndex(
      (t) => t.id === tabs[currentIndex].id
    );

    let newIndex = currentEnabledIndex;

    switch (e.key) {
      case 'ArrowLeft':
        e.preventDefault();
        newIndex = currentEnabledIndex > 0 ? currentEnabledIndex - 1 : enabledTabs.length - 1;
        break;
      case 'ArrowRight':
        e.preventDefault();
        newIndex = currentEnabledIndex < enabledTabs.length - 1 ? currentEnabledIndex + 1 : 0;
        break;
      case 'Home':
        e.preventDefault();
        newIndex = 0;
        break;
      case 'End':
        e.preventDefault();
        newIndex = enabledTabs.length - 1;
        break;
      default:
        return;
    }

    const newTab = enabledTabs[newIndex];
    if (newTab) {
      handleTabClick(newTab.id);
      tabRefs.current.get(newTab.id)?.focus();
    }
  };

  return (
    <div className={className}>
      <div
        ref={containerRef}
        role="tablist"
        className="
          relative flex
          bg-white/50 backdrop-blur-md
          border border-white/60
          rounded-xl p-1
          shadow-lg shadow-[#faf8f5]0/5
        "
      >
        {/* Animated indicator */}
        <div
          className="
            absolute bottom-1 h-1 rounded-full
            bg-gradient-to-r from-[#D97757] to-[#348296]
            transition-all duration-300 ease-out
          "
          style={{
            left: indicatorStyle.left,
            width: indicatorStyle.width,
          }}
        />

        {tabs.map((tab, index) => (
          <button
            key={tab.id}
            ref={(el) => {
              if (el) tabRefs.current.set(tab.id, el);
            }}
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`tabpanel-${tab.id}`}
            tabIndex={activeTab === tab.id ? 0 : -1}
            disabled={tab.disabled}
            onClick={() => !tab.disabled && handleTabClick(tab.id)}
            onKeyDown={(e) => handleKeyDown(e, index)}
            className={`
              relative flex-1 px-4 py-2.5
              text-sm font-medium
              rounded-lg
              transition-all duration-200
              focus:outline-none focus:ring-2 focus:ring-[#D97757]/50 focus:ring-inset
              ${tab.disabled
                ? 'opacity-50 cursor-not-allowed text-[#1a1613]/40'
                : activeTab === tab.id
                ? 'text-[#1a1613]'
                : 'text-[#1a1613]/60 hover:text-[#1a1613] hover:bg-white/50'
              }
            `}
          >
            {/* Active background */}
            <span
              className={`
                absolute inset-0 rounded-lg
                bg-white/80
                transition-opacity duration-200
                ${activeTab === tab.id ? 'opacity-100' : 'opacity-0'}
              `}
            />
            <span className="relative">{tab.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export const TabsContent: React.FC<TabsContentProps> = ({
  tabId,
  activeTab,
  children,
  className = '',
}) => {
  const isActive = tabId === activeTab;

  return (
    <div
      role="tabpanel"
      id={`tabpanel-${tabId}`}
      aria-labelledby={`tab-${tabId}`}
      hidden={!isActive}
      className={`
        transition-all duration-300
        ${isActive ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}
        ${className}
      `}
    >
      {isActive && children}
    </div>
  );
};

export default Tabs;
