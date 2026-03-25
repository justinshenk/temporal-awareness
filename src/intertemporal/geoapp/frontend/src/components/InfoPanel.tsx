import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from './ui/Card';
import { Button } from './ui/Button';
import { Badge, BadgeGroup } from './ui/Badge';

interface SampleInfo {
  idx: number;
  text: string;
  timeHorizon: number | null;
  timeScale: string | null;
  choiceType: string | null;
  shortTermFirst: boolean | null;
  label?: string;
}

// Format time horizon (in months) to human-readable string
function formatTimeHorizon(months: number): string {
  if (months <= 0) {
    return 'No horizon';
  }
  if (months < 1) {
    const days = Math.round(months * 30);
    if (days <= 0) return 'Less than a day';
    return days === 1 ? '1 day' : `${days} days`;
  }
  if (months < 12) {
    return months === 1 ? '1 month' : `${months.toFixed(1).replace(/\.0$/, '')} months`;
  }
  const years = months / 12;
  if (years < 100) {
    return years === 1 ? '1 year' : `${years.toFixed(1).replace(/\.0$/, '')} years`;
  }
  if (years < 1000) {
    const decades = years / 10;
    return `${decades.toFixed(0)} decades`;
  }
  const centuries = years / 100;
  return centuries === 1 ? '1 century' : `${centuries.toFixed(0)} centuries`;
}

interface InfoPanelProps {
  selectedSample: SampleInfo | null;
  isLoading?: boolean;
  onClose?: () => void;
  className?: string;
  /** Section markers from config (e.g., {situation_marker: "SITUATION:"}) */
  markers?: Record<string, string>;
}

const CopyIcon = () => (
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
      d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
    />
  </svg>
);

const CheckIcon = () => (
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
      d="M5 13l4 4L19 7"
    />
  </svg>
);

const CloseIcon = () => (
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
      d="M6 18L18 6M6 6l12 12"
    />
  </svg>
);

// Time pattern to highlight time horizons in text
// Matches patterns like "5 days", "10 years", "3 months", etc.
const TIME_PATTERN = /(\d+\s+(?:days?|weeks?|months?|years?|hours?|minutes?))/gi;

// Option pattern for parsing choices - supports a-z, A-Z, 0-9 labels with ) or .
const OPTION_PATTERN = /^([a-zA-Z0-9])[).]\s*/;

interface ParsedSection {
  name: string;
  content: string;
  options?: { label: string; text: string }[];
}

// Default markers if none provided (fallback)
const DEFAULT_MARKERS: Record<string, string> = {
  situation_marker: 'SITUATION:',
  task_marker: 'TASK:',
  consider_marker: 'CONSIDER:',
  action_marker: 'ACTION:',
  format_marker: 'FORMAT:',
};

// Extract section name from marker (e.g., "SITUATION:" -> "SITUATION")
const getSectionName = (marker: string): string => {
  return marker.replace(/:$/, '').toUpperCase();
};

// Parse the prompt text into structured sections
const parsePromptText = (text: string, markers?: Record<string, string>): ParsedSection[] => {
  // Handle empty/null/undefined text
  if (!text || typeof text !== 'string') {
    return [{ name: 'CONTENT', content: '' }];
  }

  const sections: ParsedSection[] = [];

  // Use provided markers or defaults
  const effectiveMarkers = markers && Object.keys(markers).length > 0 ? markers : DEFAULT_MARKERS;

  // Get section names from markers in order
  const markerOrder = ['situation_marker', 'task_marker', 'consider_marker', 'action_marker', 'format_marker'];
  const sectionNames = markerOrder
    .filter(key => key in effectiveMarkers)
    .map(key => getSectionName(effectiveMarkers[key]));

  // Normalize line breaks
  const normalizedText = text.replace(/\\n/g, '\n');

  // Split by section headers
  let remaining = normalizedText;

  for (let i = 0; i < sectionNames.length; i++) {
    const currentSection = sectionNames[i];
    const nextSections = sectionNames.slice(i + 1);

    // Create pattern for current section
    const sectionPattern = new RegExp(`${currentSection}:\\s*`, 'i');
    const match = remaining.match(sectionPattern);

    if (match) {
      // Find where this section starts
      const startIdx = remaining.indexOf(match[0]);
      const contentStart = startIdx + match[0].length;

      // Find where next section starts (if any)
      let endIdx = remaining.length;
      for (const nextSection of nextSections) {
        const nextPattern = new RegExp(`${nextSection}:`, 'i');
        const nextMatch = remaining.slice(contentStart).match(nextPattern);
        if (nextMatch) {
          endIdx = contentStart + remaining.slice(contentStart).indexOf(nextMatch[0]);
          break;
        }
      }

      const content = remaining.slice(contentStart, endIdx).trim();

      // For TASK section, try to extract options
      if (currentSection === 'TASK') {
        const options: { label: string; text: string }[] = [];

        // Split content by lines and parse options
        const lines = content.split('\n');
        let introLines: string[] = [];
        let currentOption: { label: string; lines: string[] } | null = null;

        for (const line of lines) {
          const trimmedLine = line.trim();
          const optMatch = trimmedLine.match(OPTION_PATTERN);

          if (optMatch) {
            // Save previous option if exists
            if (currentOption) {
              const prevText = currentOption.lines.join(' ').trim();
              if (prevText) {
                options.push({
                  label: currentOption.label,
                  text: prevText
                });
              }
            }
            // Start new option
            currentOption = {
              label: optMatch[1],
              lines: [trimmedLine.slice(optMatch[0].length)]
            };
          } else if (currentOption) {
            // Continue current option (multi-line option text)
            if (trimmedLine) {
              currentOption.lines.push(trimmedLine);
            }
          } else {
            // Part of intro text
            introLines.push(line);
          }
        }

        // Don't forget the last option
        if (currentOption) {
          const optText = currentOption.lines.join(' ').trim();
          if (optText) {
            options.push({
              label: currentOption.label,
              text: optText
            });
          }
        }

        // Filter out any options with empty text that may have been added
        const validOptions = options.filter(opt => opt.text.length > 0);
        const intro = introLines.join('\n').trim();
        sections.push({ name: currentSection, content: intro, options: validOptions.length > 0 ? validOptions : undefined });
      } else {
        sections.push({ name: currentSection, content });
      }

      remaining = remaining.slice(endIdx);
    }
  }

  // If no sections found, return the whole text as a single section
  if (sections.length === 0) {
    return [{ name: 'CONTENT', content: normalizedText }];
  }

  return sections;
};

// Highlight time values in text
const HighlightedText: React.FC<{ text: string }> = ({ text }) => {
  if (!text) return null;
  const parts = text.split(TIME_PATTERN);

  return (
    <>
      {parts.map((part, idx) => {
        // Create a fresh regex for each test to avoid global flag lastIndex issues
        const timePatternTest = /(\d+\s+(?:days?|weeks?|months?|years?|hours?|minutes?))/i;
        if (timePatternTest.test(part)) {
          return (
            <span
              key={idx}
              className="bg-gradient-to-r from-[#D97757]/20 to-[#348296]/20 text-[#7c3aed] font-semibold px-1.5 py-0.5 rounded"
            >
              {part}
            </span>
          );
        }
        return <span key={idx}>{part}</span>;
      })}
    </>
  );
};

// Section icon based on section name
const SectionIcon: React.FC<{ name: string }> = ({ name }) => {
  const iconClass = "w-3.5 h-3.5";

  switch (name) {
    case 'SITUATION':
      return (
        <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      );
    case 'TASK':
      return (
        <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
        </svg>
      );
    case 'CONSIDER':
      return (
        <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      );
    case 'ACTION':
      return (
        <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      );
    case 'FORMAT':
      return (
        <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
        </svg>
      );
    default:
      return (
        <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      );
  }
};

// Section color based on section name
const getSectionColor = (name: string): string => {
  switch (name) {
    case 'SITUATION': return 'border-blue-200 bg-blue-50/50';
    case 'TASK': return 'border-purple-200 bg-[#faf8f5]/50';
    case 'CONSIDER': return 'border-amber-200 bg-amber-50/50';
    case 'ACTION': return 'border-green-200 bg-green-50/50';
    case 'FORMAT': return 'border-gray-200 bg-gray-50/50';
    default: return 'border-gray-200 bg-gray-50/50';
  }
};

const getSectionHeaderColor = (name: string): string => {
  switch (name) {
    case 'SITUATION': return 'text-blue-700';
    case 'TASK': return 'text-purple-700';
    case 'CONSIDER': return 'text-amber-700';
    case 'ACTION': return 'text-green-700';
    case 'FORMAT': return 'text-gray-600';
    default: return 'text-gray-600';
  }
};

// Formatted prompt display component
const FormattedPrompt: React.FC<{ text: string; selectedOption?: string; markers?: Record<string, string> }> = ({ text, selectedOption, markers }) => {
  const sections = parsePromptText(text, markers);

  return (
    <div className="space-y-2.5">
      {sections.map((section, idx) => (
        <div
          key={idx}
          className={`rounded-lg border p-2.5 ${getSectionColor(section.name)}`}
        >
          <div className={`flex items-center gap-1.5 mb-1.5 ${getSectionHeaderColor(section.name)}`}>
            <SectionIcon name={section.name} />
            <span className="text-xs font-semibold uppercase tracking-wide">
              {section.name}
            </span>
          </div>
          {section.content && (
            <div className="text-sm text-[#1a1613] leading-relaxed">
              <HighlightedText text={section.content} />
            </div>
          )}
          {section.options && section.options.length > 0 && (
            <div className="mt-2 space-y-1.5">
              {section.options.map((opt, optIdx) => {
                const isSelected = selectedOption?.toLowerCase() === opt.label.toLowerCase();
                return (
                  <div
                    key={optIdx}
                    className={`flex items-start gap-2 p-2 rounded-md transition-colors ${
                      isSelected
                        ? 'bg-gradient-to-r from-[#D97757]/20 to-[#348296]/20 border border-[#D97757]/30'
                        : 'bg-white/60 border border-transparent'
                    }`}
                  >
                    <span
                      className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
                        isSelected
                          ? 'bg-gradient-to-r from-[#D97757] to-[#348296] text-white'
                          : 'bg-[#f5f0eb] text-purple-600'
                      }`}
                    >
                      {opt.label.toUpperCase()}
                    </span>
                    <span className="text-sm text-[#1a1613]">
                      <HighlightedText text={opt.text} />
                    </span>
                    {isSelected && (
                      <span className="ml-auto flex-shrink-0 text-xs font-medium text-[#D97757]">
                        Selected
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export const InfoPanel: React.FC<InfoPanelProps> = ({
  selectedSample,
  isLoading = false,
  onClose,
  className = '',
  markers,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (selectedSample?.text) {
      await navigator.clipboard.writeText(selectedSample.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className={`flex flex-col gap-4 ${className}`}>
      {/* Selected Sample Card */}
      <Card padding="sm" className="flex-1">
        <CardHeader className="pb-3 flex items-start justify-between">
          <CardTitle className="text-sm">
            {selectedSample ? `Sample #${selectedSample.idx}` : 'Selected Sample'}
          </CardTitle>
          {onClose && selectedSample && (
            <Button
              variant="ghost"
              size="sm"
              icon={<CloseIcon />}
              onClick={onClose}
              aria-label="Clear selection"
              className="!p-1"
            />
          )}
        </CardHeader>
        <CardContent className="py-2">
          {isLoading ? (
            <div className="space-y-3 animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-3/4" />
              <div className="h-4 bg-gray-200 rounded w-1/2" />
              <div className="h-20 bg-gray-200 rounded" />
            </div>
          ) : selectedSample ? (
            <div className="space-y-3">
              {/* Sample attributes */}
              <BadgeGroup>
                {selectedSample.choiceType && (
                  <Badge
                    variant={selectedSample.choiceType === 'short_term' ? 'warning' : 'success'}
                    size="sm"
                    dot
                  >
                    {selectedSample.choiceType === 'short_term'
                      ? 'Short-term'
                      : 'Long-term'}
                  </Badge>
                )}
                {selectedSample.timeHorizon !== null && selectedSample.timeHorizon >= 0 && (
                  <Badge variant="info" size="sm">
                    Horizon: {formatTimeHorizon(selectedSample.timeHorizon)}
                  </Badge>
                )}
                {selectedSample.timeHorizon === null && (
                  <Badge variant="secondary" size="sm">
                    No Horizon
                  </Badge>
                )}
                {selectedSample.shortTermFirst === true && (
                  <Badge variant="secondary" size="sm">
                    ST First
                  </Badge>
                )}
              </BadgeGroup>

              {/* Sample text - formatted prompt display */}
              <div className="relative group">
                <div className="max-h-64 overflow-y-auto pr-1">
                  <FormattedPrompt
                    text={selectedSample.text}
                    selectedOption={selectedSample.label}
                    markers={markers}
                  />
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  icon={copied ? <CheckIcon /> : <CopyIcon />}
                  onClick={handleCopy}
                  aria-label={copied ? 'Copied to clipboard' : 'Copy sample text'}
                  className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity bg-white/80 hover:bg-white"
                />
              </div>

              {/* Label if available */}
              {selectedSample.label && (
                <div className="text-xs text-[#1a1613]/50">
                  Label: {selectedSample.label}
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-gradient-to-br from-[#D97757]/20 to-[#348296]/20 flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-[#D97757]"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"
                  />
                </svg>
              </div>
              <p className="text-sm text-[#1a1613]/50">
                Click on a point to see details
              </p>
            </div>
          )}
        </CardContent>
        {selectedSample && (
          <CardFooter className="pt-2">
            <Button variant="ghost" size="sm" onClick={handleCopy}>
              {copied ? 'Copied!' : 'Copy Text'}
            </Button>
          </CardFooter>
        )}
      </Card>
    </div>
  );
};

export default InfoPanel;
