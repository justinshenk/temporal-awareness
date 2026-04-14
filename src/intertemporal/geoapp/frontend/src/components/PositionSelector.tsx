import { useEffect, useState, useMemo } from 'react';
import { PromptTemplateElement, ExampleSample } from '../hooks/useEmbeddings';

interface PositionSelectorProps {
  position: string;
  promptTemplate: PromptTemplateElement[];
  positionLabels: Record<string, string>;
  onPositionChange: (position: string) => void;
  exampleSample?: ExampleSample | null;
  relPosCounts?: Record<string, number>;
  isDarkMode?: boolean;
}

// Exact colors from Python FORMAT_POS_COLORS
const FORMAT_POS_COLORS: Record<string, string> = {
  chat_prefix: '#78909C', chat_prefix_tail: '#90A4AE',
  chat_suffix: '#78909C', chat_suffix_tail: '#90A4AE',
  situation_marker: '#E65100', situation_content: '#FFCC80',
  situation_tail: '#FFB74D', situation: '#FFA726', role: '#FF9800',
  task_marker: '#1565C0', task_content: '#90CAF9',
  task_tail: '#64B5F6', task_in_question: '#42A5F5',
  option_content: '#B2DFDB', options_tail: '#80CBC4',
  left_label: '#7CB342', left_reward: '#558B2F',
  left_reward_units: '#689F38', left_time: '#8BC34A',
  right_label: '#00838F', right_reward: '#00ACC1',
  right_reward_units: '#26C6DA', right_time: '#4DD0E1',
  objective_marker: '#6A1B9A', objective_content: '#CE93D8', objective_tail: '#AB47BC',
  constraint_marker: '#C62828', constraint_content: '#EF9A9A',
  constraint_tail: '#E57373', constraint_prefix: '#EF5350',
  time_horizon: '#FF5722', post_time_horizon: '#FF8A65',
  action_marker: '#2E7D32', action_content: '#A5D6A7', action_tail: '#66BB6A',
  format_marker: '#4E342E', format_content: '#BCAAA4', format_tail: '#8D6E63',
  format_choice_prefix: '#6D4C41', format_reasoning_prefix: '#A1887F',
  reasoning_ask: '#D7CCC8',
  response_choice_prefix: '#5C6BC0', response_choice: '#3F51B5',
  response_reasoning_prefix: '#7986CB', response_reasoning: '#9FA8DA',
  response_other: '#C5CAE9',
  prompt_other: '#CFD8DC',
};

const DEFAULT_COLOR = '#9E9E9E';

function getPositionColor(posName: string): string {
  return FORMAT_POS_COLORS[posName] || DEFAULT_COLOR;
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Canonical order for format positions (prompt order)
const CANONICAL_ORDER = [
  'chat_prefix', 'chat_prefix_tail',
  'situation_marker', 'situation', 'situation_content', 'situation_tail',
  'task_marker', 'task_content', 'role', 'task_in_question', 'task_tail',
  'left_label', 'option_content', 'left_reward', 'left_reward_units', 'left_time',
  'right_label', 'right_reward', 'right_reward_units', 'right_time', 'options_tail',
  'objective_marker', 'objective_content', 'objective_tail',
  'time_horizon', 'post_time_horizon',
  'constraint_marker', 'constraint_prefix', 'constraint_content', 'constraint_tail',
  'action_marker', 'action_content', 'action_tail', 'reasoning_ask',
  'format_marker', 'format_content', 'format_choice_prefix', 'format_reasoning_prefix', 'format_tail',
  'chat_suffix', 'chat_suffix_tail',
  'response_choice_prefix', 'response_choice', 'response_reasoning_prefix', 'response_reasoning', 'response_other',
];

// Positions that start the RESPONSE section
const RESPONSE_POSITIONS = new Set([
  'response_choice_prefix', 'response_choice', 'response_reasoning_prefix',
  'response_reasoning', 'response_other'
]);

export function PositionSelector({
  position,
  promptTemplate,
  positionLabels: _,
  onPositionChange,
  exampleSample,
  relPosCounts,
  isDarkMode = false,
}: PositionSelectorProps) {
  // Parse current position
  const [currentPosBase, currentRelPos] = position.includes(':')
    ? [position.split(':')[0], parseInt(position.split(':')[1])]
    : [position, null];

  const [relPosMode, setRelPosMode] = useState<'combined' | number>(
    currentRelPos !== null ? currentRelPos : 'combined'
  );

  useEffect(() => {
    setRelPosMode(currentRelPos !== null ? currentRelPos : 'combined');
  }, [currentRelPos]);

  const availableSet = new Set(promptTemplate.filter(e => e.available).map(e => e.name));
  const formatTexts = exampleSample?.format_texts || {};

  // Get ALL positions from format_texts, sorted by canonical order
  const sortedPositions = useMemo(() => {
    const positions = Object.keys(formatTexts);
    return positions.sort((a, b) => {
      const idxA = CANONICAL_ORDER.indexOf(a);
      const idxB = CANONICAL_ORDER.indexOf(b);
      // If not in canonical order, put at end
      const orderA = idxA === -1 ? 999 : idxA;
      const orderB = idxB === -1 ? 999 : idxB;
      return orderA - orderB;
    });
  }, [formatTexts]);

  // Split into prompt and response positions
  const promptPositions = sortedPositions.filter(p => !RESPONSE_POSITIONS.has(p));
  const responsePositions = sortedPositions.filter(p => RESPONSE_POSITIONS.has(p));

  const handlePositionClick = (name: string) => {
    if (relPosMode === 'combined') {
      onPositionChange(name);
    } else {
      onPositionChange(`${name}:${relPosMode}`);
    }
  };

  const handleRelPosModeChange = (mode: 'combined' | number) => {
    setRelPosMode(mode);
    if (mode === 'combined') {
      onPositionChange(currentPosBase);
    } else {
      onPositionChange(`${currentPosBase}:${mode}`);
    }
  };

  const getText = (name: string): string => {
    const text = formatTexts[name] || '';
    const cleaned = text.replace(/\n/g, ' ').trim();
    if (cleaned.length <= 20) return cleaned || name;
    return cleaned.slice(0, 19) + '…';
  };

  const renderToken = (name: string) => {
    const text = getText(name);
    const baseColor = getPositionColor(name);
    const isAvailable = availableSet.has(name);
    const isSelected = currentPosBase === name;

    // Compute styles based on state
    const bgColor = isSelected
      ? baseColor
      : hexToRgba(baseColor, isAvailable ? 0.3 : 0.15);
    const borderColor = isSelected
      ? baseColor
      : hexToRgba(baseColor, isAvailable ? 0.7 : 0.4);
    const textColor = isSelected
      ? 'white'
      : (isDarkMode ? '#e0e0e0' : '#333');

    return (
      <button
        key={name}
        onClick={() => isAvailable && handlePositionClick(name)}
        disabled={!isAvailable}
        style={{
          backgroundColor: bgColor,
          borderColor: borderColor,
          color: isAvailable ? textColor : (isDarkMode ? '#666' : '#999'),
          opacity: isAvailable ? 1 : 0.5,
        }}
        className={`
          px-1 py-0.5 rounded border text-[9px] font-medium transition-all
          ${isAvailable ? 'cursor-pointer hover:opacity-80' : 'cursor-not-allowed'}
          ${isSelected ? 'ring-2 ring-offset-1 ring-purple-400' : ''}
        `}
        title={`${name}${!isAvailable ? ' (no data)' : ''}`}
      >
        {text}
      </button>
    );
  };

  if (Object.keys(formatTexts).length === 0) {
    return (
      <div style={{
        padding: '8px',
        color: isDarkMode ? '#999' : '#666',
        backgroundColor: isDarkMode ? '#2a2623' : '#fff',
      }}>
        No example sample data available
      </div>
    );
  }

  const maxRelPos = relPosCounts?.[currentPosBase] || 1;

  // Use inline styles for reliable light/dark mode
  const containerBg = isDarkMode ? '#2a2623' : '#ffffff';
  const containerBorder = isDarkMode ? '#3a3633' : '#e5e7eb';
  const headerBg = isDarkMode ? 'rgba(185, 28, 28, 0.2)' : 'rgba(254, 242, 242, 0.5)';
  const responseBg = isDarkMode ? 'rgba(22, 101, 52, 0.2)' : 'rgba(240, 253, 244, 0.5)';
  const textMuted = isDarkMode ? '#9ca3af' : '#6b7280';
  const textPrimary = isDarkMode ? '#e5e7eb' : '#1f2937';

  return (
    <div style={{ fontSize: '10px' }}>
      <div style={{
        padding: '8px',
        backgroundColor: containerBg,
        border: `1px solid ${containerBorder}`,
        borderRadius: '8px',
        fontFamily: 'monospace',
        lineHeight: 1.6,
      }}>
        {/* PROMPT header */}
        <div style={{
          padding: '4px 8px',
          margin: '-8px -8px 8px -8px',
          backgroundColor: headerBg,
          borderBottom: '2px solid #f87171',
          borderRadius: '8px 8px 0 0',
        }}>
          <span style={{ color: '#ef4444', fontWeight: 'bold', fontSize: '9px' }}>PROMPT:</span>
        </div>

        {/* All prompt positions */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px' }}>
          {promptPositions.map(renderToken)}
        </div>

        {/* RESPONSE header */}
        {responsePositions.length > 0 && (
          <>
            <div style={{
              padding: '4px 8px',
              margin: '8px -8px -8px -8px',
              backgroundColor: responseBg,
              borderTop: '2px solid #4ade80',
              borderRadius: '0 0 8px 8px',
            }}>
              <span style={{ color: '#22c55e', fontWeight: 'bold', fontSize: '9px' }}>RESPONSE:</span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px', marginTop: '4px' }}>
                {responsePositions.map(renderToken)}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Selected position */}
      <div style={{ marginTop: '6px', fontSize: '9px', color: textMuted }}>
        Selected: <span style={{ fontWeight: 600, color: textPrimary }}>{position.replace(/_/g, ' ')}</span>
        {!availableSet.has(currentPosBase) && (
          <span style={{ marginLeft: '8px', color: '#ef4444' }}>(no data)</span>
        )}
      </div>

      {/* Rel Pos selector - select specific token index within a position */}
      {maxRelPos > 1 && (
        <div style={{
          marginTop: '8px',
          paddingTop: '8px',
          borderTop: `1px solid ${containerBorder}`
        }}>
          <div style={{ fontSize: '9px', color: textMuted, marginBottom: '4px' }}>
            Token index for "{currentPosBase}" ({maxRelPos} tokens):
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
            <button
              onClick={() => handleRelPosModeChange('combined')}
              style={{
                padding: '2px 8px',
                fontSize: '9px',
                borderRadius: '4px',
                border: `1px solid ${relPosMode === 'combined' ? '#D97757' : containerBorder}`,
                backgroundColor: relPosMode === 'combined' ? '#D97757' : containerBg,
                color: relPosMode === 'combined' ? 'white' : textMuted,
                cursor: 'pointer',
              }}
            >
              All
            </button>
            {Array.from({ length: maxRelPos }, (_, i) => (
              <button
                key={i}
                onClick={() => handleRelPosModeChange(i)}
                style={{
                  padding: '2px 8px',
                  fontSize: '9px',
                  borderRadius: '4px',
                  border: `1px solid ${relPosMode === i ? '#D97757' : containerBorder}`,
                  backgroundColor: relPosMode === i ? '#D97757' : containerBg,
                  color: relPosMode === i ? 'white' : textMuted,
                  cursor: 'pointer',
                }}
              >
                {i}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
