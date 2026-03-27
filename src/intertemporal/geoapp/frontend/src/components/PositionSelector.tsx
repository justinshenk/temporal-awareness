import { PromptTemplateElement } from '../hooks/useEmbeddings';

interface PositionSelectorProps {
  position: string;
  promptTemplate: PromptTemplateElement[];
  positionLabels: Record<string, string>;
  onPositionChange: (position: string) => void;
}

// Color scheme for different semantic regions
const REGION_COLORS: Record<string, { base: string; selected: string }> = {
  marker: {
    base: 'bg-slate-100 hover:bg-slate-200 border-slate-300 text-slate-700',
    selected: 'bg-slate-500 text-white border-slate-600',
  },
  left: {
    base: 'bg-blue-100 hover:bg-blue-200 border-blue-300 text-blue-800',
    selected: 'bg-blue-500 text-white border-blue-600',
  },
  right: {
    base: 'bg-indigo-100 hover:bg-indigo-200 border-indigo-300 text-indigo-800',
    selected: 'bg-indigo-500 text-white border-indigo-600',
  },
  time_horizon: {
    base: 'bg-amber-100 hover:bg-amber-200 border-amber-300 text-amber-800',
    selected: 'bg-amber-500 text-white border-amber-600',
  },
  response: {
    base: 'bg-green-100 hover:bg-green-200 border-green-300 text-green-800',
    selected: 'bg-green-500 text-white border-green-600',
  },
  content: {
    base: 'bg-gray-100 hover:bg-gray-200 border-gray-300 text-gray-700',
    selected: 'bg-gray-600 text-white border-gray-700',
  },
  context: {
    base: 'bg-purple-100 hover:bg-purple-200 border-purple-300 text-purple-700',
    selected: 'bg-purple-500 text-white border-purple-600',
  },
};

function getRegionColor(posName: string): { base: string; selected: string } {
  if (posName.includes('marker')) return REGION_COLORS.marker;
  if (posName.startsWith('left_')) return REGION_COLORS.left;
  if (posName.startsWith('right_')) return REGION_COLORS.right;
  if (posName.includes('time_horizon') || posName === 'post_time_horizon') return REGION_COLORS.time_horizon;
  if (posName.startsWith('response_')) return REGION_COLORS.response;
  if (posName.endsWith('_content') || posName === 'chat_prefix') return REGION_COLORS.content;
  if (['situation', 'role', 'task_in_question', 'reasoning_ask', 'reward_units'].includes(posName)) {
    return REGION_COLORS.context;
  }
  return REGION_COLORS.marker;
}

interface PositionButtonProps {
  name: string;
  label: string;
  available: boolean;
  selected: boolean;
  onClick: () => void;
}

function PositionButton({
  name,
  label,
  available,
  selected,
  onClick,
}: PositionButtonProps) {
  const colors = getRegionColor(name);

  const baseStyle = available
    ? (selected ? colors.selected + ' ring-2 ring-offset-1 ring-purple-400 shadow-md' : colors.base)
    : 'bg-gray-50 hover:bg-gray-100 border-gray-200 text-gray-400 border-dashed';

  return (
    <button
      onClick={onClick}
      className={`
        px-1.5 py-0.5 rounded border text-[10px] font-medium transition-all cursor-pointer
        ${baseStyle}
      `}
      title={available ? name : `${name} (no data)`}
    >
      [{label}]
      {!available && <span className="ml-0.5 text-[8px]">∅</span>}
    </button>
  );
}

export function PositionSelector({
  position,
  promptTemplate,
  positionLabels,
  onPositionChange,
}: PositionSelectorProps) {
  // Build available positions set from promptTemplate
  const availableSet = new Set(promptTemplate.filter(e => e.available).map(e => e.name));

  // Helper to render a position button
  const renderPos = (name: string, displayLabel?: string) => {
    const label = displayLabel || positionLabels[name] || name.replace(/_/g, ' ');
    const available = availableSet.has(name);
    const isSelected = position === name;

    return (
      <PositionButton
        key={name}
        name={name}
        label={label}
        available={available}
        selected={isSelected}
        onClick={() => onPositionChange(name)}
      />
    );
  };

  return (
    <div className="text-[11px]">
      {/* Prompt Template Visualization */}
      <div className="p-2 bg-white border border-gray-200 rounded-lg space-y-1.5 font-mono leading-relaxed">
        {/* Prompt label - mirrors MODEL RESPONSE label */}
        <div className="pb-1.5 border-b-2 border-red-400 -mx-2 px-2 -mt-2 pt-2 mb-1 bg-red-50/50 rounded-t-lg">
          <span className="text-red-600 font-bold text-[10px]">PROMPT:</span>
        </div>

        {/* Chat prefix - VERY FIRST (position 0): <|im_start|>user\n */}
        <div className="flex flex-wrap items-center gap-0.5 text-gray-400 pb-1 border-b border-dashed border-gray-200">
          {renderPos('chat_prefix', 'chat_prefix')}
          <span className="text-[8px]">(&lt;|im_start|&gt;user)</span>
        </div>

        {/* SITUATION */}
        <div className="flex flex-wrap items-center gap-0.5">
          {renderPos('situation_marker', 'SITUATION:')}
          {renderPos('situation')}
          {renderPos('situation_content', 'content')}
        </div>

        {/* TASK */}
        <div className="flex flex-wrap items-center gap-0.5">
          {renderPos('task_marker', 'TASK:')}
          <span className="text-gray-400 text-[10px]">You,</span>
          {renderPos('role')}
          <span className="text-gray-400 text-[10px]">, are tasked to</span>
          {renderPos('task_in_question', 'task')}
        </div>

        {/* Task content and options */}
        <div className="pl-3 space-y-1">
          {renderPos('task_content', 'task content')}
          <div className="flex flex-wrap items-center gap-0.5">
            {renderPos('left_label', 'a)')}
            {renderPos('left_reward')}
            <span className="text-gray-400 text-[10px]">in</span>
            {renderPos('left_time')}
          </div>
          <div className="flex flex-wrap items-center gap-0.5">
            {renderPos('right_label', 'b)')}
            {renderPos('right_reward')}
            <span className="text-gray-400 text-[10px]">in</span>
            {renderPos('right_time')}
          </div>
        </div>

        {/* CONSIDER + Time Horizon */}
        <div className="flex flex-wrap items-center gap-0.5">
          {renderPos('objective_marker', 'CONSIDER:')}
          {renderPos('objective_content', 'content')}
        </div>
        <div className="pl-3 flex flex-wrap items-center gap-0.5">
          <span className="text-gray-400 text-[10px]">Concerned about outcome in</span>
          {renderPos('time_horizon')}
          {renderPos('post_time_horizon', 'after')}
        </div>

        {/* ACTION */}
        <div className="flex flex-wrap items-center gap-0.5">
          {renderPos('action_marker', 'ACTION:')}
          {renderPos('reasoning_ask')}
          {renderPos('action_content', 'content')}
        </div>

        {/* FORMAT */}
        <div className="flex flex-wrap items-center gap-0.5">
          {renderPos('format_marker', 'FORMAT:')}
          {renderPos('format_content', 'content')}
        </div>
        <div className="pl-3 flex flex-wrap items-center gap-0.5">
          {renderPos('format_choice_prefix', 'I choose:')}
          <span className="text-gray-400 text-[10px]">&lt;a or b&gt;.</span>
          {renderPos('format_reasoning_prefix', 'My reasoning:')}
        </div>

        {/* Chat suffix - END of prompt, BEFORE model response */}
        {/* Position ~122: <|im_end|>\n<|im_start|>assistant\n */}
        <div className="pt-1.5 border-t border-dashed border-gray-300 flex flex-wrap items-center gap-0.5 text-gray-400">
          {renderPos('chat_suffix', 'chat_suffix')}
          <span className="text-[8px]">(&lt;|im_end|&gt;...&lt;|im_start|&gt;assistant)</span>
        </div>

        {/* Response - MODEL OUTPUT starts here (position 127+) */}
        <div className="pt-1.5 border-t-2 border-green-400 bg-green-50/50 -mx-2 px-2 pb-1 rounded-b-lg">
          <div className="flex flex-wrap items-center gap-0.5">
            <span className="text-green-600 font-bold text-[10px]">MODEL RESPONSE:</span>
          </div>
          <div className="pl-3 flex flex-wrap items-center gap-0.5">
            {renderPos('response_choice_prefix', 'I choose:')}
            {renderPos('response_choice', 'choice')}
          </div>
          <div className="pl-3 flex flex-wrap items-center gap-0.5">
            {renderPos('response_reasoning_prefix', 'My reasoning:')}
            {renderPos('response_reasoning', 'reasoning')}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-2 flex flex-wrap gap-2 text-[9px] text-gray-500">
        <span className="flex items-center gap-0.5">
          <span className="w-2 h-2 rounded bg-blue-200 border border-blue-300" />
          Left option
        </span>
        <span className="flex items-center gap-0.5">
          <span className="w-2 h-2 rounded bg-indigo-200 border border-indigo-300" />
          Right option
        </span>
        <span className="flex items-center gap-0.5">
          <span className="w-2 h-2 rounded bg-amber-200 border border-amber-300" />
          Time Horizon
        </span>
        <span className="flex items-center gap-0.5">
          <span className="w-2 h-2 rounded bg-green-200 border border-green-300" />
          Response
        </span>
        <span className="flex items-center gap-0.5">
          <span className="w-2 h-2 rounded bg-purple-200 border border-purple-300" />
          Context
        </span>
        <span className="flex items-center gap-0.5">
          <span className="w-2 h-2 rounded bg-gray-200 border border-gray-300" />
          Content
        </span>
      </div>

      {/* Current selection */}
      <div className="mt-1 text-[9px] text-gray-500">
        Selected: <span className="font-medium text-gray-700">{position}</span>
      </div>
    </div>
  );
}
