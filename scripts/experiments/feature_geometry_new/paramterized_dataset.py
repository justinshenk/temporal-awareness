templates = [
    {
        "id": "task_available_time",
        "template": """Task: {task}
Available time: {vlaue} {unit}

Write a plan optimized for the available time.

Output format:
Strategy: <one sentence>
Steps:
1. ...
2. ...
3. ...""",
    },
    {
        "id": "goal_time_budget",
        "template": """Goal: {task}
Time budget: {vlaue} {unit}

Provide a plan that fits this time budget.

Output format:
Strategy: <one sentence>
Steps:
1. ...
2. ...
3. ...""",
    },
    {
        "id": "objective_deadline",
        "template": """Objective: {task}
Deadline: {vlaue} {unit} from now

Give a plan appropriate for this deadline.

Output format:
Strategy: <one sentence>
Steps:
1. ...
2. ...
3. ...""",
    },
]

tasks = {
    # Personal / lifestyle (medium horizon, moderate structure)
    "plan an international trip": {"hours", "days"},
    # Physical / long-horizon, high dependency
    "prepare for a marathon": {"weeks", "months", "years"},
    # Coordination-heavy, multi-agent
    "organize my friend's wedding": {"days", "weeks", "months"},
    # Life transition, loosely structured
    "move to a new city": {"hours", "days", "weeks"},
    # Skill acquisition, gradual feedback
    "learn a new language": {"days", "weeks", "months"},
    # Creative, open-ended, long horizon
    "write a novel": {"weeks", "months", "years"},
    "write a short story": {"hours", "days"},
    # Operational / resource-constrained
    "renovate an apartment": {"hours", "days"},
    # Business, multi-stage scaling
    "expand a local business to new markets": {"months", "years", "decades"},
    # High-stakes coordination, dynamic environment
    "coordinate disaster relief efforts": {"days", "weeks", "months"},
    # Retain one software task for contrast
    "create an e-commerce website": {"hours", "days", "weeks"},
    "establish a self-sustaining civilization on a new planet": {
        "decades",
        "centuries",
        "millennia",
    },
    "design and preserve a knowledge archive for future civilizations": {
        "years",
        "decades",
        "centuries",
    },
    "restore a degraded ecosystem to long-term stability": {
        "years",
        "decades",
        "centuries",
    },
    "build a city designed to survive natural disasters": {"decades", "centuries"},
    "ensure long-term safe containment of hazardous materials": {
        "months",
        "years",
        "decades",
    },
    "create an institution that remains stable and effective": {
        "months",
        "years",
        "decades",
    },
    "plan the long-term survival strategy of humanity": {"centuries", "millennia"},
    "design infrastructure resilient to climate change": {"decades", "centuries"},
}


values = [1, 2, 3, 4, 5, 10, 50, 100]
