"""
thought_process.py
------------------
Translates the raw OCEAN vector into human-readable psychological insights.

Design principle: rule-based engine now, ML-swappable later.
Each dimension is evaluated at three levels (Low / Medium / High),
then combined into composite behavioral styles.
"""

from __future__ import annotations
from models import OceanVector, ThoughtProcessInsights, TraitLevel


# ─────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────

def _level(score: float) -> str:
    if score >= 0.67:
        return "High"
    elif score >= 0.34:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────
# Per-Trait Descriptors
# ─────────────────────────────────────────────

_TRAIT_DESCRIPTORS: dict[str, dict[str, dict]] = {
    "O": {  # Openness
        "High": {
            "label": "High",
            "description": "Intellectually curious, imaginative, and open to change. Drawn to novel ideas and unconventional solutions.",
        },
        "Medium": {
            "label": "Medium",
            "description": "Balances practicality with creativity. Open to new experiences within a structured framework.",
        },
        "Low": {
            "label": "Low",
            "description": "Grounded and pragmatic. Values proven approaches, consistency, and concrete results over abstract ideas.",
        },
    },
    "C": {  # Conscientiousness
        "High": {
            "label": "High",
            "description": "Disciplined, organized, and goal-driven. Consistently follows through on commitments with high self-regulation.",
        },
        "Medium": {
            "label": "Medium",
            "description": "Generally reliable with a moderate level of planning. Can flex between structure and spontaneity.",
        },
        "Low": {
            "label": "Low",
            "description": "Flexible and adaptable. May prefer spontaneous action over detailed planning; works best with autonomy.",
        },
    },
    "E": {  # Extraversion
        "High": {
            "label": "High",
            "description": "Energized by social interaction. Assertive, talkative, and action-oriented; thrives in collaborative environments.",
        },
        "Medium": {
            "label": "Medium",
            "description": "Ambivert — comfortable in both social settings and solitary work. Adapts energy to context.",
        },
        "Low": {
            "label": "Low",
            "description": "Reflective and independent. Prefers depth over breadth in relationships; recharges through solitude.",
        },
    },
    "A": {  # Agreeableness
        "High": {
            "label": "High",
            "description": "Empathetic, cooperative, and trusting. Prioritizes harmony and others' well-being in decision-making.",
        },
        "Medium": {
            "label": "Medium",
            "description": "Balances cooperation with assertiveness. Diplomatic but willing to push back when needed.",
        },
        "Low": {
            "label": "Low",
            "description": "Direct, competitive, and analytical. Prioritizes objective outcomes over social comfort.",
        },
    },
    "N": {  # Neuroticism
        "High": {
            "label": "High",
            "description": "Emotionally sensitive and reactive. Experiences mood variation strongly; often introspective and empathetic.",
        },
        "Medium": {
            "label": "Medium",
            "description": "Moderate emotional responsiveness. Can manage stress but may be affected under sustained pressure.",
        },
        "Low": {
            "label": "Low",
            "description": "Emotionally stable and resilient. Tends to remain calm under pressure; recovers quickly from setbacks.",
        },
    },
}

_TRAIT_NAMES = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}


# ─────────────────────────────────────────────
# Composite Style Inference
# ─────────────────────────────────────────────

def _decision_style(vec: OceanVector) -> str:
    o, c, n = _level(vec.openness), _level(vec.conscientiousness), _level(vec.neuroticism)

    if o == "High" and c == "High":
        return "Analytical Visionary — evaluates options creatively but decides with structured logic."
    if o == "High" and c == "Low":
        return "Intuitive Explorer — decides quickly based on instinct and curiosity; embraces ambiguity."
    if o == "Low" and c == "High":
        return "Methodical Executor — relies on data, checklists, and proven frameworks before deciding."
    if n == "High":
        return "Cautious Deliberator — weighs risks carefully; may over-analyse under pressure."
    if n == "Low" and c == "High":
        return "Confident Planner — makes decisive, well-reasoned choices with minimal second-guessing."
    return "Pragmatic Realist — adapts decision-making style to context; balances logic with gut feel."


def _work_style(vec: OceanVector) -> str:
    c, e, o = _level(vec.conscientiousness), _level(vec.extraversion), _level(vec.openness)

    if c == "High" and e == "Low":
        return "Deep-focus Specialist — excels at independent, detail-oriented work with clear milestones."
    if c == "High" and e == "High":
        return "Structured Collaborator — combines disciplined execution with strong team coordination."
    if c == "Low" and e == "High":
        return "Energetic Generalist — thrives on variety, brainstorming, and fast-paced team environments."
    if o == "High" and c == "Low":
        return "Creative Experimenter — prefers open-ended, exploratory projects with room for iteration."
    if o == "Low" and c == "High":
        return "Process-Oriented Executor — prefers well-defined tasks with clear steps and measurable outputs."
    return "Adaptive Worker — adjusts approach based on task demands; comfortable in hybrid roles."


def _social_behavior(vec: OceanVector) -> str:
    e, a = _level(vec.extraversion), _level(vec.agreeableness)

    if e == "High" and a == "High":
        return "Collaborative Connector — builds rapport effortlessly; strong team player and people motivator."
    if e == "High" and a == "Low":
        return "Assertive Leader — socially confident and direct; drives outcomes through persuasion and debate."
    if e == "Low" and a == "High":
        return "Quiet Empath — listens deeply and builds meaningful one-on-one relationships."
    if e == "Low" and a == "Low":
        return "Independent Analyst — prefers working autonomously; values competence over social bonding."
    return "Situational Socializer — adapts social energy to environment; effective in both solo and team contexts."


def _strengths(vec: OceanVector) -> list[str]:
    strengths = []
    if vec.openness >= 0.67:
        strengths.append("Creative problem-solving")
        strengths.append("Comfort with ambiguity and change")
    if vec.conscientiousness >= 0.67:
        strengths.append("Reliable execution and follow-through")
        strengths.append("Strong self-discipline and time management")
    if vec.extraversion >= 0.67:
        strengths.append("Persuasive communication")
        strengths.append("Networking and relationship-building")
    if vec.agreeableness >= 0.67:
        strengths.append("Empathy and team cohesion")
        strengths.append("Conflict resolution")
    if vec.neuroticism <= 0.33:
        strengths.append("Emotional resilience under pressure")
        strengths.append("Calm, rational decision-making in crises")
    return strengths[:5] or ["Adaptability across diverse contexts"]


def _growth_areas(vec: OceanVector) -> list[str]:
    areas = []
    if vec.openness < 0.34:
        areas.append("Exploring perspectives outside your comfort zone")
    if vec.conscientiousness < 0.34:
        areas.append("Building structured habits and follow-through")
    if vec.extraversion < 0.34:
        areas.append("Developing assertiveness in group settings")
    if vec.agreeableness < 0.34:
        areas.append("Practising active listening and empathy")
    if vec.neuroticism > 0.67:
        areas.append("Emotional regulation and stress management techniques")
    return areas[:3] or ["Maintaining current balance across all dimensions"]


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def build_trait_levels(vec: OceanVector) -> dict[str, TraitLevel]:
    """Return a TraitLevel for each of the 5 traits."""
    raw = {"O": vec.openness, "C": vec.conscientiousness,
           "E": vec.extraversion, "A": vec.agreeableness, "N": vec.neuroticism}
    result = {}
    for code, score in raw.items():
        lvl = _level(score)
        descriptor = _TRAIT_DESCRIPTORS[code][lvl]
        result[_TRAIT_NAMES[code]] = TraitLevel(
            label=lvl,
            score=round(score, 4),
            description=descriptor["description"],
        )
    return result


def infer_thought_process(vec: OceanVector) -> ThoughtProcessInsights:
    """Convert OCEAN vector → human-readable psychological profile."""
    return ThoughtProcessInsights(
        decision_style=_decision_style(vec),
        work_style=_work_style(vec),
        social_behavior=_social_behavior(vec),
        strengths=_strengths(vec),
        growth_areas=_growth_areas(vec),
    )
