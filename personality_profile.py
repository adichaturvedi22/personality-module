"""
personality_profile.py
----------------------
Builds a structured PersonalityProfile from an OceanVector.

Determines:
  - A human-readable personality type label (archetype)
  - Dominant trait
  - Full trait breakdown (via thought_process module)
  - Thought process insights
"""

from __future__ import annotations
from models import OceanVector, PersonalityProfile
from thought_process import build_trait_levels, infer_thought_process


# ─────────────────────────────────────────────
# Archetype Table
# ─────────────────────────────────────────────
# Each archetype is matched by checking dominant/secondary trait patterns.
# Evaluated in priority order; first match wins.

_ARCHETYPES: list[dict] = [
    # (conditions: dict of trait→level_required, label)
    {
        "requires": {"O": "High", "E": "High", "C": "High"},
        "label": "The Visionary Leader",
        "summary": "Innovative, driven, and people-focused. You inspire others while turning creative ideas into structured action.",
    },
    {
        "requires": {"O": "High", "E": "High", "C": "Low"},
        "label": "The Creative Energizer",
        "summary": "Spontaneous and imaginative. You generate ideas at scale and energize teams — best paired with structured executors.",
    },
    {
        "requires": {"O": "High", "E": "Low", "C": "High"},
        "label": "The Deep Thinker",
        "summary": "Methodical and intellectually curious. You produce your best work in focused, independent environments.",
    },
    {
        "requires": {"O": "High", "E": "Low", "C": "Low"},
        "label": "The Free-Spirited Explorer",
        "summary": "Open and reflective. You seek meaning over structure, often thriving in creative or research-oriented paths.",
    },
    {
        "requires": {"E": "High", "C": "High", "A": "High"},
        "label": "The Collaborative Builder",
        "summary": "Sociable, disciplined, and empathetic. You excel at growing teams, managing people, and delivering outcomes together.",
    },
    {
        "requires": {"E": "High", "C": "High", "A": "Low"},
        "label": "The Strategic Driver",
        "summary": "Results-oriented and assertive. You lead with logic and leverage social confidence to drive competitive outcomes.",
    },
    {
        "requires": {"E": "Low", "C": "High", "A": "High"},
        "label": "The Quiet Achiever",
        "summary": "Reliable, empathetic, and precise. You deliver exceptional results behind the scenes while supporting others.",
    },
    {
        "requires": {"E": "Low", "C": "High", "N": "Low"},
        "label": "The Resilient Specialist",
        "summary": "Stable, disciplined, and detail-oriented. You thrive in high-precision roles requiring consistency and calm.",
    },
    {
        "requires": {"A": "High", "E": "High", "N": "High"},
        "label": "The Empathetic Connector",
        "summary": "Socially attuned and emotionally aware. You build deep relationships and advocate strongly for others.",
    },
    {
        "requires": {"N": "High", "O": "High"},
        "label": "The Sensitive Innovator",
        "summary": "Deeply feeling and creative. You process the world richly and channel emotional depth into original work.",
    },
    {
        "requires": {"N": "Low", "C": "High"},
        "label": "The Steady Operator",
        "summary": "Calm, organized, and dependable. You are the anchor in any team — unshakeable under pressure.",
    },
]

_FALLBACK_ARCHETYPE = {
    "label": "The Balanced Generalist",
    "summary": "Well-rounded with no extreme trait dominance. Versatile across many fields and adaptable to different environments.",
}


def _level(score: float) -> str:
    if score >= 0.67:
        return "High"
    elif score >= 0.34:
        return "Medium"
    return "Low"


def _match_archetype(vec: OceanVector) -> tuple[str, str]:
    levels = {
        "O": _level(vec.openness),
        "C": _level(vec.conscientiousness),
        "E": _level(vec.extraversion),
        "A": _level(vec.agreeableness),
        "N": _level(vec.neuroticism),
    }
    for archetype in _ARCHETYPES:
        if all(levels.get(t) == req for t, req in archetype["requires"].items()):
            return archetype["label"], archetype["summary"]
    return _FALLBACK_ARCHETYPE["label"], _FALLBACK_ARCHETYPE["summary"]


def _dominant_trait(vec: OceanVector) -> str:
    scores = {
        "Openness": vec.openness,
        "Conscientiousness": vec.conscientiousness,
        "Extraversion": vec.extraversion,
        "Agreeableness": vec.agreeableness,
        # Invert N so that "low neuroticism" shows as Emotional Stability
        "Emotional Stability": 1.0 - vec.neuroticism,
    }
    return max(scores, key=scores.__getitem__)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def build_profile(vec: OceanVector) -> PersonalityProfile:
    """Construct a full PersonalityProfile from an OceanVector."""
    type_label, _summary = _match_archetype(vec)
    traits = build_trait_levels(vec)
    thought_process = infer_thought_process(vec)
    dominant = _dominant_trait(vec)

    return PersonalityProfile(
        type_label=type_label,
        traits=traits,
        thought_process=thought_process,
        dominant_trait=dominant,
    )
