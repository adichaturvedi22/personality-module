"""
questions.py
------------
20 MCQ questions (Likert 1–5) designed to capture maximum OCEAN signal.
Each trait has 4 questions; ~50% are reverse-coded to reduce acquiescence bias.

reverse_coded=True means: effective_score = 6 - raw_score
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: int
    text: str
    trait: str          # "O" | "C" | "E" | "A" | "N"
    reverse_coded: bool
    category: str       # human-readable trait name


# ─────────────────────────────────────────────
# Question Bank
# ─────────────────────────────────────────────

QUESTIONS: list[Question] = [

    # ── OPENNESS (O) ──────────────────────────────
    Question(
        id=1,
        text="I enjoy exploring new ideas and perspectives, even if they challenge what I already believe.",
        trait="O", reverse_coded=False, category="Openness"
    ),
    Question(
        id=2,
        text="I prefer sticking to familiar methods rather than experimenting with new approaches.",
        trait="O", reverse_coded=True, category="Openness"
    ),
    Question(
        id=3,
        text="I find abstract concepts and theoretical thinking genuinely interesting.",
        trait="O", reverse_coded=False, category="Openness"
    ),
    Question(
        id=4,
        text="I rarely seek out creative or artistic experiences in my free time.",
        trait="O", reverse_coded=True, category="Openness"
    ),

    # ── CONSCIENTIOUSNESS (C) ─────────────────────
    Question(
        id=5,
        text="I always plan my work carefully and follow through on my commitments.",
        trait="C", reverse_coded=False, category="Conscientiousness"
    ),
    Question(
        id=6,
        text="I often leave tasks unfinished and move on to something else before completing them.",
        trait="C", reverse_coded=True, category="Conscientiousness"
    ),
    Question(
        id=7,
        text="I set clear goals for myself and track my progress regularly.",
        trait="C", reverse_coded=False, category="Conscientiousness"
    ),
    Question(
        id=8,
        text="I tend to act on impulse rather than thinking things through carefully.",
        trait="C", reverse_coded=True, category="Conscientiousness"
    ),

    # ── EXTRAVERSION (E) ──────────────────────────
    Question(
        id=9,
        text="I feel energized after spending time with a large group of people.",
        trait="E", reverse_coded=False, category="Extraversion"
    ),
    Question(
        id=10,
        text="I prefer spending my evenings alone or with one close friend rather than at social events.",
        trait="E", reverse_coded=True, category="Extraversion"
    ),
    Question(
        id=11,
        text="I am usually the first to start conversations or introduce myself to new people.",
        trait="E", reverse_coded=False, category="Extraversion"
    ),
    Question(
        id=12,
        text="I find it draining to be the center of attention in group situations.",
        trait="E", reverse_coded=True, category="Extraversion"
    ),

    # ── AGREEABLENESS (A) ─────────────────────────
    Question(
        id=13,
        text="I genuinely care about the well-being of others, even strangers.",
        trait="A", reverse_coded=False, category="Agreeableness"
    ),
    Question(
        id=14,
        text="I can be competitive and am willing to argue firmly for my point of view.",
        trait="A", reverse_coded=True, category="Agreeableness"
    ),
    Question(
        id=15,
        text="I find it easy to trust others and assume good intentions.",
        trait="A", reverse_coded=False, category="Agreeableness"
    ),
    Question(
        id=16,
        text="I sometimes find it hard to be sympathetic when people complain about their problems.",
        trait="A", reverse_coded=True, category="Agreeableness"
    ),

    # ── NEUROTICISM (N) ───────────────────────────
    Question(
        id=17,
        text="I often worry about things that might go wrong, even when things are going well.",
        trait="N", reverse_coded=False, category="Neuroticism"
    ),
    Question(
        id=18,
        text="I generally stay calm and composed even in stressful or uncertain situations.",
        trait="N", reverse_coded=True, category="Neuroticism"
    ),
    Question(
        id=19,
        text="My mood can change noticeably throughout the day based on events around me.",
        trait="N", reverse_coded=False, category="Neuroticism"
    ),
    Question(
        id=20,
        text="I rarely feel overwhelmed or anxious when facing challenges or deadlines.",
        trait="N", reverse_coded=True, category="Neuroticism"
    ),
]

# Fast lookup by id
QUESTION_MAP: dict[int, Question] = {q.id: q for q in QUESTIONS}

# Trait grouping for scoring
TRAIT_QUESTION_IDS: dict[str, list[int]] = {}
for q in QUESTIONS:
    TRAIT_QUESTION_IDS.setdefault(q.trait, []).append(q.id)


def get_question(question_id: int) -> Question:
    if question_id not in QUESTION_MAP:
        raise ValueError(f"Unknown question_id: {question_id}")
    return QUESTION_MAP[question_id]


def get_all_questions() -> list[dict]:
    """Serialize questions for API response (text + id only — no metadata leaked)."""
    return [
        {
            "id": q.id,
            "text": q.text,
            "category": q.category,
            "scale": {"min": 1, "max": 5, "labels": {
                "1": "Strongly Disagree",
                "2": "Disagree",
                "3": "Neutral",
                "4": "Agree",
                "5": "Strongly Agree",
            }},
        }
        for q in QUESTIONS
    ]
