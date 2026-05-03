"""
scoring_engine.py
-----------------
Converts raw Likert answers → normalized OCEAN vector (0–1).

Pipeline:
  1. Apply reverse coding where flagged
  2. Sum scores per trait
  3. Normalize to [0, 1]
  4. Compute confidence score (internal consistency via split-half correlation)
"""

from __future__ import annotations
import statistics
from models import AnswerPayload, OceanVector
from questions import QUESTION_MAP, TRAIT_QUESTION_IDS


# Likert scale bounds
LIKERT_MIN = 1
LIKERT_MAX = 5
LIKERT_RANGE = LIKERT_MAX - LIKERT_MIN          # 4
QUESTIONS_PER_TRAIT = 4
RAW_MIN_PER_TRAIT = QUESTIONS_PER_TRAIT * LIKERT_MIN    # 4
RAW_MAX_PER_TRAIT = QUESTIONS_PER_TRAIT * LIKERT_MAX    # 20


def _apply_reverse_coding(question_id: int, raw_score: int) -> int:
    """Flip score for reverse-coded questions: 6 - raw."""
    q = QUESTION_MAP[question_id]
    return (LIKERT_MAX + LIKERT_MIN) - raw_score if q.reverse_coded else raw_score


def _normalize(raw: float, raw_min: float = RAW_MIN_PER_TRAIT,
               raw_max: float = RAW_MAX_PER_TRAIT) -> float:
    """Map [raw_min, raw_max] → [0.0, 1.0] and clamp."""
    if raw_max == raw_min:
        return 0.5
    normalized = (raw - raw_min) / (raw_max - raw_min)
    return round(max(0.0, min(1.0, normalized)), 4)


def _compute_confidence(answers: list[AnswerPayload]) -> float:
    """
    Confidence score ∈ [0, 1] based on intra-trait consistency.

    For each trait, split its 4 questions into two halves and measure
    how consistent the respondent was within each trait.
    A perfectly consistent respondent → 1.0; random answers → ~0.5.

    Method: for each trait, compute the standard deviation of the
    4 (adjusted) scores. Low variance = high confidence.
    """
    answer_map = {a.question_id: a.score for a in answers}
    trait_variances: list[float] = []

    for trait, qids in TRAIT_QUESTION_IDS.items():
        adjusted = [_apply_reverse_coding(qid, answer_map[qid]) for qid in qids]
        # stdev of 0 means perfect consistency
        if len(adjusted) > 1:
            sd = statistics.stdev(adjusted)
            # max possible stdev on a 1–5 scale with 4 items ≈ 2.0
            trait_variances.append(sd / 2.0)

    if not trait_variances:
        return 1.0

    avg_variance = statistics.mean(trait_variances)
    confidence = 1.0 - avg_variance
    return round(max(0.0, min(1.0, confidence)), 4)


def score(answers: list[AnswerPayload]) -> OceanVector:
    """
    Main entry point.
    Takes 20 answers, returns an OceanVector.
    """
    answer_map: dict[int, int] = {a.question_id: a.score for a in answers}
    trait_scores: dict[str, float] = {}

    for trait, qids in TRAIT_QUESTION_IDS.items():
        raw_total = sum(
            _apply_reverse_coding(qid, answer_map[qid])
            for qid in qids
        )
        trait_scores[trait] = _normalize(raw_total)

    confidence = _compute_confidence(answers)

    return OceanVector(
        openness=trait_scores["O"],
        conscientiousness=trait_scores["C"],
        extraversion=trait_scores["E"],
        agreeableness=trait_scores["A"],
        neuroticism=trait_scores["N"],
        confidence_score=confidence,
    )
