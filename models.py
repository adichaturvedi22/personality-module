"""
models.py
---------
Pydantic schemas for request/response validation and internal data structures.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import uuid
from datetime import datetime


# ─────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────

class AnswerPayload(BaseModel):
    """Single answer to one MCQ question."""
    question_id: int = Field(..., ge=1, le=20, description="Question number (1–20)")
    score: int = Field(..., ge=1, le=5, description="Likert scale response (1–5)")
    time_taken_ms: Optional[int] = Field(None, description="Time taken in ms (future ML hook)")


class TestSubmission(BaseModel):
    """Full test submission from the frontend."""
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    answers: list[AnswerPayload]
    session_metadata: Optional[dict] = Field(default=None, description="Browser/session info")

    @field_validator("answers")
    @classmethod
    def validate_answer_count(cls, v: list[AnswerPayload]) -> list[AnswerPayload]:
        if len(v) != 20:
            raise ValueError(f"Expected exactly 20 answers, got {len(v)}")
        ids = {a.question_id for a in v}
        if ids != set(range(1, 21)):
            raise ValueError("Answers must cover all question IDs 1–20")
        return v


# ─────────────────────────────────────────────
# Internal Data Structures
# ─────────────────────────────────────────────

class OceanVector(BaseModel):
    """Normalized OCEAN trait scores (0.0–1.0)."""
    openness: float = Field(..., ge=0.0, le=1.0)
    conscientiousness: float = Field(..., ge=0.0, le=1.0)
    extraversion: float = Field(..., ge=0.0, le=1.0)
    agreeableness: float = Field(..., ge=0.0, le=1.0)
    neuroticism: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Response consistency (0–1)")

    def as_dict(self) -> dict[str, float]:
        return {
            "O": self.openness,
            "C": self.conscientiousness,
            "E": self.extraversion,
            "A": self.agreeableness,
            "N": self.neuroticism,
        }


class TraitLevel(BaseModel):
    label: str         # "Low" | "Medium" | "High"
    score: float       # raw 0–1
    description: str


class ThoughtProcessInsights(BaseModel):
    decision_style: str
    work_style: str
    social_behavior: str
    strengths: list[str]
    growth_areas: list[str]


class PersonalityProfile(BaseModel):
    type_label: str
    traits: dict[str, TraitLevel]
    thought_process: ThoughtProcessInsights
    dominant_trait: str


class CareerMatch(BaseModel):
    rank: int
    field: str
    riasec_codes: list[str]
    match_score: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    example_roles: list[str]


# ─────────────────────────────────────────────
# API Response Models
# ─────────────────────────────────────────────

class TestResult(BaseModel):
    user_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    ocean_vector: OceanVector
    personality_profile: PersonalityProfile
    career_recommendations: list[CareerMatch]
    raw_answers: list[AnswerPayload]
