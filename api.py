"""
api.py
------
FastAPI REST layer.

Endpoints:
  GET  /questions                   → returns the 20 MCQs
  POST /submit-test                 → runs full pipeline, returns TestResult
  GET  /results/{user_id}          → fetch a stored result
  GET  /career-recommendations      → top global career fields (static reference)
  POST /feedback                    → submit post-result user feedback
  GET  /health                      → health check

Run with:
  uvicorn api:app --reload --port 8000
"""

from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import TestSubmission, TestResult
from questions import get_all_questions
from career_mapping import CAREER_FIELDS
from orchestrator import run_pipeline
from data_logger import init_db, get_result, log_feedback
from college_recommender import recommend_colleges, CollegeMatch

app = FastAPI(
    title="AI Career Counsellor — Personality Engine",
    description="OCEAN-based personality assessment and career recommendation API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    init_db()


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class FeedbackPayload(BaseModel):
    user_id: str
    rating: int                          # 1–5
    chosen_career: str | None = None
    comment: str | None = None


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/questions")
def get_questions():
    """Return all 20 MCQs (no reverse-coding metadata exposed)."""
    return {
        "count": 20,
        "scale": "Likert 1–5 (1=Strongly Disagree, 5=Strongly Agree)",
        "questions": get_all_questions(),
    }


@app.post("/submit-test", response_model=TestResult)
def submit_test(payload: TestSubmission):
    """
    Run the full pipeline on 20 answers.
    Returns personality profile + career recommendations.
    """
    try:
        return run_pipeline(payload)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")


@app.get("/results/{user_id}")
def get_results(user_id: str):
    """Retrieve a previously stored result by user_id."""
    data = get_result(user_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"No result found for user_id: {user_id}")
    return data


@app.get("/career-recommendations")
def list_career_fields():
    """Static reference: all career fields with RIASEC codes and example roles."""
    return [
        {
            "field": cf.name,
            "riasec_codes": cf.riasec_codes,
            "example_roles": cf.example_roles,
        }
        for cf in CAREER_FIELDS
    ]


@app.get("/college-recommendations/{user_id}", response_model=list[CollegeMatch])
def college_recommendations(
    user_id: str,
    state: str | None = None,
    top_n: int = 10,
    require_naac: bool = False,
):
    """
    Recommend Indian colleges based on a completed personality test result.
    Uses OCEAN vector + top career matches to rank colleges by:
      - Stream relevance (40%)
      - NAAC quality grade (35%)
      - Personality-environment fit (25%)

    Args:
        user_id      : from a completed /submit-test call
        state        : optional filter e.g. "Uttar Pradesh"
        top_n        : how many colleges to return (default 10, max 50)
        require_naac : if true, only return NAAC-graded colleges
    """
    stored = get_result(user_id)
    if not stored:
        raise HTTPException(status_code=404, detail=f"No test result found for user_id: {user_id}")

    ocean = stored["ocean_vector"]
    careers = stored["career_recommendations"]

    try:
        top_n = min(top_n, 50)
        colleges = recommend_colleges(
            ocean=ocean,
            career_matches=careers,
            state=state,
            top_n=top_n,
            require_naac=require_naac,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not colleges:
        raise HTTPException(
            status_code=404,
            detail=f"No colleges found for state='{state}'. Try without a state filter."
        )

    return colleges


@app.post("/feedback")
def submit_feedback(payload: FeedbackPayload):
    """Log user's post-assessment feedback. Critical for future ML training."""
    if not 1 <= payload.rating <= 5:
        raise HTTPException(status_code=422, detail="Rating must be between 1 and 5.")
    log_feedback(
        user_id=payload.user_id,
        rating=payload.rating,
        chosen_career=payload.chosen_career,
        comment=payload.comment,
    )
    return {"status": "feedback logged", "user_id": payload.user_id}
