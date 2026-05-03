"""
orchestrator.py
---------------
Single pipeline entry point. Wires all modules together.

Usage:
    from orchestrator import run_pipeline
    result = run_pipeline(submission)
"""

from __future__ import annotations
from models import TestSubmission, TestResult
from scoring_engine import score
from personality_profile import build_profile
from career_mapping import map_careers
from data_logger import init_db, log_result


def run_pipeline(submission: TestSubmission, top_careers: int = 5) -> TestResult:
    """
    Full pipeline:
      Answers → OCEAN Vector → Personality Profile → Career Map → Logged Result
    """
    # Step 1: OCEAN scoring
    ocean_vector = score(submission.answers)

    # Step 2: Personality profile (includes thought process)
    personality = build_profile(ocean_vector)

    # Step 3: Career recommendations
    careers = map_careers(ocean_vector, top_n=top_careers)

    # Step 4: Assemble result
    result = TestResult(
        user_id=submission.user_id,
        ocean_vector=ocean_vector,
        personality_profile=personality,
        career_recommendations=careers,
        raw_answers=submission.answers,
    )

    # Step 5: Log to DB (non-blocking; errors here shouldn't fail the user response)
    try:
        init_db()
        log_result(result)
    except Exception as e:
        print(f"[WARN] Data logging failed: {e}")

    return result
