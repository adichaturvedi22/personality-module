"""
test_runner.py
--------------
Comprehensive test suite for the Career Counsellor module.

Covers:
  Unit tests  → each module independently
  Integration → full pipeline end-to-end
  Edge cases  → boundary answers, wrong inputs, consistency checks

Run with:
  python test_runner.py              # all tests, verbose
  python test_runner.py -v           # extra verbose
"""

from __future__ import annotations
import sys
import os
import json
import traceback
from typing import Callable

# Make sure imports work from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from models import AnswerPayload, TestSubmission, OceanVector
from questions import QUESTIONS, QUESTION_MAP, TRAIT_QUESTION_IDS, get_all_questions
from scoring_engine import score, _apply_reverse_coding, _normalize
from personality_profile import build_profile
from career_mapping import map_careers, get_riasec_scores
from thought_process import infer_thought_process, build_trait_levels
from orchestrator import run_pipeline
from data_logger import init_db, log_result, get_result, log_feedback


# ─────────────────────────────────────────────
# Test Harness
# ─────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results: list[tuple[str, bool, str]] = []


def test(name: str, fn: Callable) -> None:
    try:
        fn()
        results.append((name, True, ""))
        print(f"  {PASS}  {name}")
    except AssertionError as e:
        results.append((name, False, str(e)))
        print(f"  {FAIL}  {name}\n         → AssertionError: {e}")
    except Exception as e:
        results.append((name, False, traceback.format_exc()))
        print(f"  {FAIL}  {name}\n         → {type(e).__name__}: {e}")


def section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_answers(score_map: dict[int, int] | None = None, default: int = 3) -> list[AnswerPayload]:
    """Build 20 AnswerPayload objects. score_map overrides per question_id."""
    sm = score_map or {}
    return [AnswerPayload(question_id=i, score=sm.get(i, default)) for i in range(1, 21)]


def make_submission(score_map: dict[int, int] | None = None, default: int = 3,
                    user_id: str = "test-user-001") -> TestSubmission:
    return TestSubmission(user_id=user_id, answers=make_answers(score_map, default))


# ─────────────────────────────────────────────
# SECTION 1: Question Bank
# ─────────────────────────────────────────────

def run_question_tests():
    section("1. Question Bank")

    def test_count():
        assert len(QUESTIONS) == 20, f"Expected 20 questions, got {len(QUESTIONS)}"

    def test_unique_ids():
        ids = [q.id for q in QUESTIONS]
        assert len(ids) == len(set(ids)), "Duplicate question IDs found"
        assert set(ids) == set(range(1, 21)), "Question IDs must be 1–20"

    def test_traits_balanced():
        for trait, qids in TRAIT_QUESTION_IDS.items():
            assert len(qids) == 4, f"Trait {trait} has {len(qids)} questions, expected 4"

    def test_reverse_coding_present():
        reversed_qs = [q for q in QUESTIONS if q.reverse_coded]
        assert len(reversed_qs) >= 8, f"Expected ≥8 reverse-coded questions, got {len(reversed_qs)}"

    def test_serialization():
        qs = get_all_questions()
        assert len(qs) == 20
        for q in qs:
            assert "reverse_coded" not in q, "reverse_coded must not leak to client"
            assert "id" in q and "text" in q and "category" in q

    test("20 questions in bank", test_count)
    test("Unique sequential IDs 1–20", test_unique_ids)
    test("4 questions per OCEAN trait", test_traits_balanced)
    test("≥8 reverse-coded questions", test_reverse_coding_present)
    test("Serialization hides internal metadata", test_serialization)


# ─────────────────────────────────────────────
# SECTION 2: Scoring Engine
# ─────────────────────────────────────────────

def run_scoring_tests():
    section("2. Scoring Engine")

    def test_all_fives_neutral():
        """
        Answering 5 to ALL questions = 0.5 on every trait.
        Why: each trait has ~2 forward + 2 reverse questions.
        Forward 5 = 5pts; Reverse 5 → flipped to 1pt. They cancel → mid-range.
        This is intentional — it catches acquiescence bias.
        """
        answers = make_answers(default=5)
        vec = score(answers)
        for attr in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            val = getattr(vec, attr)
            assert 0.4 <= val <= 0.6, f"{attr}={val} (expected ~0.5 due to reverse coding)"

    def test_maximized_openness():
        """
        To maximize openness: 5 on forward Q (1,3), 1 on reverse Q (2,4).
        Adjusted: all 5 → raw=20 → normalized=1.0.
        """
        sm = {1: 5, 2: 1, 3: 5, 4: 1}   # openness maximized
        answers = make_answers(score_map=sm, default=3)
        vec = score(answers)
        assert vec.openness > 0.9, f"Expected openness~1.0, got {vec.openness}"

    def test_minimized_conscientiousness():
        """Min C: 1 on forward (5,7), 5 on reverse (6,8) → adjusted: all 1 → normalized=0.0."""
        sm = {5: 1, 6: 5, 7: 1, 8: 5}
        answers = make_answers(score_map=sm, default=3)
        vec = score(answers)
        assert vec.conscientiousness < 0.1, f"Expected C~0.0, got {vec.conscientiousness}"

    def test_all_ones_neutral():
        """All-1 mirrors all-5: forward=1pt, reverse flipped to 5pt → ~0.5 per trait."""
        answers = make_answers(default=1)
        vec = score(answers)
        for attr in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            val = getattr(vec, attr)
            assert 0.4 <= val <= 0.6, f"{attr}={val} (expected ~0.5)"

    def test_scores_in_range():
        answers = make_answers(default=3)
        vec = score(answers)
        for attr in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            val = getattr(vec, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of [0,1]"

    def test_confidence_true_consistency():
        """
        TRUE consistency: forward=5, reverse=1 for each trait → all adjusted=5 within trait.
        Standard deviation within trait = 0 → confidence = 1.0.
        """
        sm = {}
        # O: forward 1,3 = 5; reverse 2,4 = 1
        sm.update({1: 5, 2: 1, 3: 5, 4: 1})
        # C: forward 5,7 = 5; reverse 6,8 = 1
        sm.update({5: 5, 6: 1, 7: 5, 8: 1})
        # E: forward 9,11 = 5; reverse 10,12 = 1
        sm.update({9: 5, 10: 1, 11: 5, 12: 1})
        # A: forward 13,15 = 5; reverse 14,16 = 1
        sm.update({13: 5, 14: 1, 15: 5, 16: 1})
        # N: forward 17,19 = 5; reverse 18,20 = 1
        sm.update({17: 5, 18: 1, 19: 5, 20: 1})
        answers = make_answers(score_map=sm)
        vec = score(answers)
        assert vec.confidence_score >= 0.95, f"Expected confidence ~1.0, got {vec.confidence_score}"

    def test_confidence_low_inconsistency():
        """
        Low consistency: answers jump 1→5→1→5 within the same trait's questions.
        This creates max stdev within each trait → low confidence.
        """
        sm = {}
        # O: Q1=1, Q2=5(raw, forward-adj=5), Q3=1, Q4=5(raw,forward-adj=1 after reverse→1) 
        # Simplest way: alternate extremes on forward questions within each trait
        sm.update({1: 1, 2: 5, 3: 5, 4: 1})   # O: adjusted = 1, 1, 5, 5 → stdev=2
        sm.update({5: 1, 6: 1, 7: 5, 8: 5})   # C: adjusted = 1, 5, 5, 1 → stdev=2
        sm.update({9: 1, 10: 1, 11: 5, 12: 5})
        sm.update({13: 1, 14: 1, 15: 5, 16: 5})
        sm.update({17: 1, 18: 1, 19: 5, 20: 5})
        answers = make_answers(score_map=sm)
        vec = score(answers)
        assert vec.confidence_score < 0.6, f"Expected lower confidence, got {vec.confidence_score}"

    def test_normalize_bounds():
        assert _normalize(4, 4, 20) == 0.0
        assert _normalize(20, 4, 20) == 1.0
        assert _normalize(12, 4, 20) == 0.5

    def test_reverse_coding():
        q = QUESTION_MAP[2]  # reverse coded O
        assert q.reverse_coded is True
        assert _apply_reverse_coding(2, 5) == 1
        assert _apply_reverse_coding(2, 1) == 5
        q_fwd = QUESTION_MAP[1]  # forward coded
        assert _apply_reverse_coding(1, 4) == 4

    test("All-5 answers → ~0.5 per trait (reverse coding neutralizes)", test_all_fives_neutral)
    test("All-1 answers → ~0.5 per trait (same cancellation)", test_all_ones_neutral)
    test("Maximize openness: 5 on forward, 1 on reverse → score ~1.0", test_maximized_openness)
    test("Minimize conscientiousness: 1 on forward, 5 on reverse → score ~0.0", test_minimized_conscientiousness)
    test("All scores in [0.0, 1.0]", test_scores_in_range)
    test("Directionally consistent answers → confidence ~1.0", test_confidence_true_consistency)
    test("Contradictory answers within traits → low confidence", test_confidence_low_inconsistency)
    test("Normalize() boundary values", test_normalize_bounds)
    test("Reverse coding flips 1↔5", test_reverse_coding)


# ─────────────────────────────────────────────
# SECTION 3: Thought Process
# ─────────────────────────────────────────────

def run_thought_process_tests():
    section("3. Thought Process Inference")

    def test_all_fields_populated():
        vec = OceanVector(openness=0.8, conscientiousness=0.7, extraversion=0.9,
                          agreeableness=0.6, neuroticism=0.2, confidence_score=0.9)
        tp = infer_thought_process(vec)
        assert tp.decision_style
        assert tp.work_style
        assert tp.social_behavior
        assert len(tp.strengths) > 0
        assert len(tp.growth_areas) > 0

    def test_trait_levels_all_five():
        vec = OceanVector(openness=0.8, conscientiousness=0.7, extraversion=0.9,
                          agreeableness=0.6, neuroticism=0.2, confidence_score=0.9)
        traits = build_trait_levels(vec)
        assert set(traits.keys()) == {"Openness", "Conscientiousness",
                                       "Extraversion", "Agreeableness", "Neuroticism"}
        for name, tl in traits.items():
            assert tl.label in {"Low", "Medium", "High"}, f"{name} has invalid label"
            assert tl.description

    def test_high_n_growth_areas():
        vec = OceanVector(openness=0.5, conscientiousness=0.5, extraversion=0.5,
                          agreeableness=0.5, neuroticism=0.9, confidence_score=0.8)
        tp = infer_thought_process(vec)
        assert any("emotional" in g.lower() or "stress" in g.lower()
                   for g in tp.growth_areas), f"Expected stress-related growth area, got {tp.growth_areas}"

    test("All thought process fields populated", test_all_fields_populated)
    test("All 5 trait levels present with valid labels", test_trait_levels_all_five)
    test("High N → stress management in growth areas", test_high_n_growth_areas)


# ─────────────────────────────────────────────
# SECTION 4: Personality Profile
# ─────────────────────────────────────────────

def run_profile_tests():
    section("4. Personality Profile Builder")

    def test_profile_has_all_fields():
        vec = OceanVector(openness=0.8, conscientiousness=0.8, extraversion=0.8,
                          agreeableness=0.6, neuroticism=0.2, confidence_score=0.9)
        p = build_profile(vec)
        assert p.type_label, "type_label is empty"
        assert p.dominant_trait, "dominant_trait is empty"
        assert p.traits
        assert p.thought_process

    def test_visionary_leader_archetype():
        """High O + E + C → Visionary Leader."""
        vec = OceanVector(openness=0.9, conscientiousness=0.9, extraversion=0.9,
                          agreeableness=0.5, neuroticism=0.1, confidence_score=1.0)
        p = build_profile(vec)
        assert p.type_label == "The Visionary Leader", f"Got: {p.type_label}"

    def test_fallback_archetype():
        """Medium scores everywhere → Balanced Generalist."""
        vec = OceanVector(openness=0.5, conscientiousness=0.5, extraversion=0.5,
                          agreeableness=0.5, neuroticism=0.5, confidence_score=0.6)
        p = build_profile(vec)
        assert p.type_label == "The Balanced Generalist", f"Got: {p.type_label}"

    test("Profile has all required fields", test_profile_has_all_fields)
    test("High O+C+E → Visionary Leader archetype", test_visionary_leader_archetype)
    test("Medium scores → Balanced Generalist fallback", test_fallback_archetype)


# ─────────────────────────────────────────────
# SECTION 5: Career Mapping
# ─────────────────────────────────────────────

def run_career_tests():
    section("5. Career Mapping Engine")

    def test_returns_top5():
        vec = OceanVector(openness=0.8, conscientiousness=0.7, extraversion=0.8,
                          agreeableness=0.6, neuroticism=0.2, confidence_score=0.9)
        careers = map_careers(vec, top_n=5)
        assert len(careers) == 5, f"Expected 5, got {len(careers)}"

    def test_ranked_correctly():
        vec = OceanVector(openness=0.8, conscientiousness=0.7, extraversion=0.8,
                          agreeableness=0.6, neuroticism=0.2, confidence_score=0.9)
        careers = map_careers(vec, top_n=5)
        for i, c in enumerate(careers):
            assert c.rank == i + 1, f"Rank mismatch at position {i}: rank={c.rank}"
        scores = [c.match_score for c in careers]
        assert scores == sorted(scores, reverse=True), "Careers not sorted by score"

    def test_scores_in_range():
        vec = OceanVector(openness=0.8, conscientiousness=0.7, extraversion=0.8,
                          agreeableness=0.6, neuroticism=0.2, confidence_score=0.9)
        careers = map_careers(vec)
        for c in careers:
            assert 0.0 <= c.match_score <= 1.0, f"{c.field}: score {c.match_score} out of range"

    def test_high_creative_gets_creative_career():
        """High O, low C → should surface creative/artistic fields."""
        vec = OceanVector(openness=1.0, conscientiousness=0.1, extraversion=0.7,
                          agreeableness=0.6, neuroticism=0.3, confidence_score=0.9)
        careers = map_careers(vec, top_n=5)
        creative_fields = {"UX Design & Creative Technology", "Media, Journalism & Content Creation",
                           "Marketing & Brand Strategy", "Entrepreneurship & Startups"}
        names = {c.field for c in careers}
        assert names & creative_fields, f"No creative field in top 5: {names}"

    def test_riasec_scores_sum_to_expected():
        vec = OceanVector(openness=0.5, conscientiousness=0.5, extraversion=0.5,
                          agreeableness=0.5, neuroticism=0.5, confidence_score=0.8)
        riasec = get_riasec_scores(vec)
        assert set(riasec.keys()) == {"R", "I", "A", "S", "E", "C"}
        for code, s in riasec.items():
            assert 0.0 <= s <= 1.0, f"RIASEC {code}={s} out of range"

    test("Returns exactly top-N careers", test_returns_top5)
    test("Careers sorted by match_score descending", test_ranked_correctly)
    test("All match_scores in [0.0, 1.0]", test_scores_in_range)
    test("High Openness surfaces creative fields in top 5", test_high_creative_gets_creative_career)
    test("RIASEC scores: all 6 codes, all in [0,1]", test_riasec_scores_sum_to_expected)


# ─────────────────────────────────────────────
# SECTION 6: Input Validation
# ─────────────────────────────────────────────

def run_validation_tests():
    section("6. Input Validation & Edge Cases")

    def test_wrong_answer_count():
        from pydantic import ValidationError
        try:
            TestSubmission(user_id="x", answers=[AnswerPayload(question_id=1, score=3)])
            assert False, "Should have raised ValidationError"
        except (ValidationError, ValueError):
            pass

    def test_score_out_of_range():
        from pydantic import ValidationError
        try:
            AnswerPayload(question_id=1, score=6)
            assert False, "Should have raised ValidationError"
        except (ValidationError, ValueError):
            pass

    def test_duplicate_question_ids():
        from pydantic import ValidationError
        duped = [AnswerPayload(question_id=1, score=3)] * 20
        try:
            TestSubmission(user_id="x", answers=duped)
            assert False, "Should have raised ValidationError for duplicate IDs"
        except (ValidationError, ValueError):
            pass

    def test_all_neutral_answers():
        """Neutral answers should produce mid-range scores, not crash."""
        result = run_pipeline(make_submission(default=3))
        v = result.ocean_vector
        for attr in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            val = getattr(v, attr)
            assert 0.3 < val < 0.7, f"{attr}={val} not in mid range for neutral answers"

    test("19 answers → ValidationError", test_wrong_answer_count)
    test("Score=6 → ValidationError (max is 5)", test_score_out_of_range)
    test("Duplicate question IDs → ValidationError", test_duplicate_question_ids)
    test("All-neutral answers → mid-range scores", test_all_neutral_answers)


# ─────────────────────────────────────────────
# SECTION 7: Full Pipeline (Integration)
# ─────────────────────────────────────────────

def run_integration_tests():
    section("7. Full Pipeline Integration")

    def test_pipeline_complete_output():
        result = run_pipeline(make_submission(default=4, user_id="integration-test-001"))
        assert result.user_id == "integration-test-001"
        assert result.ocean_vector
        assert result.personality_profile
        assert result.personality_profile.type_label
        assert result.personality_profile.thought_process
        assert len(result.career_recommendations) == 5
        assert result.timestamp

    def test_pipeline_reproducible():
        """Same answers → same result (deterministic)."""
        sm = {i: ((i % 5) + 1) for i in range(1, 21)}
        r1 = run_pipeline(make_submission(score_map=sm, user_id="repro-1"))
        r2 = run_pipeline(make_submission(score_map=sm, user_id="repro-2"))
        assert r1.ocean_vector.openness == r2.ocean_vector.openness
        assert r1.career_recommendations[0].field == r2.career_recommendations[0].field

    def test_result_serializable():
        result = run_pipeline(make_submission(default=3, user_id="serial-test-001"))
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["user_id"] == "serial-test-001"
        assert "ocean_vector" in parsed
        assert "career_recommendations" in parsed

    def test_introvert_analyst_profile():
        """High O+C, Low E, Low N → Deep Thinker / data/engineering careers."""
        sm = {}
        # Force High O (Q1,3=5; Q2,4=1 since reverse coded → effective 5)
        sm.update({1: 5, 2: 1, 3: 5, 4: 1})
        # Force High C (Q5,7=5; Q6,8=1)
        sm.update({5: 5, 6: 1, 7: 5, 8: 1})
        # Force Low E (Q9,11=1; Q10,12=5 since reverse coded → effective 1)
        sm.update({9: 1, 10: 5, 11: 1, 12: 5})
        # Medium A (all 3)
        sm.update({13: 3, 14: 3, 15: 3, 16: 3})
        # Low N (Q17,19=1; Q18,20=5 since reverse coded → effective 1)
        sm.update({17: 1, 18: 5, 19: 1, 20: 5})

        result = run_pipeline(make_submission(score_map=sm, user_id="introvert-analyst"))
        v = result.ocean_vector
        assert v.openness > 0.6, f"Expected high O, got {v.openness}"
        assert v.extraversion < 0.4, f"Expected low E, got {v.extraversion}"

        top_careers = {c.field for c in result.career_recommendations[:3]}
        analytical = {
            "Data Science & AI Research", "Software Engineering & Architecture",
            "Management Consulting", "Finance & Investment",
            "Education & Academic Research",   # High O + I type maps here too
        }
        assert top_careers & analytical, f"Expected analytical career in top 3, got: {top_careers}"

    test("Pipeline produces complete, valid output", test_pipeline_complete_output)
    test("Same answers → same result (deterministic)", test_pipeline_reproducible)
    test("Result is fully JSON-serializable", test_result_serializable)
    test("Introvert-analyst profile → analytical careers in top 3", test_introvert_analyst_profile)


# ─────────────────────────────────────────────
# SECTION 8: Data Logger
# ─────────────────────────────────────────────

def run_logger_tests():
    section("8. Data Logger (SQLite)")

    def test_init_db():
        init_db()   # must not raise

    def test_log_and_retrieve():
        result = run_pipeline(make_submission(default=4, user_id="logger-test-001"))
        stored = get_result("logger-test-001")
        assert stored is not None, "Stored result not found"
        assert stored["user_id"] == "logger-test-001"

    def test_feedback_logging():
        log_feedback("logger-test-001", rating=4,
                     chosen_career="Data Science & AI Research",
                     comment="Very accurate!")

    def test_missing_user_returns_none():
        result = get_result("nonexistent-user-xyz-999")
        assert result is None

    test("init_db() completes without error", test_init_db)
    test("Log result → retrieve by user_id", test_log_and_retrieve)
    test("Feedback logging without error", test_feedback_logging)
    test("Missing user_id → None (not exception)", test_missing_user_returns_none)


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  🧠 CAREER COUNSELLOR MODULE — TEST SUITE")
    print("=" * 55)

    run_question_tests()
    run_scoring_tests()
    run_thought_process_tests()
    run_profile_tests()
    run_career_tests()
    run_validation_tests()
    run_integration_tests()
    run_logger_tests()

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed

    print(f"\n{'=' * 55}")
    print(f"  RESULTS: {passed}/{len(results)} passed", end="")
    if failed:
        print(f"   |   {failed} failed ← see above")
    else:
        print("   🎉 All tests passed!")
    print("=" * 55)

    if failed:
        print("\nFailed tests:")
        for name, ok, err in results:
            if not ok:
                print(f"  ❌ {name}")
                if err:
                    print(f"     {err.splitlines()[0]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
