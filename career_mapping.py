"""
career_mapping.py
-----------------
Maps OCEAN personality vector → RIASEC interest codes → ranked career fields.

Pipeline:
  1. Score each RIASEC type against the OCEAN vector (weighted formula)
  2. For each career field, compute a composite match score
  3. Return top-N ranked CareerMatch objects

RIASEC Types:
  R = Realistic     (doers)
  I = Investigative (thinkers)
  A = Artistic      (creators)
  S = Social        (helpers)
  E = Enterprising  (persuaders)
  C = Conventional  (organizers)

OCEAN → RIASEC mapping is derived from empirical research
(Larson et al., 2002; Barrick et al., 2003).
"""

from __future__ import annotations
from dataclasses import dataclass
from models import OceanVector, CareerMatch


# ─────────────────────────────────────────────
# RIASEC Weight Matrix
# ─────────────────────────────────────────────
# Each RIASEC type is a weighted combination of OCEAN traits.
# Weights are signed floats (positive = promotes, negative = suppresses).
# Rows: RIASEC types  |  Columns: O, C, E, A, N

@dataclass(frozen=True)
class RiasecType:
    code: str
    name: str
    weights: dict[str, float]   # OCEAN trait → weight


RIASEC_TYPES: list[RiasecType] = [
    RiasecType(
        code="R", name="Realistic",
        weights={"O": -0.2, "C": 0.4, "E": -0.1, "A": -0.1, "N": -0.2},
    ),
    RiasecType(
        code="I", name="Investigative",
        weights={"O": 0.5, "C": 0.4, "E": -0.3, "A": 0.0, "N": -0.1},
    ),
    RiasecType(
        code="A", name="Artistic",
        weights={"O": 0.6, "C": -0.3, "E": 0.1, "A": 0.1, "N": 0.1},
    ),
    RiasecType(
        code="S", name="Social",
        weights={"O": 0.1, "C": 0.0, "E": 0.4, "A": 0.6, "N": 0.0},
    ),
    RiasecType(
        code="E", name="Enterprising",
        weights={"O": 0.2, "C": 0.3, "E": 0.6, "A": -0.1, "N": -0.2},
    ),
    RiasecType(
        code="C", name="Conventional",
        weights={"O": -0.3, "C": 0.6, "E": 0.0, "A": 0.1, "N": -0.2},
    ),
]


# ─────────────────────────────────────────────
# Career Field Database
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class CareerField:
    name: str
    riasec_codes: list[str]          # Primary RIASEC codes for this field
    example_roles: list[str]
    explanation_template: str        # Uses {trait_summary} placeholder


CAREER_FIELDS: list[CareerField] = [
    CareerField(
        name="Product Management",
        riasec_codes=["E", "I", "C"],
        example_roles=["Product Manager", "Associate PM", "Chief Product Officer"],
        explanation_template=(
            "Your blend of strategic thinking and people skills makes you a natural at "
            "bridging technology and business. PMs who thrive share your trait profile."
        ),
    ),
    CareerField(
        name="Software Engineering & Architecture",
        riasec_codes=["I", "R", "C"],
        example_roles=["Software Engineer", "Solutions Architect", "Backend Developer"],
        explanation_template=(
            "Your analytical depth and attention to detail align strongly with engineering roles "
            "that demand systematic problem-solving and precision."
        ),
    ),
    CareerField(
        name="Data Science & AI Research",
        riasec_codes=["I", "C", "R"],
        example_roles=["Data Scientist", "ML Engineer", "Research Analyst"],
        explanation_template=(
            "Your investigative curiosity and comfort with abstraction suit data-intensive roles "
            "that require extracting insight from complexity."
        ),
    ),
    CareerField(
        name="UX Design & Creative Technology",
        riasec_codes=["A", "I", "S"],
        example_roles=["UX Designer", "Creative Director", "Interaction Designer"],
        explanation_template=(
            "Your openness and empathy enable you to design experiences that feel human. "
            "Creative technology roles reward your blend of imagination and user-centricity."
        ),
    ),
    CareerField(
        name="Entrepreneurship & Startups",
        riasec_codes=["E", "A", "I"],
        example_roles=["Founder", "Co-founder", "Growth Lead", "Startup Operator"],
        explanation_template=(
            "High openness and enterprising energy are core founder traits. "
            "You're energized by ambiguity and motivated by building something new."
        ),
    ),
    CareerField(
        name="Marketing & Brand Strategy",
        riasec_codes=["A", "E", "S"],
        example_roles=["Brand Strategist", "Digital Marketer", "Content Lead", "CMO"],
        explanation_template=(
            "Your creative instincts and social awareness make you effective at crafting "
            "narratives that resonate and campaigns that drive behaviour."
        ),
    ),
    CareerField(
        name="People & Organizational Psychology",
        riasec_codes=["S", "I", "A"],
        example_roles=["HR Business Partner", "Organizational Psychologist", "L&D Manager"],
        explanation_template=(
            "Your empathy and curiosity about human behaviour equip you well for roles "
            "that develop people, culture, and organizational effectiveness."
        ),
    ),
    CareerField(
        name="Management Consulting",
        riasec_codes=["E", "C", "I"],
        example_roles=["Strategy Consultant", "Management Analyst", "Business Advisor"],
        explanation_template=(
            "Your structured thinking and confident communication align with consulting, "
            "where you diagnose complex problems and guide senior decision-makers."
        ),
    ),
    CareerField(
        name="Education & Academic Research",
        riasec_codes=["S", "I", "A"],
        example_roles=["University Lecturer", "Researcher", "Curriculum Designer"],
        explanation_template=(
            "Your passion for ideas and genuine care for others make education and research "
            "a natural fit — you thrive when sharing knowledge and exploring new questions."
        ),
    ),
    CareerField(
        name="Finance & Investment",
        riasec_codes=["C", "E", "I"],
        example_roles=["Financial Analyst", "Investment Manager", "Risk Analyst"],
        explanation_template=(
            "Your conscientiousness and analytical precision are assets in finance, "
            "where disciplined thinking and risk awareness drive strong outcomes."
        ),
    ),
    CareerField(
        name="Healthcare & Clinical Practice",
        riasec_codes=["S", "I", "R"],
        example_roles=["Physician", "Clinical Psychologist", "Nurse Practitioner"],
        explanation_template=(
            "Your empathy and investigative mindset are central to effective healthcare. "
            "Clinical roles reward both your people skills and diagnostic thinking."
        ),
    ),
    CareerField(
        name="Law & Public Policy",
        riasec_codes=["E", "I", "C"],
        example_roles=["Lawyer", "Policy Analyst", "Legal Researcher", "Judge"],
        explanation_template=(
            "Your precision, argumentation skills, and strategic mindset are well-suited "
            "to legal and policy environments that require persuasion and structured reasoning."
        ),
    ),
    CareerField(
        name="Media, Journalism & Content Creation",
        riasec_codes=["A", "E", "S"],
        example_roles=["Journalist", "Content Creator", "Documentary Filmmaker", "Editor"],
        explanation_template=(
            "Your curiosity and communication ability drive impactful storytelling. "
            "Media roles reward those who can surface truth and connect with audiences."
        ),
    ),
    CareerField(
        name="Engineering & Technical Operations",
        riasec_codes=["R", "C", "I"],
        example_roles=["Civil Engineer", "Industrial Designer", "Operations Manager"],
        explanation_template=(
            "Your methodical approach and hands-on orientation suit engineering and operations "
            "roles where precision, reliability, and technical mastery are valued."
        ),
    ),
    CareerField(
        name="Social Work & Non-Profit Leadership",
        riasec_codes=["S", "A", "E"],
        example_roles=["Social Worker", "NGO Director", "Community Organizer"],
        explanation_template=(
            "Your deep empathy and drive for impact align with mission-driven careers "
            "where improving lives is the primary measure of success."
        ),
    ),
]


# ─────────────────────────────────────────────
# Scoring Logic
# ─────────────────────────────────────────────

def _ocean_dict(vec: OceanVector) -> dict[str, float]:
    return {
        "O": vec.openness,
        "C": vec.conscientiousness,
        "E": vec.extraversion,
        "A": vec.agreeableness,
        "N": vec.neuroticism,
    }


def _score_riasec_type(rt: RiasecType, ocean: dict[str, float]) -> float:
    """Dot product of OCEAN vector with RIASEC weights, clamped to [0, 1]."""
    raw = sum(ocean[trait] * weight for trait, weight in rt.weights.items())
    # raw spans approximately [-0.8, 1.0]; shift and normalize
    normalized = (raw + 0.8) / 1.8
    return round(max(0.0, min(1.0, normalized)), 4)


def _score_career_field(field: CareerField, riasec_scores: dict[str, float]) -> float:
    """
    Career match score = weighted average of its primary RIASEC scores.
    Primary code has highest weight; subsequent codes are lower.
    """
    weights = [0.5, 0.3, 0.2]
    score = 0.0
    for i, code in enumerate(field.riasec_codes[:3]):
        score += riasec_scores.get(code, 0.0) * weights[i]
    return round(score, 4)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def map_careers(vec: OceanVector, top_n: int = 5) -> list[CareerMatch]:
    """
    Main entry point.
    Returns top_n ranked CareerMatch objects sorted by match_score descending.
    """
    ocean = _ocean_dict(vec)

    # Step 1: Score all RIASEC types
    riasec_scores: dict[str, float] = {
        rt.code: _score_riasec_type(rt, ocean) for rt in RIASEC_TYPES
    }

    # Step 2: Score all career fields
    scored_fields: list[tuple[CareerField, float]] = [
        (field, _score_career_field(field, riasec_scores))
        for field in CAREER_FIELDS
    ]

    # Step 3: Sort and take top_n
    scored_fields.sort(key=lambda x: x[1], reverse=True)
    top_fields = scored_fields[:top_n]

    # Step 4: Build CareerMatch objects
    results: list[CareerMatch] = []
    for rank, (field, score) in enumerate(top_fields, start=1):
        results.append(CareerMatch(
            rank=rank,
            field=field.name,
            riasec_codes=field.riasec_codes,
            match_score=score,
            explanation=field.explanation_template,
            example_roles=field.example_roles,
        ))

    return results


def get_riasec_scores(vec: OceanVector) -> dict[str, float]:
    """Expose raw RIASEC scores for debugging or dashboard display."""
    ocean = _ocean_dict(vec)
    return {rt.code: _score_riasec_type(rt, ocean) for rt in RIASEC_TYPES}
