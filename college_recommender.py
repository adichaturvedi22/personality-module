"""
college_recommender.py
----------------------
Recommends Indian colleges based on:
  1. Stream match     — does the college offer subjects relevant to top careers?
  2. NAAC quality     — government-graded quality signal (A++ → C)
  3. Personality fit  — OCEAN traits mapped to college environment

Loads data/master_colleges.csv at startup (built by college_data_prep.py).
Zero external API calls. Fully offline.

Usage:
    from college_recommender import recommend_colleges
    colleges = recommend_colleges(ocean_vector, career_matches, state=None, top_n=10)
"""

from __future__ import annotations

import os
import pandas as pd
from functools import lru_cache
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# Output Model
# ─────────────────────────────────────────────

class CollegeMatch(BaseModel):
    rank: int
    name: str
    state: str
    district: str
    location: str                   # Urban / Rural
    college_type: str
    management: str
    university_name: str
    naac_grade: str                 # A++ / A+ / A / B++ / B+ / B / C / Ungraded
    naac_cgpa: float | None
    streams_offered: list[str]
    match_score: float = Field(..., ge=0.0, le=1.0)
    score_breakdown: dict[str, float]
    website: str | None


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MASTER_CSV = os.path.join(os.path.dirname(__file__), "data", "master_colleges.csv")

# NAAC grade → numeric quality score
NAAC_SCORE: dict[str, float] = {
    "A++": 1.00,
    "A+":  0.90,
    "A":   0.78,
    "B++": 0.65,
    "B+":  0.52,
    "B":   0.40,
    "C":   0.25,
}

# Career field → relevant streams in master dataset
CAREER_TO_STREAMS: dict[str, list[str]] = {
    "Software Engineering & Architecture":  ["Engineering", "Computer_Science"],
    "Data Science & AI Research":           ["Engineering", "Computer_Science", "Science"],
    "Product Management":                   ["Engineering", "Management", "Commerce"],
    "Marketing & Brand Strategy":           ["Management", "Commerce", "Arts"],
    "Entrepreneurship & Startups":          ["Management", "Commerce", "Engineering"],
    "UX Design & Creative Technology":      ["Engineering", "Computer_Science", "Arts"],
    "Finance & Investment":                 ["Commerce", "Management"],
    "Law & Public Policy":                  ["Law", "Arts"],
    "Healthcare & Clinical Practice":       ["Medicine", "Science"],
    "Education & Academic Research":        ["Education", "Arts", "Science"],
    "Management Consulting":                ["Management", "Commerce", "Engineering"],
    "Media, Journalism & Content Creation": ["Arts", "Commerce"],
    "Social Work & Non-Profit Leadership":  ["Arts", "Education"],
    "People & Organizational Psychology":   ["Arts", "Management"],
    "Engineering & Technical Operations":   ["Engineering", "Science"],
}

ALL_STREAMS = [
    "Arts", "Science", "Commerce", "Computer_Science",
    "Management", "Education", "Engineering", "Medicine",
    "Agriculture", "Law",
]

# Scoring weights — must sum to 1.0
WEIGHT_STREAM      = 0.40
WEIGHT_NAAC        = 0.35
WEIGHT_PERSONALITY = 0.25


# ─────────────────────────────────────────────
# Data Loading (cached — loads once at startup)
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_master() -> pd.DataFrame:
    if not os.path.exists(MASTER_CSV):
        raise FileNotFoundError(
            f"master_colleges.csv not found at {MASTER_CSV}. "
            "Run college_data_prep.py first."
        )
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    # Ensure bool columns
    for s in ALL_STREAMS:
        col = f"has_{s}"
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    df["naac_cgpa"] = pd.to_numeric(df["naac_cgpa"], errors="coerce")
    df["naac_grade"] = df["naac_grade"].fillna("Ungraded").str.strip().str.upper()
    df["state"]      = df["state"].fillna("").str.strip().str.title()
    df["location"]   = df["location"].fillna("").str.strip().str.title()
    df["management"] = df["management"].fillna("").str.strip()
    return df


# ─────────────────────────────────────────────
# Scoring Functions
# ─────────────────────────────────────────────

def _stream_score(row: pd.Series, needed_streams: list[str]) -> float:
    """
    Fraction of needed streams the college offers.
    e.g. needs [Engineering, CS], college has Engineering only → 0.5
    """
    if not needed_streams:
        return 0.5
    matched = sum(1 for s in needed_streams if row.get(f"has_{s}", False))
    return matched / len(needed_streams)


def _naac_score(grade: str) -> float:
    return NAAC_SCORE.get(grade.upper() if grade else "", 0.15)


def _personality_score(row: pd.Series, ocean: dict[str, float]) -> float:
    """
    Map OCEAN traits to college environment preferences.

    High E  → prefers Urban location
    High C  → prefers Government / Aided management (structured)
    High O  → prefers Autonomous college (flexible curriculum)
    High A  → prefers Government Aided (collaborative, inclusive)
    Low  N  → slight preference for stable government institutions
    """
    score = 0.0
    location  = str(row.get("location", "")).lower()
    mgmt      = str(row.get("management", "")).lower()
    ctype     = str(row.get("college_type", "")).lower()

    # Extraversion → urban preference
    if ocean.get("E", 0.5) > 0.6 and "urban" in location:
        score += 0.25
    elif ocean.get("E", 0.5) < 0.4 and "rural" in location:
        score += 0.20

    # Conscientiousness → structured/government management
    if ocean.get("C", 0.5) > 0.6:
        if "government" in mgmt or "aided" in mgmt:
            score += 0.25
    else:
        if "private" in mgmt and "unaided" in mgmt:
            score += 0.15

    # Openness → autonomous colleges (flexible curriculum)
    if ocean.get("O", 0.5) > 0.65 and "autonomous" in ctype:
        score += 0.25

    # Agreeableness → government aided (inclusive)
    if ocean.get("A", 0.5) > 0.6 and "aided" in mgmt:
        score += 0.15

    # Low Neuroticism → stable government institutions
    if ocean.get("N", 0.5) < 0.35 and "government" in mgmt:
        score += 0.10

    return min(score, 1.0)


def _get_streams_offered(row: pd.Series) -> list[str]:
    return [s for s in ALL_STREAMS if row.get(f"has_{s}", False)]


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def recommend_colleges(
    ocean: dict[str, float],
    career_matches: list[dict],
    state: str | None = None,
    top_n: int = 10,
    require_naac: bool = False,
) -> list[CollegeMatch]:
    """
    Main entry point.

    Args:
        ocean          : dict with keys O, C, E, A, N (0–1 floats)
        career_matches : list of CareerMatch dicts (from career_mapping output)
        state          : optional state filter e.g. "Uttar Pradesh"
        top_n          : number of results to return
        require_naac   : if True, only return NAAC-graded colleges

    Returns:
        List of CollegeMatch sorted by match_score descending
    """
    df = _load_master().copy()

    # ── Filter ────────────────────────────────
    if state:
        state_clean = state.strip().title()
        df = df[df["state"].str.title() == state_clean]
        if df.empty:
            # Fallback: partial match
            df = _load_master().copy()
            df = df[df["state"].str.contains(state, case=False, na=False)]

    if require_naac:
        df = df[df["naac_grade"] != "UNGRADED"]

    if df.empty:
        return []

    # ── Build needed streams from top careers ──
    needed_streams: list[str] = []
    for cm in career_matches[:3]:           # use top 3 career matches
        field = cm.get("field", "") if isinstance(cm, dict) else getattr(cm, "field", "")
        needed_streams.extend(CAREER_TO_STREAMS.get(field, []))
    # Deduplicate, preserve order
    seen: set[str] = set()
    unique_streams: list[str] = []
    for s in needed_streams:
        if s not in seen:
            seen.add(s)
            unique_streams.append(s)
    needed_streams = unique_streams[:5]     # cap at 5 streams

    # ── Score each college ────────────────────
    df = df.copy()
    df["_stream"]      = df.apply(_stream_score, axis=1, needed_streams=needed_streams)
    df["_naac"]        = df["naac_grade"].apply(_naac_score)
    df["_personality"] = df.apply(_personality_score, axis=1, ocean=ocean)

    df["_total"] = (
        df["_stream"]      * WEIGHT_STREAM +
        df["_naac"]        * WEIGHT_NAAC +
        df["_personality"] * WEIGHT_PERSONALITY
    ).round(4)

    # ── Sort + take top N ─────────────────────
    df = df.sort_values("_total", ascending=False).head(top_n * 3)  # oversample then trim
    df = df.drop_duplicates(subset=["name", "state"]).head(top_n)

    # ── Build output ──────────────────────────
    results: list[CollegeMatch] = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        results.append(CollegeMatch(
            rank=rank,
            name=str(row["name"]),
            state=str(row["state"]),
            district=str(row.get("district", "")),
            location=str(row.get("location", "Unknown")),
            college_type=str(row.get("college_type", "")),
            management=str(row.get("management", "")),
            university_name=str(row.get("university_name", "")),
            naac_grade=str(row["naac_grade"]),
            naac_cgpa=row["naac_cgpa"] if pd.notna(row.get("naac_cgpa")) else None,
            streams_offered=_get_streams_offered(row),
            match_score=float(row["_total"]),
            score_breakdown={
                "stream_match":      round(float(row["_stream"]),      3),
                "naac_quality":      round(float(row["_naac"]),        3),
                "personality_fit":   round(float(row["_personality"]), 3),
            },
            website=str(row["website"]) if pd.notna(row.get("website")) else None,
        ))

    return results
