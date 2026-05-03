"""
data_logger.py
--------------
Persists test results to SQLite for:
  - Audit trails
  - Future ML model training
  - Analytics and aggregation

Schema:
  - test_sessions     → one row per submission
  - answers           → one row per answer (20 per session)
  - ocean_scores      → one row per session
  - career_results    → one row per recommended career per session
  - user_feedback     → optional post-result feedback
"""

from __future__ import annotations
import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager
from models import TestResult


DB_PATH = os.getenv("CAREER_DB_PATH", "career_counselor.db")


# ─────────────────────────────────────────────
# Schema Init
# ─────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS test_sessions (
    user_id         TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    confidence_score REAL,
    personality_type TEXT,
    dominant_trait  TEXT,
    raw_payload     TEXT   -- full JSON snapshot for ML
);

CREATE TABLE IF NOT EXISTS answers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    question_id     INTEGER NOT NULL,
    score           INTEGER NOT NULL,
    time_taken_ms   INTEGER,
    FOREIGN KEY (user_id) REFERENCES test_sessions(user_id)
);

CREATE TABLE IF NOT EXISTS ocean_scores (
    user_id             TEXT PRIMARY KEY,
    openness            REAL,
    conscientiousness   REAL,
    extraversion        REAL,
    agreeableness       REAL,
    neuroticism         REAL,
    FOREIGN KEY (user_id) REFERENCES test_sessions(user_id)
);

CREATE TABLE IF NOT EXISTS career_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    rank        INTEGER NOT NULL,
    field_name  TEXT NOT NULL,
    match_score REAL NOT NULL,
    riasec_codes TEXT,
    FOREIGN KEY (user_id) REFERENCES test_sessions(user_id)
);

CREATE TABLE IF NOT EXISTS user_feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    rating          INTEGER CHECK(rating BETWEEN 1 AND 5),
    chosen_career   TEXT,
    comment         TEXT,
    submitted_at    TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES test_sessions(user_id)
);

CREATE INDEX IF NOT EXISTS idx_answers_user ON answers(user_id);
CREATE INDEX IF NOT EXISTS idx_careers_user ON career_results(user_id);
"""


@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist. Call once at startup."""
    with _get_conn() as conn:
        conn.executescript(_SCHEMA_SQL)


# ─────────────────────────────────────────────
# Write Operations
# ─────────────────────────────────────────────

def log_result(result: TestResult) -> None:
    """Persist a complete TestResult to the database."""
    with _get_conn() as conn:
        # 1. Session
        conn.execute(
            """INSERT OR REPLACE INTO test_sessions
               (user_id, timestamp, confidence_score, personality_type, dominant_trait, raw_payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                result.user_id,
                result.timestamp,
                result.ocean_vector.confidence_score,
                result.personality_profile.type_label,
                result.personality_profile.dominant_trait,
                result.model_dump_json(),       # full snapshot → training data
            ),
        )

        # 2. Answers
        conn.executemany(
            "INSERT INTO answers (user_id, question_id, score, time_taken_ms) VALUES (?, ?, ?, ?)",
            [
                (result.user_id, a.question_id, a.score, a.time_taken_ms)
                for a in result.raw_answers
            ],
        )

        # 3. OCEAN scores
        v = result.ocean_vector
        conn.execute(
            """INSERT OR REPLACE INTO ocean_scores
               (user_id, openness, conscientiousness, extraversion, agreeableness, neuroticism)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (result.user_id, v.openness, v.conscientiousness,
             v.extraversion, v.agreeableness, v.neuroticism),
        )

        # 4. Career results
        conn.executemany(
            """INSERT INTO career_results
               (user_id, rank, field_name, match_score, riasec_codes)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (result.user_id, c.rank, c.field, c.match_score,
                 json.dumps(c.riasec_codes))
                for c in result.career_recommendations
            ],
        )


def log_feedback(user_id: str, rating: int, chosen_career: str | None = None,
                 comment: str | None = None) -> None:
    """Append user feedback — critical for supervised ML training labels."""
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO user_feedback (user_id, rating, chosen_career, comment)
               VALUES (?, ?, ?, ?)""",
            (user_id, rating, chosen_career, comment),
        )


# ─────────────────────────────────────────────
# Read Operations
# ─────────────────────────────────────────────

def get_result(user_id: str) -> dict | None:
    """Retrieve the raw JSON snapshot for a user session."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT raw_payload FROM test_sessions WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row:
            return json.loads(row["raw_payload"])
        return None


def get_training_export(limit: int = 10_000) -> list[dict]:
    """
    Export training data: OCEAN vectors + top career + feedback rating.
    Joins sessions with feedback for supervised ML training.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                o.user_id,
                o.openness, o.conscientiousness, o.extraversion,
                o.agreeableness, o.neuroticism,
                s.personality_type, s.dominant_trait, s.confidence_score,
                f.chosen_career, f.rating
            FROM ocean_scores o
            JOIN test_sessions s ON o.user_id = s.user_id
            LEFT JOIN user_feedback f ON o.user_id = f.user_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
