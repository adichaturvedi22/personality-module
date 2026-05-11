# 🧠 AI Career Counsellor — Personality & College Recommendation Engine

> A Python-based backend module that analyses personality through psychometric testing, maps it to career fields, and recommends Indian colleges — all without any external paid API.

---

## 📌 Table of Contents

- [What This Module Does](#-what-this-module-does)
- [Why It Exists](#-why-it-exists)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [Setup & Installation](#-setup--installation)
- [Running the Server](#-running-the-server)
- [Testing](#-testing)
- [Dataset Sources](#-dataset-sources)
- [Future Roadmap](#-future-roadmap)

---

## 🎯 What This Module Does

This module is the backend brain of an AI-powered career counselling web application. A user answers **20 carefully designed questions** and the system:

1. **Analyses their personality** using the scientifically validated Big Five (OCEAN) model
2. **Builds a personality profile** with a named archetype and human-readable insights
3. **Recommends career fields** ranked by compatibility with their personality
4. **Recommends Indian colleges** filtered by their career goals, personality type, state preference, and NAAC quality grade

Everything runs locally — no paid API, no rate limits, no third-party dependency for core functionality.

---

## 💡 Why It Exists

Most career guidance tools in India either:
- Ask vague questions with no scientific basis
- Give generic results that don't account for personality
- Recommend careers without connecting them to accessible institutions

This module solves all three problems by combining **psychometric science** (Big Five OCEAN model), **career research** (RIASEC framework from O\*NET), and **real government data** (53,473 colleges from AISHE + NAAC grading) into a single, fast, offline pipeline.

---

## ⚙️ How It Works

### The Full Pipeline

```
User submits 20 answers (Likert 1–5)
            ↓
    OCEAN Scoring Engine
    (reverse coding + normalization + confidence score)
            ↓
    Thought Process Inference
    (decision style, work style, social behaviour)
            ↓
    Personality Profile Builder
    (named archetype from 12 types + dominant trait)
            ↓
    Career Mapping Engine
    (OCEAN → RIASEC → 15 career fields, ranked by score)
            ↓
    College Recommendation Engine
    (stream match + NAAC quality + personality-environment fit)
            ↓
    Structured JSON result → Frontend / Dashboard
```

---

### Step 1 — 20 MCQ Questions

The questions use the **compressed Big Five (OCEAN)** model — 4 questions per trait, designed to capture maximum personality signal with minimum user fatigue.

| Trait | Measures | Example Question |
|---|---|---|
| **Openness (O)** | Curiosity, creativity, love of ideas | "I enjoy exploring new ideas even if they challenge what I believe" |
| **Conscientiousness (C)** | Discipline, organization, follow-through | "I always plan my work carefully and meet my commitments" |
| **Extraversion (E)** | Social energy, assertiveness, enthusiasm | "I feel energized after spending time with large groups" |
| **Agreeableness (A)** | Empathy, cooperation, trust | "I genuinely care about the well-being of others, even strangers" |
| **Neuroticism (N)** | Emotional sensitivity, stress response | "I often worry about things that might go wrong" |

**Key design choices:**
- ~50% of questions are **reverse-coded** to prevent acquiescence bias (the habit of agreeing with everything)
- Answering all 5s or all 1s produces mid-range scores — the system catches lazy responses
- Response time per question is logged as a future ML signal

---

### Step 2 — OCEAN Scoring Engine

Converts raw answers into a normalized personality vector:

```
O = 0.78    (Openness)
C = 0.61    (Conscientiousness)
E = 0.82    (Extraversion)
A = 0.55    (Agreeableness)
N = 0.30    (Neuroticism)
Confidence = 0.91
```

**How scoring works:**
- Reverse-coded questions are flipped: `effective score = 6 − raw score`
- Scores per trait are summed and normalized to `[0.0, 1.0]`
- A **confidence score** is computed from intra-trait consistency — if a user contradicts themselves within the same trait, confidence drops

---

### Step 3 — Thought Process Inference

Translates raw scores into human-readable psychological insights across three dimensions:

| Dimension | Example Output |
|---|---|
| **Decision Style** | "Analytical Visionary — evaluates options creatively but decides with structured logic" |
| **Work Style** | "Structured Collaborator — combines disciplined execution with strong team coordination" |
| **Social Behaviour** | "Collaborative Connector — builds rapport effortlessly; strong team player" |

Also generates:
- **Top 5 Strengths** based on dominant traits
- **Top 3 Growth Areas** based on low-scoring traits

---

### Step 4 — Personality Profile Builder

Matches the OCEAN vector to one of **12 named archetypes**:

| Archetype | Trait Pattern |
|---|---|
| The Visionary Leader | High O + High C + High E |
| The Creative Energizer | High O + High E + Low C |
| The Deep Thinker | High O + Low E + High C |
| The Strategic Driver | High E + High C + Low A |
| The Quiet Achiever | Low E + High C + High A |
| The Empathetic Connector | High A + High E + High N |
| The Resilient Specialist | Low E + High C + Low N |
| The Balanced Generalist | All traits in medium range |
| ...and 4 more | Various combinations |

---

### Step 5 — Career Mapping Engine

Maps the OCEAN vector → RIASEC interest codes → ranked career fields.

**RIASEC types scored:**

| Code | Type | Key OCEAN drivers |
|---|---|---|
| R | Realistic | High C, Low O |
| I | Investigative | High O, High C, Low E |
| A | Artistic | High O, Low C |
| S | Social | High E, High A |
| E | Enterprising | High E, High C |
| C | Conventional | High C, Low O |

**15 career fields in the recommendation pool:**

Software Engineering, Data Science & AI, Product Management, UX Design, Entrepreneurship, Marketing & Brand Strategy, Management Consulting, Finance & Investment, Healthcare, Law & Public Policy, Education & Research, Media & Journalism, Social Work, Engineering & Technical Operations, People & Organizational Psychology.

Each recommendation includes a match score, relevant RIASEC codes, explanation, and example job roles.

---

### Step 6 — College Recommendation Engine

Recommends Indian colleges from a master dataset of **53,473 institutions** built from government sources.

**Scoring formula:**

```
Match Score = Stream Match (40%) + NAAC Quality (35%) + Personality Fit (25%)
```

| Component | What it checks |
|---|---|
| **Stream Match** | Does the college teach subjects relevant to the user's top career fields? |
| **NAAC Quality** | A++ = 1.0, A+ = 0.9, A = 0.78, B++ = 0.65, B+ = 0.52, B = 0.40, C = 0.25 |
| **Personality Fit** | High E → Urban colleges; High C → Government/Aided; High O → Autonomous; High A → Government Aided |

**Dataset coverage:**
- 53,473 colleges across 36 states
- 31,769 colleges with inherited NAAC quality grade
- 21,078 colleges with confirmed stream/subject data
- Filterable by state, NAAC grade requirement, and result count

---

## 🛠 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API Framework | FastAPI | Auto-generates Swagger docs, async-ready, fast |
| Data Validation | Pydantic v2 | Strict schema enforcement on all inputs/outputs |
| Data Processing | Pandas | Handles 53k-row dataset efficiently |
| Database | SQLite | Zero-config local storage for test results and training data |
| Server | Uvicorn | ASGI server, supports hot reload for development |
| ML (future) | scikit-learn | Already installed, hooks ready for adaptive testing |

---

## 📁 Project Structure

```
persolanity module/
│
├── api.py                    # FastAPI app — all REST endpoints
├── orchestrator.py           # Pipeline glue — wires all modules together
│
├── models.py                 # Pydantic schemas (request/response models)
├── questions.py              # 20 MCQs with OCEAN trait mapping
├── scoring_engine.py         # OCEAN vector calculator
├── thought_process.py        # Trait → human-readable insight engine
├── personality_profile.py    # Archetype builder
├── career_mapping.py         # RIASEC + career field ranker
│
├── college_recommender.py    # College recommendation engine
├── college_data_prep.py      # One-time dataset merge script
│
├── data_logger.py            # SQLite persistence + training data export
├── test_runner.py            # 37-test validation suite
├── requirements.txt          # All dependencies
│
└── data/                     # ← NOT in git (see .gitignore)
    ├── College-ALL_COLLEGE.xlsx
    ├── University-ALL_UNIVERSITIES.xlsx
    ├── Institutions_accredited_by_NAAC_*.xlsx
    ├── Report-133-*.csv
    └── master_colleges.csv   # built by college_data_prep.py
```

---

## 🔌 API Endpoints

### `GET /health`
Quick server health check.
```json
{ "status": "ok", "version": "1.0.0" }
```

---

### `GET /questions`
Returns all 20 MCQ questions for the frontend to display. Reverse-coding metadata is never exposed.
```json
{
  "count": 20,
  "scale": "Likert 1–5",
  "questions": [
    { "id": 1, "text": "I enjoy exploring new ideas...", "category": "Openness" }
  ]
}
```

---

### `POST /submit-test`
Core endpoint. Takes 20 answers, runs the full pipeline.

**Request body:**
```json
{
  "user_id": "aditya-001",
  "answers": [
    { "question_id": 1, "score": 4 },
    { "question_id": 2, "score": 2 }
  ]
}
```

**Response includes:**
- `ocean_vector` — 5 trait scores + confidence
- `personality_profile` — archetype, traits, thought process insights
- `career_recommendations` — top 5 ranked career fields with scores

---

### `GET /results/{user_id}`
Retrieves a previously stored test result from the database.

---

### `GET /college-recommendations/{user_id}`
Recommends colleges based on a completed personality test.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `state` | string | None | Filter by Indian state e.g. `Uttar Pradesh` |
| `top_n` | integer | 10 | Number of results (max 50) |
| `require_naac` | boolean | false | Only return NAAC-graded colleges |

**Example:**
```
GET /college-recommendations/aditya-001?state=Maharashtra&top_n=15&require_naac=true
```

**Response per college:**
```json
{
  "rank": 1,
  "name": "Khalsa College for Women Amritsar",
  "state": "Punjab",
  "naac_grade": "A++",
  "naac_cgpa": 3.85,
  "streams_offered": ["Arts", "Science", "Engineering"],
  "match_score": 0.90,
  "score_breakdown": {
    "stream_match": 1.0,
    "naac_quality": 1.0,
    "personality_fit": 0.45
  },
  "website": "www.khalsacollege.edu.in"
}
```

---

### `GET /career-recommendations`
Static reference list of all 15 career fields with RIASEC codes and example roles.

---

### `POST /feedback`
Logs user feedback after seeing their results. This data becomes the training dataset for future ML model improvements.

```json
{
  "user_id": "aditya-001",
  "rating": 4,
  "chosen_career": "Data Science & AI Research",
  "comment": "Very accurate result"
}
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/personality-module.git
cd personality-module
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the datasets
Download these files from the official government sources and place them inside a `data/` folder in the project root:

| File | Source |
|---|---|
| `College-ALL_COLLEGE.xlsx` | [AISHE — aishe.gov.in](https://aishe.gov.in) |
| `University-ALL_UNIVERSITIES.xlsx` | [AISHE — aishe.gov.in](https://aishe.gov.in) |
| `Institutions_accredited_by_NAAC_having_valid_accreditation.xlsx` | [NAAC — naac.gov.in](https://naac.gov.in) |
| `Report-133-20042015035450626PM-2012-2013.csv` | [AISHE Reports](https://aishe.gov.in/aishe/reports) |

### 4. Build the master college dataset (run once)
```bash
python college_data_prep.py
```
This merges all four dataset files into `data/master_colleges.csv`. Takes about 30 seconds. You only need to run this again if you update the source files.

---

## ▶️ Running the Server

```bash
uvicorn api:app --reload --port 8000
```

- API is live at: `http://127.0.0.1:8000`
- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`

The Swagger UI lets you test every endpoint visually without writing any code — recommended for first-time testing.

---

## 🧪 Testing

The module ships with a 37-test validation suite covering every layer of the pipeline:

```bash
python test_runner.py
```

Expected output:
```
─────────────────────────────────────────────────────
  1. Question Bank
─────────────────────────────────────────────────────
  ✅ PASS  20 questions in bank
  ✅ PASS  Unique sequential IDs 1–20
  ✅ PASS  4 questions per OCEAN trait
  ...

═════════════════════════════════════════════════════
  RESULTS: 37/37 passed  🎉 All tests passed!
═════════════════════════════════════════════════════
```

**What the tests cover:**

| Section | Tests |
|---|---|
| Question Bank | Count, unique IDs, balanced traits, metadata safety |
| Scoring Engine | Reverse coding math, normalization, confidence score |
| Thought Process | All 5 trait levels, composite style inference |
| Personality Profile | Archetype matching, fallback handling |
| Career Mapping | Ranking order, score ranges, profile-to-career accuracy |
| Input Validation | Wrong answer count, out-of-range scores, duplicate IDs |
| Integration | End-to-end pipeline, deterministic output, JSON serialization |
| Data Logger | Write/read round-trip, feedback logging, missing user handling |

---

## 📊 Dataset Sources

All data used in this project is sourced from official Indian government websites:

| Dataset | Source | Records | Used For |
|---|---|---|---|
| All India Survey on Higher Education (AISHE) — Colleges | Ministry of Education | 53,473 | College base data |
| All India Survey on Higher Education (AISHE) — Universities | Ministry of Education | 1,410 | University linkage |
| NAAC Accreditation Data | National Assessment and Accreditation Council | 497 | Quality grading |
| AISHE Report 133 — Enrolment by Subject | Ministry of Education | 52,229 | Stream detection |

---

## 🔮 Future Roadmap

The module is architected to support ML upgrades without rewriting existing code:

### Phase 2 — Adaptive Testing
Next question selected based on previous answers. High-O responses unlock different follow-ups than low-O responses. Reduces test length while improving signal quality.

### Phase 3 — NLP Analysis
Open-ended answer boxes alongside MCQs. Sentiment and keyword analysis feeds additional signals into the OCEAN vector.

### Phase 4 — Behavioural Signals
Response time per question is already being logged. Patterns like hesitation on social questions or fast answers on structure questions become secondary features.

### Phase 5 — ML Recommendation Model
Once enough `/feedback` data accumulates (~500+ responses), train a supervised classifier on:
- Input: OCEAN vector + top career field
- Label: user rating + chosen career
- Output: personalized score adjustments per user segment

The `data_logger.py` module is already collecting everything needed for this. A `GET /export-training-data` endpoint can dump it directly into a pandas DataFrame for model training.

---

## 🤝 Contributing

This project is part of a college final year project. Contributions, suggestions, and feedback are welcome via GitHub Issues.

---

## 📄 License

For academic and educational use. Dataset files are property of their respective government sources (AISHE, NAAC) and are not redistributed in this repository.
