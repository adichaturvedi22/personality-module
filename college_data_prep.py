"""
college_data_prep.py
--------------------
ONE-TIME script. Run this once to build data/master_colleges.csv.
After that, college_recommender.py loads the master file at startup.

Merge chain:
  College-ALL_COLLEGE.xlsx         (53,473 colleges — base)
      + University-ALL_UNIVERSITIES.xlsx  (1,410 universities)
      + NAAC xlsx                         (494 graded universities)
  → 31,769 colleges inherit a NAAC quality grade

  Report-133 CSV                   (stream enrollment data)
  → derives which subjects each college offers

Final output: data/master_colleges.csv
  ~52k rows, columns:
  aishe_code, name, state, district, location, college_type,
  management, university_name, naac_grade, naac_cgpa,
  has_Arts, has_Science, has_Commerce, has_Computer_Science,
  has_Management, has_Education, has_Engineering,
  has_Medicine, has_Agriculture, has_Law

Run:
  python college_data_prep.py
"""

import os
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

COLLEGE_XLSX  = os.path.join(DATA_DIR, "College-ALL_COLLEGE.xlsx")
UNIV_XLSX     = os.path.join(DATA_DIR, "University-ALL_UNIVERSITIES.xlsx")
NAAC_XLSX     = os.path.join(DATA_DIR, "Institutions_accredited_by_NAAC_having_valid_accreditation-as_on_14082025_1.xlsx")
REPORT_CSV    = os.path.join(DATA_DIR, "Report-133-20042015035450626PM-2012-2013.csv")
OUTPUT_CSV    = os.path.join(DATA_DIR, "master_colleges.csv")


# ── Helpers ───────────────────────────────────────────────────────────

def _clean_name(s: str) -> str:
    """Lowercase, strip whitespace — used for fuzzy name matching fallback."""
    if pd.isna(s):
        return ""
    return str(s).lower().strip()


# ── Step 1: Load college base ─────────────────────────────────────────

print("Loading college base dataset...")
college = pd.read_excel(COLLEGE_XLSX, header=2, dtype=str)
college.columns = [c.strip() for c in college.columns]
college = college.rename(columns={
    "Aishe Code":           "aishe_code",
    "Name":                 "name",
    "State":                "state",
    "District":             "district",
    "Website":              "website",
    "Year Of Establishment":"year_established",
    "Location":             "location",
    "College Type":         "college_type",
    "Manegement":           "management",
    "University Aishe Code":"univ_aishe_code",
    "University Name":      "university_name",
    "University Type":      "university_type",
})
college = college.dropna(subset=["aishe_code", "name"])
print(f"  Colleges loaded: {len(college):,}")


# ── Step 2: Load universities + NAAC grades ───────────────────────────

print("Loading university + NAAC data...")
univ = pd.read_excel(UNIV_XLSX, header=2, dtype=str)
univ.columns = [c.strip() for c in univ.columns]
univ = univ.rename(columns={"Aishe Code": "aishe_code", "Name": "univ_name"})

naac = pd.read_excel(NAAC_XLSX, dtype=str)
naac.columns = [c.strip() for c in naac.columns]
naac = naac.rename(columns={
    "Aishe-Id":        "aishe_id",
    "Current Grade":   "naac_grade",
    "Current CGPA":    "naac_cgpa",
})
naac["naac_cgpa"] = pd.to_numeric(naac["naac_cgpa"], errors="coerce")

# Merge university + NAAC
univ_naac = univ.merge(
    naac[["aishe_id", "naac_grade", "naac_cgpa"]],
    left_on="aishe_code", right_on="aishe_id", how="left"
)
print(f"  Universities with NAAC grade: {univ_naac['naac_grade'].notna().sum():,} / {len(univ_naac):,}")


# ── Step 3: Attach NAAC grade to colleges via university ──────────────

print("Attaching NAAC grades to colleges via university code...")
college = college.merge(
    univ_naac[["aishe_code", "naac_grade", "naac_cgpa"]],
    left_on="univ_aishe_code", right_on="aishe_code",
    how="left",
    suffixes=("", "_univ"),
)
college = college.drop(columns=["aishe_code_univ"], errors="ignore")
print(f"  Colleges with inherited NAAC grade: {college['naac_grade'].notna().sum():,} / {len(college):,}")


# ── Step 4: Derive stream profiles from Report-133 ────────────────────

print("Loading stream enrollment data...")
report = pd.read_csv(
    REPORT_CSV, encoding="latin1", on_bad_lines="skip", low_memory=False
)

STREAM_PAIRS = [
    ("Arts",             "Arts - Male",                            "Arts - Female"),
    ("Science",          "Science - Male",                         "Science - Female"),
    ("Commerce",         "Commerce - Male",                        "Commerce - Female"),
    ("Computer_Science", "Computer App / Computer Science - Male", "Computer App / Computer Science - Female"),
    ("Management",       "Management - Male",                      "Management - Female"),
    ("Education",        "Education - Male",                       "Education - Female"),
    ("Engineering",      "Engineering and Technology - Male",      "Engineering and Technology - Female"),
    ("Medicine",         "Medicine - Male",                        "Medicine - Female"),
    ("Agriculture",      "Agriculture - Male",                     "Agriculture - Female"),
    ("Law",              "Law - Male",                             "Law - Female"),
]

for stream, male_col, female_col in STREAM_PAIRS:
    report[f"has_{stream}"] = (
        pd.to_numeric(report[male_col],   errors="coerce").fillna(0) +
        pd.to_numeric(report[female_col], errors="coerce").fillna(0)
    ) > 0

# Extract college name ID from "College Name (Id: C-XXXXX)" pattern
report["aishe_code_extracted"] = (
    report["Name Of College"]
    .str.extract(r"\(Id:\s*(C-\d+)\)", expand=False)
)

stream_cols = [f"has_{s}" for s, _, _ in STREAM_PAIRS]
stream_df = report[["aishe_code_extracted"] + stream_cols].dropna(subset=["aishe_code_extracted"])
stream_df = stream_df.groupby("aishe_code_extracted").max().reset_index()
stream_df = stream_df.rename(columns={"aishe_code_extracted": "aishe_code"})

print(f"  Stream profiles built: {len(stream_df):,} colleges")


# ── Step 5: Merge streams into master ────────────────────────────────

print("Merging stream profiles into master dataset...")
master = college.merge(stream_df, on="aishe_code", how="left")

# Fill missing stream flags as False
for col in stream_cols:
    master[col] = master[col].fillna(False)

print(f"  Colleges with stream data: {master[stream_cols].any(axis=1).sum():,}")


# ── Step 6: Final cleanup ─────────────────────────────────────────────

print("Final cleanup...")
keep_cols = [
    "aishe_code", "name", "state", "district", "location",
    "college_type", "management", "university_name", "university_type",
    "year_established", "website",
    "naac_grade", "naac_cgpa",
] + stream_cols

master = master[keep_cols].copy()
master["naac_cgpa"] = pd.to_numeric(master["naac_cgpa"], errors="coerce")

# Normalise state names
master["state"] = master["state"].str.strip().str.title()
master["district"] = master["district"].str.strip().str.title()
master["location"] = master["location"].str.strip().str.title()
master["management"] = master["management"].str.strip()
master["naac_grade"] = master["naac_grade"].str.strip().str.upper()


# ── Step 7: Save ─────────────────────────────────────────────────────

master.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ master_colleges.csv saved → {OUTPUT_CSV}")
print(f"   Total colleges : {len(master):,}")
print(f"   With NAAC grade: {master['naac_grade'].notna().sum():,}")
print(f"   With streams   : {master[stream_cols].any(axis=1).sum():,}")
print(f"   States covered : {master['state'].nunique()}")
print(f"   Columns        : {master.columns.tolist()}")
