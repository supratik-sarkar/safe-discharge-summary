"""Select surgical cohorts from MIMIC-III demo (robust NOTEEVENTS loader)."""
import duckdb, pandas as pd
from pathlib import Path

ROOT = Path.home() / "Desktop/HealthcareAI_Stanford"
DATA = ROOT / "data/mimic_demo/mimic-iii-clinical-database-demo-1.4"
OUT  = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

def load_duck(table):
    df = con.execute(
        f"SELECT * FROM read_csv_auto('{DATA}/{table}.csv', header=True)"
    ).df()
    df.columns = [c.upper() for c in df.columns]
    return df

patients   = load_duck("PATIENTS")
admissions = load_duck("ADMISSIONS")
proc_icd   = load_duck("PROCEDURES_ICD")
d_proc     = load_duck("D_ICD_PROCEDURES")

# NOTEEVENTS has embedded newlines — use pandas, which handles quoted multiline fields.
notes = pd.read_csv(
    DATA / "NOTEEVENTS.csv",
    dtype=str,
    engine="python",           # tolerant of messy quoting
    on_bad_lines="skip",
    quotechar='"',
    escapechar="\\",
)
notes.columns = [c.upper() for c in notes.columns]

print("Loaded:",
      {"PATIENTS":len(patients),"ADMISSIONS":len(admissions),
       "NOTEEVENTS":len(notes),"PROCEDURES_ICD":len(proc_icd),
       "D_ICD_PROCEDURES":len(d_proc)})
print("NOTEEVENTS columns:", list(notes.columns))
print("NOTEEVENTS sample categories:", notes["CATEGORY"].dropna().unique()[:10])

# Drop notes without HADM_ID (discharge summaries have one, others may not)
notes = notes[notes["HADM_ID"].notna()].copy()
notes["HADM_ID"] = notes["HADM_ID"].astype(float).astype("Int64")

# Surgical cohort by procedure title
proc = proc_icd.merge(d_proc[["ICD9_CODE","LONG_TITLE"]], on="ICD9_CODE", how="left")
CATS = {
  "cardiac":    r"coronary|cabg|valve|cardiac|aortic|bypass",
  "orthopedic": r"arthroplasty|fracture|hip|knee|spinal fusion|femur|tibia",
  "vascular":   r"endarterectomy|aneurysm|vascular|angioplasty|stent.*artery",
  "neuro":      r"craniotomy|craniectomy|laminectomy|cerebr|ventriculostomy",
  "abdominal":  r"colectomy|appendectomy|cholecystectomy|laparotomy|hernia|gastrectomy",
}
rows = []
for cat, pat in CATS.items():
    m = proc[proc.LONG_TITLE.fillna("").str.contains(pat, case=False, regex=True)].copy()
    m["category"] = cat
    rows.append(m)
cohort = pd.concat(rows).drop_duplicates(["SUBJECT_ID","HADM_ID","category"])
cohort["HADM_ID"] = cohort["HADM_ID"].astype(float).astype("Int64")
print("\nRaw surgical matches per category:")
print(cohort.groupby("category").HADM_ID.nunique())

# Qualify admissions: >=5 notes AND has a discharge summary or post-op note
def qualifies(g):
    if len(g) < 5: return False
    cats = g.CATEGORY.fillna("").str.lower()
    txt  = g.TEXT.fillna("").str.lower()
    return cats.str.contains("discharge").any() or txt.str.contains(r"post[- ]?op", regex=True).any()

eligible = notes.groupby("HADM_ID").filter(qualifies).HADM_ID.unique()
print(f"\nAdmissions with >=5 notes + DS/post-op: {len(eligible)}")

final = cohort[cohort.HADM_ID.isin(eligible)]
print("\nFinal surgical cohort per category:")
print(final.groupby("category").HADM_ID.nunique())

# Fallback: any admission that has a discharge summary
if final.empty:
    print("\n[fallback] Using all admissions with a discharge summary.")
    ds_hadm = notes[notes.CATEGORY.fillna("").str.lower().str.contains("discharge")].HADM_ID.unique()
    final = pd.DataFrame({"HADM_ID": ds_hadm, "SUBJECT_ID": 0, "category": "any"})
    eligible_notes = notes[notes.HADM_ID.isin(ds_hadm)]
else:
    eligible_notes = notes[notes.HADM_ID.isin(final.HADM_ID)]

final.to_parquet(OUT/"cohort.parquet", index=False)
eligible_notes.to_parquet(OUT/"cohort_notes.parquet", index=False)
print(f"\nSaved cohort: {len(final)} rows")
print(f"Saved cohort_notes: {len(eligible_notes)} notes across {eligible_notes['HADM_ID'].nunique()} admissions")
