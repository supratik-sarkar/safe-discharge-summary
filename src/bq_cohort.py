from google.cloud import bigquery
from pathlib import Path

PROJECT = "sds-ss-2026q2"
OUT = Path.home() / "Desktop/HealthcareAI_Stanford/results"
OUT.mkdir(parents=True, exist_ok=True)

client = bigquery.Client(project=PROJECT)

sql = """
WITH surgical AS (
  SELECT DISTINCT p.subject_id, p.hadm_id,
    CASE
      WHEN REGEXP_CONTAINS(LOWER(d.long_title), r'coronary|cabg|valve|cardiac|aortic|bypass') THEN 'cardiac'
      WHEN REGEXP_CONTAINS(LOWER(d.long_title), r'arthroplasty|hip|knee|fracture|spinal fusion|femur|tibia') THEN 'orthopedic'
      WHEN REGEXP_CONTAINS(LOWER(d.long_title), r'endarterectomy|aneurysm|angioplasty') THEN 'vascular'
      WHEN REGEXP_CONTAINS(LOWER(d.long_title), r'craniotomy|craniectomy|laminectomy|ventriculostomy') THEN 'neuro'
      WHEN REGEXP_CONTAINS(LOWER(d.long_title), r'colectomy|cholecystectomy|laparotomy|hernia|gastrectomy|appendectomy') THEN 'abdominal'
    END AS category
  FROM `physionet-data.mimiciii_clinical.procedures_icd` p
  JOIN `physionet-data.mimiciii_clinical.d_icd_procedures` d USING(icd9_code)
),
qualified AS (
  SELECT hadm_id
  FROM `physionet-data.mimiciii_notes.noteevents`
  WHERE hadm_id IS NOT NULL
  GROUP BY hadm_id
  HAVING COUNT(*) >= 5
     AND COUNTIF(LOWER(category) = 'discharge summary') >= 1
),
sampled AS (
  SELECT s.subject_id, s.hadm_id, s.category
  FROM surgical s
  JOIN qualified q USING(hadm_id)
  WHERE s.category IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY s.category
    ORDER BY FARM_FINGERPRINT(CAST(s.hadm_id AS STRING))
  ) <= 10
)
SELECT
  n.subject_id, n.hadm_id, n.chartdate, n.category AS note_category,
  n.description, n.text, sam.category AS surgical_category
FROM sampled sam
JOIN `physionet-data.mimiciii_notes.noteevents` n USING(hadm_id)
"""

print("Running cohort query on BigQuery...")
df = client.query(sql).to_dataframe()
print(f"Pulled {len(df):,} notes across {df['hadm_id'].nunique()} admissions")
print("\nAdmissions per surgical category:")
print(df.groupby('surgical_category')['hadm_id'].nunique())

df.columns = [c.upper() for c in df.columns]
df.rename(columns={'NOTE_CATEGORY': 'CATEGORY', 'SURGICAL_CATEGORY': 'SURGICAL'}, inplace=True)

df["CHARTDATE"] = df["CHARTDATE"].astype(str)
df.to_parquet(OUT / "cohort_notes.parquet", index=False)
print(f"\nSaved: {OUT/'cohort_notes.parquet'}")
