from google.cloud import bigquery
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

PROJECT = "sds-ss-2026q2"
OUT = Path.home() / "Desktop/HealthcareAI_Stanford/results"
OUT.mkdir(parents=True, exist_ok=True)
client = bigquery.Client(project=PROJECT)

sql = """
WITH surgical AS (
  SELECT DISTINCT p.subject_id, p.hadm_id,
    CASE
      WHEN REGEXP_CONTAINS(LOWER(d.long_title), r'coronary|cabg|valve|cardiac|aortic') THEN 'cardiac'
      WHEN REGEXP_CONTAINS(LOWER(d.long_title), r'arthroplasty|hip|knee|fracture|spinal fusion|femur|tibia') THEN 'orthopedic'
      WHEN REGEXP_CONTAINS(LOWER(d.long_title), r'endarterectomy|aneurysm|angioplasty') THEN 'vascular'
      WHEN REGEXP_CONTAINS(LOWER(d.long_title), r'craniotomy|craniectomy|laminectomy') THEN 'neuro'
      WHEN REGEXP_CONTAINS(LOWER(d.long_title), r'colectomy|cholecystectomy|laparotomy|hernia|gastrectomy|appendectomy') THEN 'abdominal'
    END AS category
  FROM `physionet-data.mimiciii_clinical.procedures_icd` p
  JOIN `physionet-data.mimiciii_clinical.d_icd_procedures` d USING(icd9_code)
),
agg AS (
  SELECT hadm_id,
    COUNT(*) AS n_notes,
    DATETIME_DIFF(MAX(COALESCE(charttime, CAST(chartdate AS DATETIME))),
                  MIN(COALESCE(charttime, CAST(chartdate AS DATETIME))), HOUR) AS span_hours,
    COUNTIF(LOWER(category)='discharge summary') AS n_ds,
    COUNTIF(REGEXP_CONTAINS(LOWER(COALESCE(description,'')), r'post[- ]?op|operative')
            OR REGEXP_CONTAINS(LOWER(text), r'post[- ]?operative|post[- ]?op note')) AS n_postop
  FROM `physionet-data.mimiciii_notes.noteevents`
  WHERE hadm_id IS NOT NULL
  GROUP BY hadm_id
),
qualified AS (
  SELECT hadm_id FROM agg
  WHERE n_notes >= 5 AND span_hours >= 12 AND n_ds >= 1 AND n_postop >= 1
),
sampled AS (
  SELECT s.subject_id, s.hadm_id, s.category
  FROM surgical s JOIN qualified q USING(hadm_id)
  WHERE s.category IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY s.category
    ORDER BY FARM_FINGERPRINT(CAST(s.hadm_id AS STRING))) <= 8
)
SELECT n.subject_id, n.hadm_id, CAST(n.chartdate AS STRING) AS chartdate,
       n.category AS note_category, n.description, n.text, sam.category AS surgical
FROM sampled sam JOIN `physionet-data.mimiciii_notes.noteevents` n USING(hadm_id)
"""

print("Running cohort v2 query...")
df = client.query(sql).to_dataframe()
df.columns = [c.upper() for c in df.columns]
df["CATEGORY"] = df["NOTE_CATEGORY"].fillna("").str.strip()

# Split: ground-truth DS vs source notes
ds_mask = df.CATEGORY.str.lower() == "discharge summary"
ground_truth = df[ds_mask].groupby("HADM_ID").agg(
    text=("TEXT", lambda x: "\n\n".join(x)),
    surgical=("SURGICAL", "first")).reset_index()
source_notes = df[~ds_mask].copy()

print(f"Admissions: {df.HADM_ID.nunique()} | Source notes: {len(source_notes)} | Ground-truth DS: {len(ground_truth)}")
print("\nPer surgical category:")
print(df.groupby('SURGICAL').HADM_ID.nunique())

# Save as arrow-native to avoid dbdate issues
pq.write_table(pa.Table.from_pandas(source_notes), OUT/"cohort_notes_v2.parquet")
pq.write_table(pa.Table.from_pandas(ground_truth), OUT/"ground_truth_ds.parquet")
print(f"\nSaved: cohort_notes_v2.parquet, ground_truth_ds.parquet")
