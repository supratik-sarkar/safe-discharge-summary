"""MIMIC-IV v3.1 Multi-Morbidity ICU Cohort Builder (BigQuery)."""
from google.cloud import bigquery
from pathlib import Path
import pandas as pd

PROJECT = "sds-ss-2026q2"
ROOT = Path.home() / "Desktop" / "HealthcareAI_Stanford"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

client = bigquery.Client(project=PROJECT)

SQL = """
WITH base_icu AS (
    SELECT icu.subject_id, icu.hadm_id, icu.stay_id, icu.intime, icu.outtime,
           ROUND(icu.los, 2) AS icu_los_days, pat.anchor_age, pat.gender,
           adm.admittime, adm.dischtime, adm.hospital_expire_flag, adm.discharge_location,
           ROW_NUMBER() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime ASC) AS stay_rank
    FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
    JOIN `physionet-data.mimiciv_3_1_hosp.patients` pat ON icu.subject_id = pat.subject_id
    JOIN `physionet-data.mimiciv_3_1_hosp.admissions` adm ON icu.hadm_id = adm.hadm_id
    WHERE pat.anchor_age >= 18 AND icu.los > 1.0
),
first_stays AS (SELECT * FROM base_icu WHERE stay_rank = 1),
cci_flags AS (
    SELECT hadm_id,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^I2[12]|^I252') THEN 1 ELSE 0 END) AS mi,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^I50|^I43|^I110|^I130|^I132') THEN 1 ELSE 0 END) AS chf,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^I7[01]|^I73|^K551') THEN 1 ELSE 0 END) AS pvd,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^G4[56]|^H340|^I6') THEN 1 ELSE 0 END) AS cvd,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^F0[0-3]|^F051|^G30|^G311') THEN 1 ELSE 0 END) AS dementia,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^J4[0-7]|^J6[0-7]|^J684|^J701|^J703') THEN 1 ELSE 0 END) AS copd,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^M0[5689]|^M3[2-6]|^M45') THEN 1 ELSE 0 END) AS ctd,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^K2[5-8]') THEN 1 ELSE 0 END) AS pud,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^K7[034]|^B18') THEN 1 ELSE 0 END) AS mild_liver,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^E1[0-4][016890]') THEN 1 ELSE 0 END) AS dm_no_cc,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^G8[1-3]') THEN 1 ELSE 0 END) AS hemiplegia,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^N1[89]|^I120|^I131') THEN 1 ELSE 0 END) AS renal,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^E1[0-4][2-57]') THEN 1 ELSE 0 END) AS dm_cc,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^C[0-7][0-6]|^C8[1-58]|^C9[0-7]')
                 AND NOT REGEXP_CONTAINS(icd_code, r'^C7[7-9]|^C80') THEN 1 ELSE 0 END) AS cancer,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^K704|^K721|^K76[567]|^I850') THEN 1 ELSE 0 END) AS severe_liver,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^C7[7-9]|^C80') THEN 1 ELSE 0 END) AS metastatic,
        MAX(CASE WHEN REGEXP_CONTAINS(icd_code, r'^B2[0-24]') THEN 1 ELSE 0 END) AS hiv
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    WHERE icd_version = 10
    GROUP BY hadm_id
),
cci_scores AS (
    SELECT hadm_id,
        (mi+chf+pvd+cvd+dementia+copd+ctd+pud+mild_liver+dm_no_cc)
        + (hemiplegia+renal+dm_cc+cancer)*2 + (severe_liver)*3 + (metastatic+hiv)*6 AS cci_score,
        mi,chf,pvd,cvd,dementia,copd,ctd,pud,mild_liver,dm_no_cc,
        hemiplegia,renal,dm_cc,cancer,severe_liver,metastatic,hiv
    FROM cci_flags
),
has_notes AS (SELECT DISTINCT hadm_id FROM `physionet-data.mimiciv_note.discharge`),
has_labs AS (SELECT DISTINCT hadm_id FROM `physionet-data.mimiciv_3_1_hosp.labevents` WHERE hadm_id IS NOT NULL),
has_meds AS (SELECT DISTINCT hadm_id FROM `physionet-data.mimiciv_3_1_hosp.prescriptions` WHERE hadm_id IS NOT NULL),
has_vitals AS (
    SELECT DISTINCT icu.hadm_id
    FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
    JOIN `physionet-data.mimiciv_3_1_icu.icustays` icu ON ce.stay_id = icu.stay_id
    WHERE ce.valuenum IS NOT NULL
),
has_procedures AS (SELECT DISTINCT hadm_id FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd` WHERE hadm_id IS NOT NULL),
final_cohort AS (
    SELECT f.subject_id, f.hadm_id, f.stay_id, f.anchor_age, f.gender,
           f.intime AS icu_intime, f.outtime AS icu_outtime, f.icu_los_days,
           f.admittime, f.dischtime, f.hospital_expire_flag, f.discharge_location,
           c.cci_score, c.mi, c.chf, c.pvd, c.cvd, c.dementia, c.copd,
           c.ctd, c.pud, c.mild_liver, c.dm_no_cc, c.hemiplegia, c.renal,
           c.dm_cc, c.cancer, c.severe_liver, c.metastatic, c.hiv
    FROM first_stays f
    JOIN cci_scores c ON f.hadm_id = c.hadm_id
    JOIN has_notes n ON f.hadm_id = n.hadm_id
    JOIN has_labs lb ON f.hadm_id = lb.hadm_id
    JOIN has_meds m ON f.hadm_id = m.hadm_id
    JOIN has_vitals v ON f.hadm_id = v.hadm_id
    JOIN has_procedures p ON f.hadm_id = p.hadm_id
    WHERE c.cci_score >= 3
)
SELECT *,
    DATE_DIFF(DATE(dischtime), DATE(admittime), DAY) AS hospital_los_days,
    CASE WHEN cci_score BETWEEN 3 AND 4 THEN 'moderate'
         WHEN cci_score BETWEEN 5 AND 6 THEN 'high'
         WHEN cci_score >= 7 THEN 'very_high' END AS complexity_tier
FROM final_cohort
ORDER BY cci_score DESC, icu_los_days DESC
"""

print("Running MIMIC-IV cohort query (30-90s, scans ~10-15GB)...")
df = client.query(SQL).to_dataframe()

# Cast BigQuery date types to plain strings to avoid parquet round-trip issues
for col in df.columns:
    if df[col].dtype.name in ("dbdate", "dbtime", "dbtimestamp"):
        df[col] = df[col].astype(str)

print(f"\n{'='*55}")
print(f"COHORT SUMMARY")
print(f"{'='*55}")
print(f"Total patients: {len(df):,}")
print(f"\nDemographics:")
print(f"  Age:    mean={df['anchor_age'].mean():.1f}  median={df['anchor_age'].median():.0f}")
print(f"  Gender: {df['gender'].value_counts().to_dict()}")
print(f"\nICU LOS (days): mean={df['icu_los_days'].mean():.1f}  median={df['icu_los_days'].median():.1f}")
print(f"Hospital LOS:   mean={df['hospital_los_days'].mean():.1f}  median={df['hospital_los_days'].median():.1f}")
print(f"\nCCI score: mean={df['cci_score'].mean():.1f}  range=[{df['cci_score'].min()}, {df['cci_score'].max()}]")
print(f"Tiers: {df['complexity_tier'].value_counts().to_dict()}")
print(f"In-hospital mortality: {df['hospital_expire_flag'].mean()*100:.1f}%")

cci_cols = ["mi","chf","pvd","cvd","dementia","copd","ctd","pud","mild_liver","dm_no_cc",
            "hemiplegia","renal","dm_cc","cancer","severe_liver","metastatic","hiv"]
top5 = sorted([(c, int(df[c].sum())) for c in cci_cols], key=lambda x: -x[1])[:5]
print(f"\nTop 5 comorbidities:")
for name, count in top5:
    print(f"  {name:14s} {count:6,}  ({count/len(df)*100:5.1f}%)")

df.to_parquet(OUT / "mimiciv_cohort.parquet", index=False)
df.to_csv(OUT / "mimiciv_cohort.csv", index=False)
print(f"\nSaved to: {OUT}/mimiciv_cohort.{{parquet,csv}}")

assert len(df) > 0 and df['cci_score'].min() >= 3 and df["icu_los_days"].min() >= 1.0
print("Sanity checks passed.")
