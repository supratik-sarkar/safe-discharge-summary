from google.cloud import bigquery
client = bigquery.Client(project="sds-ss-2026q2")
try:
    df = client.query("SELECT COUNT(*) AS n FROM `physionet-data.mimiciv_icu.icustays`").to_dataframe()
    print(f"MIMIC-IV ACCESS GRANTED — icustays rows: {df['n'].iloc[0]:,}")
except Exception as e:
    print(f"ACCESS NOT YET GRANTED: {type(e).__name__}: {str(e)[:200]}")
