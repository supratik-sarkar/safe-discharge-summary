from google.cloud import bigquery

PROJECT = "sds-ss-2026q2"
client = bigquery.Client(project=PROJECT)

try:
    df = client.query("""
        SELECT COUNT(*) AS n
        FROM `physionet-data.mimiciii_notes.noteevents`
    """).to_dataframe()
    print(f"ACCESS GRANTED — noteevents row count: {df['n'].iloc[0]:,}")
except Exception as e:
    print("ACCESS NOT YET GRANTED")
    print(f"Error: {type(e).__name__}: {str(e)[:300]}")
