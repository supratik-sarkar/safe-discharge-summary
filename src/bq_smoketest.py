from google.cloud import bigquery

PROJECT = "sds-ss-2026q2"

client = bigquery.Client(project=PROJECT)
sql = """
SELECT name, SUM(number) AS n
FROM `bigquery-public-data.usa_names.usa_1910_2013`
WHERE state = 'CA'
GROUP BY name
ORDER BY n DESC
LIMIT 5
"""
df = client.query(sql).to_dataframe()
print(df)
