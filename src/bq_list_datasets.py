from google.cloud import bigquery
client = bigquery.Client(project="sds-ss-2026q2")
datasets = list(client.list_datasets("physionet-data"))
print(f"Accessible datasets in physionet-data: {len(datasets)}")
for d in sorted(datasets, key=lambda x: x.dataset_id):
    print(f"  {d.dataset_id}")
