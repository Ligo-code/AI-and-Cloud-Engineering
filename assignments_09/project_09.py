# Video link: TODO

import json
from datetime import date

import pandas as pd
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient


ACCOUNT_URL = "https://kseniiactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

LATITUDE = 35.2271
LONGITUDE = -80.8431

credential = DefaultAzureCredential()

container = ContainerClient(
    account_url=ACCOUNT_URL,
    container_name=CONTAINER,
    credential=credential,
) # Python goes directly to https://kseniiactd2026sa.blob.core.windows.net

print("Connected to Azure Blob Storage")

# Extract
url = (
    f"https://api.open-meteo.com/v1/forecast"
    f"?latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&hourly=temperature_2m,precipitation"
    f"&forecast_days=7"
)

response = requests.get(url)
response.raise_for_status()

data = response.json()

print("Weather data downloaded")
print(data.keys())

# Serialize
payload = json.dumps(data).encode("utf-8")

print(f"Serialized data size: {len(payload)} bytes")

# Load
today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

container.upload_blob(
    blob_path,
    payload,
    overwrite=True,
)

print(
    f"Uploaded {len(payload)} bytes to {blob_path}"
)

# Verify
print("\nBlobs in container:")

for blob in container.list_blobs():
    print(f"{blob.name} ({blob.size} bytes)")

# Read Back
raw = container.download_blob(blob_path).readall()

data_back = json.loads(raw.decode("utf-8"))

df = pd.DataFrame(data_back["hourly"])

print("\nFirst 5 rows:")
print(df.head())

# Save downloaded file locally
with open(
    "assignments_09/outputs/weather_raw.json",
    "wb" # write binary
) as f:
    f.write(raw)

print("\nSaved JSON to outputs/weather_raw.json")