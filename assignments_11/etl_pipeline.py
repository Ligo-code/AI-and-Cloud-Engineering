# VIDEO LINK: https://youtu.be/qCO_Dd8YMos

from datetime import date
import os
import requests
from dotenv import load_dotenv
from prefect import flow, task
from openai import OpenAI
import json
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient

load_dotenv()

CITY = "New York City"
LATITUDE = 40.7128
LONGITUDE = -74.0060

ACCOUNT_URL = "https://kseniiactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

MAX_RECORDS = 24

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}


@task(retries=2, retry_delay_seconds=10)
def extract(latitude: float, longitude: float, city: str) -> dict:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
    )

    response = requests.get(url)
    response.raise_for_status()

    print(f"Extracted 7-day hourly forecast data for {city} ({latitude}, {longitude})")

    return response.json()


@task
def transform(data: dict) -> list:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    hourly = data["hourly"]

    records = []

    for i in range(min(MAX_RECORDS, len(hourly["time"]))):
        record = {
            "city": CITY,
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        }

        records.append(record)

    enriched = []

    for i, record in enumerate(records):
        user_message = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        raw_label = response.choices[0].message.content.strip().lower()

        if raw_label in VALID_LABELS:
            label = raw_label
        else:
            label = "unknown"

        enriched.append(
            {
                **record,
                "running_conditions": label,
            }
        )

        if (i + 1) % 6 == 0:
            print(f"Classified {i + 1}/{len(records)} records")

    print(f"Transform complete: {len(enriched)} records enriched")

    return enriched

@task
def load(records: list, blob_path: str) -> None:
    credential = DefaultAzureCredential()

    container = ContainerClient(
        ACCOUNT_URL,
        CONTAINER,
        credential=credential,
    )

    payload = json.dumps(records, indent=2).encode("utf-8")

    container.upload_blob(
        blob_path,
        payload,
        overwrite=True,
    )

    print(f"Loaded {len(records)} records to {blob_path}")
    print(f"Uploaded {len(payload)} bytes")


@flow(log_prints=True)
def etl_pipeline():
    today = date.today().isoformat()

    blob_path = f"final/{today}/weather_etl.json"

    data = extract(LATITUDE, LONGITUDE, CITY)

    enriched = transform(data)

    load(enriched, blob_path)

    print(f"Pipeline complete. Results at {blob_path}")



if __name__ == "__main__":
    etl_pipeline()