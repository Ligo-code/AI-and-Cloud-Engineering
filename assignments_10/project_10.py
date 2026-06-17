
# Video link: TODO

"""
Reflection:
Classifying weather conditions for outdoor running is an okay use of an LLM because the words
"good", "marginal", and "bad" involve some judgment. However, this specific task could probably
be handled better with deterministic code because temperature and precipitation are already
structured numeric values. A rule-based approach would be faster, cheaper, and more consistent.
The tradeoff is that rules are less flexible and may miss edge cases that a human-like judgment
could handle more naturally.
"""

import json
import os
from pathlib import Path

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() 

ACCOUNT_URL = "https://kseniiactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
DATA_DATE = "2026-06-16" # use date for the uploded dataset from the week9 assignment

VALID_LABELS = {"good", "marginal", "bad"}

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

def make_user_message(record):
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )


def reshape_hourly_data(data):
    hourly = data["hourly"]
    records = []

    for i in range(len(hourly["time"])):
        records.append(
            {
                "time": hourly["time"][i],
                "temperature_2m": hourly["temperature_2m"][i],
                "precipitation": hourly["precipitation"][i],
            }
        )

    return records

def load_weather_data(container, today):
    blob_path = f"raw/{today}/weather.json"

    try:
        raw = container.download_blob(blob_path).readall()
        print(f"Loaded raw data from Blob Storage: {blob_path}")
        return json.loads(raw.decode("utf-8"))

    except Exception as error:
        print(f"Could not load {blob_path} from Blob Storage.")
        print(f"Using fallback dataset instead. Error: {error}")

        fallback_path = Path("assignments/resources/weather_raw.json")

        with fallback_path.open("r", encoding="utf-8") as file:
            return json.load(file)

def classify_records(records):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    enriched = []

    for i, record in enumerate(records[:24]):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_message(record)},
            ],
        )

        raw_label = response.choices[0].message.content.strip().lower()
        label = raw_label if raw_label in VALID_LABELS else "unknown"

        enriched.append({**record, "conditions": label})

        if (i + 1) % 6 == 0:
            print(f"Processed {i + 1} records...")

    return enriched

def upload_enriched_data(container, enriched, today):
    processed_path = f"processed/{today}/weather_classified.json"

    payload = json.dumps(enriched, indent=2).encode("utf-8")

    container.upload_blob(
        name=processed_path,
        data=payload,
        overwrite=True,
    )

    print(f"\nUploaded processed data to Blob Storage: {processed_path}")

def spot_check(enriched):
    df = pd.DataFrame(enriched)

    print("\nCondition counts:")
    print(df["conditions"].value_counts())

    print("\nFirst 5 rows:")
    print(df.head())

def save_first_10_records(enriched):
    output_path = Path("assignments_10/outputs/first_10_records.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(enriched[:10], file, indent=2)

    print(f"\nSaved first 10 records to {output_path}")

def main():
    today = DATA_DATE

    credential = DefaultAzureCredential()

    container = ContainerClient(
        account_url=ACCOUNT_URL,
        container_name=CONTAINER,
        credential=credential,
    )

    data = load_weather_data(container, today)

    records = reshape_hourly_data(data)

    print(f"Loaded {len(records)} hourly records")

    print("\nFirst record:")
    print(records[0])

    enriched = classify_records(records)

    print("\nFirst enriched record:")
    print(enriched[0])

    upload_enriched_data(container, enriched, today)
    spot_check(enriched)
    save_first_10_records(enriched)

if __name__ == "__main__":
    main()