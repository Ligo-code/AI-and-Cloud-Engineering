from prefect import task
from prefect.logging import get_run_logger


# =========================
# Prefect Orchestration
# =========================

# Prefect Question 1
"""
A @task is a single unit of work inside a pipeline.
It should usually do one focused operation, such as calling an API,
transforming data, or uploading a file.

A @flow is the orchestrator. It calls tasks in the correct order,
tracks the whole pipeline run, and shows the run state in the Prefect UI.

For a small helper function that only converts Celsius to Fahrenheit,
I would usually not decorate it with @task. It is a pure in-memory
calculation with no I/O, so making it a Prefect task would add unnecessary
orchestration overhead. I would keep it as a regular helper function unless
I specifically needed to track it in the Prefect UI.
"""


# Prefect Question 2
# @task(retries=3, retry_delay_seconds=30)


# Prefect Question 3
"""
If extract is Completed, transform is Failed, and load never ran,
I would click into the flow run in the Prefect UI and inspect the transform
task. I would open the Logs tab and look for the error message, traceback,
task state, and any printed progress messages before the failure.

I would expect to see what exception caused the failure, such as an API error,
missing environment variable, bad response format, or an issue inside the
transformation logic. Since load depends on transform, load would not run
because Prefect stops the downstream task when the upstream task fails.
"""


# =========================
# Production Patterns
# =========================

# Production Question 1
"""
raise_for_status() checks the HTTP response status code. If the response
has an error status like 404 or 500, it raises an exception.

This is better than only printing "error" because a printed message does
not stop the pipeline. If the API returns a 500 error and the code only
prints an error, the pipeline may continue with bad or missing data, and
downstream tasks might still run.

With raise_for_status(), Prefect sees the exception, marks the extract task
as Failed, records the failure in the logs, and does not run downstream tasks.
This is safer because the pipeline fails visibly instead of silently producing
incorrect output.
"""


# Production Question 2
"""
overwrite=True makes the load step idempotent. If the pipeline is re-run
after a crash or bug fix, the upload can replace the existing blob at the
same path instead of failing because the blob already exists.

In this scenario, the crash happened during transform, so the final blob may
not have been written yet. But if a previous run had already created the blob,
overwrite=True protects the rerun from failing during upload.

Without overwrite=True, Azure Blob Storage would reject the upload if a blob
already existed at final/{today}/weather_etl.json, and the rerun could fail
even after the original bug was fixed.
"""


# Production Question 3
@task
def load_records(records: list, blob_path: str) -> None:
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}")