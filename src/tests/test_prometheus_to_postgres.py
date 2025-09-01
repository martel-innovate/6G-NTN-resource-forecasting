import os
import pytest
from scripts.prometheus_to_postgres import fetch_cpu_usage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DB_NAME=os.getenv('DB_NAME')
DB_USER=os.getenv('DB_USER')
DB_PASSWORD=os.getenv('DB_SECRET')
DB_HOST=os.getenv('DB_HOSTNAME')
DB_PORT=os.getenv('DB_PORT')
PROMETHEUS_HOSTNAME = os.getenv('PROMETHEUS_HOSTNAME')
METRIC = "container_cpu_usage_seconds_total" 
#METRIC = "container_memory_usage_bytes"



def test_fetch_cpu_usage():
    # Ensure required env vars exist
    prometheus_host = os.environ.get("PROMETHEUS_HOSTNAME")

    assert prometheus_host is not None, "PROMETHEUS_HOSTNAME must be set"
    assert METRIC is not None, "METRIC must be set"

    # Call the original function ignoring the Prefect decorator
    result = fetch_cpu_usage.fn()

    # Basic checks
    assert isinstance(result, list), "Result should be a list"
    # At least we should get a response (may be empty if Prometheus has no data yet)
    assert result is not None

    if len(result) > 0:
        # Example: check structure of the first result
        first = result[0]
        assert "metric" in first
        assert "values" in first
        print(f"First entry: {first}")
        print(f"Entry length: {len(result)}")