import os
import pytest
from scripts.prometheus_to_postgres import fetch_prometheus_data

@pytest.mark.parametrize("metric_name", [
    "container_cpu_usage_seconds_total",
    "container_memory_usage_bytes"
])
def test_fetch_prometheus_data(metric_name):
    # Ensure required env vars exist
    prometheus_host = os.environ.get("PROMETHEUS_HOSTNAME")
    assert prometheus_host is not None, "PROMETHEUS_HOSTNAME must be set"
    assert metric_name is not None, "metric_name must be set"

    # Call the original function ignoring the Prefect decorator
    result = fetch_prometheus_data.fn(metric_name)  # pass metric_name explicitly

    # Basic checks
    assert isinstance(result, list), "Result should be a list"
    assert result is not None

    if len(result) > 0:
        # Example: check structure of the first result
        first = result[0]
        assert "metric" in first
        assert "values" in first
        print(f"Metric: {metric_name} | First entry: {first} | Total entries: {len(result)}")
