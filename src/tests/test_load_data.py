import pytest
import pandas as pd
import os
from scripts.distributed_LSTM_univariate import load_data

@pytest.fixture
def real_logger():
    class DummyLogger:
        def __init__(self):
            self.messages = []

        def info(self, msg):
            print(msg)  # optionally print to console during test
            self.messages.append(msg)

    return DummyLogger()


@pytest.mark.parametrize("metric_name", [
    "cpu_usage_upf",
    "memory_usage_amf"
])
def test_load_data_real_db(real_logger, metric_name):
    # Ensure required env vars exist
    prometheus_host = os.environ.get("PROMETHEUS_HOSTNAME")
    assert prometheus_host is not None, "PROMETHEUS_HOSTNAME must be set"
    assert metric_name is not None, "metric_name must be set"

    # Call the Prefect task like a normal function
    df = load_data.fn(real_logger, metric_name)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "No data fetched — check the metric_name or DB content"

    # Check columns
    expected_columns = ["datetime", "metric_name", "value"]
    for col in expected_columns:
        assert col in df.columns, f"Column {col} missing in result"