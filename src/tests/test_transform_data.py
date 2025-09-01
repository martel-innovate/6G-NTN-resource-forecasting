import pytest
from datetime import datetime, timezone
from scripts.prometheus_to_postgres import transform_data, METRIC


def make_entry(pod_name, metric, values):
    """Helper to build a Prometheus query result entry."""
    global METRIC
    METRIC = metric  # temporarily override the global for testing
    return {
        "metric": {"pod": pod_name},
        "values": values
    }

def test_transform_data_cpu():
    entry = make_entry("upf2-open5gs-upf-1", "container_cpu_usage_seconds_total", [[1234567890, "0.5"]])
    result = transform_data.fn([entry], METRIC)  # call original function ignoring Prefect decorator
    assert len(result) == 1
    name, value, ts = result[0]
    assert name == "cpu_usage_upf"
    assert value == 0.5
    assert isinstance(ts, datetime)

def test_transform_data_memory():
    entry = make_entry("upf2-open5gs-upf-1", "container_memory_usage_bytes", [[1234567890, "1024"]])
    result = transform_data.fn([entry], METRIC)
    assert len(result) == 1
    name, value, ts = result[0]
    assert name == "memory_usage_upf"
    assert value == 1024.0
    assert isinstance(ts, datetime)

def test_transform_data_unrecognized_pod():
    entry = make_entry("unknown-pod", "container_cpu_usage_seconds_total", [[1234567890, "0.5"]])
    with pytest.raises(ValueError, match="Unrecognized VNF"):
        transform_data.fn([entry], METRIC)

def test_transform_data_unrecognized_metric():
    entry = make_entry("upf2-open5gs-upf-1", "weird_metric", [[1234567890, "0.5"]])
    with pytest.raises(ValueError, match="Unrecognized metric_name"):
        transform_data.fn([entry], METRIC)

def test_transform_data_multiple_values():
    entry = make_entry("upf2-open5gs-upf-1", "container_cpu_usage_seconds_total", [
        [1234567890, "0.5"],
        [1234567950, "0.7"]
    ])
    result = transform_data.fn([entry], METRIC)
    assert len(result) == 2
    names = [r[0] for r in result]
    values = [r[1] for r in result]
    timestamps = [r[2] for r in result]
    assert all(name == "cpu_usage_upf" for name in names)
    assert values == [0.5, 0.7]
    assert all(isinstance(ts, datetime) for ts in timestamps)

def test_transform_data_realistic_cpu_result():
    entry = {'metric': {'namespace': 'open5gs', 'pod': 'upf2-open5gs-upf-76c9c98c84-tfwcp'}, 'values': [[1756733725, '0.00011235605641296914'], [1756733785, '0.00011189600631986969'], [1756733845, '0.00013826667811075007']]}
    
    result = transform_data.fn([entry], "container_cpu_usage_seconds_total")

    assert len(result) == 3