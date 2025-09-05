import pytest
import pandas as pd
from datetime import datetime, timedelta
from darts import TimeSeries
from scripts.distributed_LSTM_univariate import preprocessing, data_transformation, split_dataset, normalize_series

@pytest.fixture
def sample_df():
    """
    Create a realistic sample dataframe with multiple metrics.
    Timestamps every 5 seconds.
    """
    base_time = datetime(2025, 9, 5, 0, 0, 0)
    timestamps = [base_time + timedelta(seconds=5*i) for i in range(50)]
    data = {
        "datetime": timestamps * 2,  # two metrics
        "metric_name": ["metric1"]*50 + ["metric2"]*50,
        "value": list(range(50)) + list(range(50, 100)),
    }
    df = pd.DataFrame(data)
    return df

def test_preprocessing(sample_df):
    """
    Integration test for preprocessing() task using real function calls.
    Bypasses Prefect orchestration by calling `.fn()` for all tasks.
    """
    # Step 1: Data transformation
    df_resampled = data_transformation.fn(sample_df)

    # Step 2: Train/test split
    series, train, test = split_dataset.fn(df_resampled)

    # Step 3: Normalization
    series_transformed, train_transformed, val_transformed = normalize_series.fn(series, train, test)

    # Combine results like preprocessing()
    data_transformed = {
        'series': series_transformed,
        'train': train_transformed,
        'val': val_transformed
    }

    # Assertions
    assert isinstance(data_transformed, dict)
    assert "series" in data_transformed and "train" in data_transformed and "val" in data_transformed

    assert isinstance(data_transformed["series"], TimeSeries)
    assert isinstance(data_transformed["train"], TimeSeries)
    assert isinstance(data_transformed["val"], TimeSeries)

    # Total length check
    total_length = len(data_transformed["series"])
    train_length = len(data_transformed["train"])
    val_length = len(data_transformed["val"])
    assert total_length == train_length + val_length

    # Values are finite
    import numpy as np
    series_values = data_transformed["series"].values()
    assert not np.isnan(series_values).any(), "NaN values found after preprocessing"
