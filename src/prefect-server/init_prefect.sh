#!/bin/bash
set -e

# Create a work pool named LSTM_forecasting if it doesn't exist
prefect work-pool inspect LSTM_forecasting >/dev/null 2>&1 || \
prefect work-pool create LSTM_forecasting --type process

# Work pool 'LSTM_forecasting' is ready.

# Create a work pool named metrics_ingestion if it doesn't exist
prefect work-pool inspect metrics_ingestion >/dev/null 2>&1 || \
prefect work-pool create metrics_ingestion --type process

# Work pool 'metrics_ingestion' is ready.

# Create a work pool named postgres_retention if it doesn't exist
prefect work-pool inspect postgres_retention >/dev/null 2>&1 || \
prefect work-pool create postgres_retention --type process

# Work pool 'postgres_retention' is ready.

echo "Work pools 'LSTM_forecasting', 'metrics_ingestion' and 'postgres_retention' ready."
