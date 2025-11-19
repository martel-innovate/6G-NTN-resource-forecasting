#!/bin/bash
set -e

# Create a work pool named LSTM_forecasting if it doesn't exist
prefect work-pool inspect LSTM_forecasting >/dev/null 2>&1 || \
prefect work-pool create LSTM_forecasting --type process

echo "Work pool 'LSTM_forecasting' is ready."

# Create a work pool named metrics_ingestion if it doesn't exist
prefect work-pool inspect metrics_ingestion >/dev/null 2>&1 || \
prefect work-pool create metrics_ingestion --type process

echo "Work pool 'metrics_ingestion' is ready."
