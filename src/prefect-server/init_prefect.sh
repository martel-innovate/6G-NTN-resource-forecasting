#!/bin/bash
set -e

# Create a work pool named LSTM_forecasting if it doesn't exist
prefect work-pool inspect LSTM_forecasting >/dev/null 2>&1 || \
prefect work-pool create LSTM_forecasting --type process

echo "Work pool 'LSTM_forecasting' is ready."
