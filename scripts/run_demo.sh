#!/usr/bin/env bash
set -euo pipefail

python -m src.train

echo "Use models/synthetic_sensor_data.csv to build a recent-readings CSV and run inference:"
echo "python -m src.infer --input recent_readings.csv --model-dir models"
