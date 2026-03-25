from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _load_tf_model(model_path: str):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required for inference.") from exc
    return tf.keras.models.load_model(model_path)


def predict_failure_risk(input_csv: str, model_dir: str = "models") -> float:
    model_dir_path = Path(model_dir)
    metadata = json.loads((model_dir_path / "metadata.json").read_text())
    sensor_cols = metadata["sensor_columns"]
    window_size = int(metadata["window_size"])

    df = pd.read_csv(input_csv)
    if len(df) < window_size:
        raise ValueError(f"Need at least {window_size} rows for inference.")

    window = df[sensor_cols].tail(window_size).to_numpy(dtype=np.float32)

    scaler = joblib.load(model_dir_path / "scaler.joblib")
    model = _load_tf_model(str(model_dir_path / "predictive_maintenance_lstm.keras"))

    window_scaled = scaler.transform(window).reshape(1, window_size, len(sensor_cols))
    risk = float(model.predict(window_scaled, verbose=0)[0][0])
    return risk


def main():
    parser = argparse.ArgumentParser(description="Predict imminent failure risk")
    parser.add_argument("--input", required=True, help="CSV file with recent sensor readings")
    parser.add_argument("--model-dir", default="models", help="Directory containing model artifacts")
    args = parser.parse_args()

    risk = predict_failure_risk(args.input, args.model_dir)
    print(f"Failure risk (next horizon): {risk:.3f}")


if __name__ == "__main__":
    main()
