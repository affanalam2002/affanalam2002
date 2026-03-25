from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig, TrainingConfig
from .data_simulator import save_dataset
from .model import build_lstm_model
from .preprocess import build_windows, scale_windows


def train_pipeline(
    data_cfg: DataConfig = DataConfig(),
    train_cfg: TrainingConfig = TrainingConfig(),
    output_dir: str = "models",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_path = output_path / "synthetic_sensor_data.csv"
    df = save_dataset(str(dataset_path), data_cfg)

    X, y = build_windows(df, train_cfg)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=train_cfg.test_size + train_cfg.val_size,
        random_state=data_cfg.random_seed,
        stratify=y,
    )

    val_fraction_of_temp = train_cfg.val_size / (train_cfg.test_size + train_cfg.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - val_fraction_of_temp,
        random_state=data_cfg.random_seed,
        stratify=y_temp,
    )

    X_train, X_val, X_test, scaler = scale_windows(X_train, X_val, X_test)

    model = build_lstm_model(
        input_shape=(train_cfg.window_size, X_train.shape[-1]),
        learning_rate=train_cfg.learning_rate,
    )

    callbacks = []
    try:
        import tensorflow as tf

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=4, restore_best_weights=True
            )
        ]
    except ImportError:
        pass

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    eval_dict = model.evaluate(X_test, y_test, return_dict=True, verbose=0)

    model_path = output_path / "predictive_maintenance_lstm.keras"
    scaler_path = output_path / "scaler.joblib"
    metadata_path = output_path / "metadata.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "sensor_columns": ["temperature", "vibration", "pressure", "rpm", "humidity"],
        "window_size": train_cfg.window_size,
        "prediction_horizon": train_cfg.prediction_horizon,
        "metrics": {k: float(v) for k, v in eval_dict.items()},
    }
    pd.Series(metadata).to_json(metadata_path)

    return {
        "history": history.history,
        "evaluation": eval_dict,
        "artifacts": {
            "model": str(model_path),
            "scaler": str(scaler_path),
            "metadata": str(metadata_path),
            "dataset": str(dataset_path),
        },
    }


if __name__ == "__main__":
    result = train_pipeline()
    print("Training complete.")
    print("Evaluation:", {k: round(v, 4) for k, v in result["evaluation"].items()})
    print("Artifacts:", result["artifacts"])
