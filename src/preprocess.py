from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import TrainingConfig
from .data_simulator import SENSOR_COLUMNS


def build_windows(df: pd.DataFrame, cfg: TrainingConfig):
    """Create rolling windows and labels for imminent failure prediction.

    Label = 1 if failure occurs in the next prediction_horizon timesteps.
    """
    features: list[np.ndarray] = []
    labels: list[int] = []

    for _, machine_df in df.groupby("machine_id"):
        machine_df = machine_df.sort_values("timestep")
        sensor_values = machine_df[SENSOR_COLUMNS].to_numpy(dtype=np.float32)
        failed_values = machine_df["failed"].to_numpy(dtype=np.int32)

        max_start = len(machine_df) - cfg.window_size - cfg.prediction_horizon
        for start in range(max_start):
            end = start + cfg.window_size
            horizon_end = end + cfg.prediction_horizon
            window = sensor_values[start:end]
            future_failed = failed_values[end:horizon_end].max()
            features.append(window)
            labels.append(int(future_failed))

    X = np.stack(features)
    y = np.array(labels, dtype=np.float32)
    return X, y


def scale_windows(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    """Fit scaler on training windows and transform all sets."""
    scaler = StandardScaler()

    n_train, w, n_feat = X_train.shape
    X_train_2d = X_train.reshape(n_train * w, n_feat)
    scaler.fit(X_train_2d)

    def _transform(X: np.ndarray) -> np.ndarray:
        n, window_size, f = X.shape
        X_2d = X.reshape(n * window_size, f)
        X_scaled = scaler.transform(X_2d)
        return X_scaled.reshape(n, window_size, f)

    return _transform(X_train), _transform(X_val), _transform(X_test), scaler
