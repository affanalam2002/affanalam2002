from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DataConfig


SENSOR_COLUMNS = ["temperature", "vibration", "pressure", "rpm", "humidity"]


def _machine_health_signal(length: int, failure_point: int, rng: np.random.Generator) -> np.ndarray:
    """Creates a degradation signal that worsens near failure."""
    x = np.arange(length)
    distance_to_failure = np.clip(failure_point - x, a_min=0, a_max=None)
    degradation = np.exp(-distance_to_failure / 35.0)
    noise = rng.normal(0, 0.02, size=length)
    return np.clip(degradation + noise, 0, 1)


def generate_synthetic_data(config: DataConfig) -> pd.DataFrame:
    """Generate synthetic multivariate sensor data with failure labels.

    Returns one row per machine-timepoint.
    """
    rng = np.random.default_rng(config.random_seed)
    frames: list[pd.DataFrame] = []

    for machine_id in range(config.n_machines):
        length = config.timesteps_per_machine
        failure_point = int(rng.integers(int(length * 0.6), length))
        health_signal = _machine_health_signal(length, failure_point, rng)

        temperature = 65 + 35 * health_signal + rng.normal(0, 1.4, size=length)
        vibration = 0.25 + 1.8 * health_signal + rng.normal(0, 0.08, size=length)
        pressure = 85 - 22 * health_signal + rng.normal(0, 1.2, size=length)
        rpm = 1500 - 320 * health_signal + rng.normal(0, 20, size=length)
        humidity = 40 + 16 * health_signal + rng.normal(0, 2.0, size=length)

        timesteps = np.arange(length)
        failed = (timesteps >= failure_point).astype(int)

        machine_df = pd.DataFrame(
            {
                "machine_id": machine_id,
                "timestep": timesteps,
                "temperature": temperature,
                "vibration": vibration,
                "pressure": pressure,
                "rpm": rpm,
                "humidity": humidity,
                "failed": failed,
            }
        )
        frames.append(machine_df)

    df = pd.concat(frames, ignore_index=True)
    return df


def save_dataset(path: str, config: DataConfig) -> pd.DataFrame:
    df = generate_synthetic_data(config)
    df.to_csv(path, index=False)
    return df
