from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """Configuration for synthetic manufacturing sensor data."""

    n_machines: int = 100
    timesteps_per_machine: int = 300
    random_seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for windowing and model training."""

    window_size: int = 30
    prediction_horizon: int = 10
    test_size: float = 0.2
    val_size: float = 0.2
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
