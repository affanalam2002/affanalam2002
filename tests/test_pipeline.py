import unittest

from src.config import DataConfig, TrainingConfig
from src.data_simulator import SENSOR_COLUMNS, generate_synthetic_data
from src.preprocess import build_windows


class PipelineShapeTests(unittest.TestCase):
    def test_synthetic_data_has_expected_columns(self):
        cfg = DataConfig(n_machines=5, timesteps_per_machine=50, random_seed=0)
        df = generate_synthetic_data(cfg)
        expected_cols = {"machine_id", "timestep", "failed", *SENSOR_COLUMNS}
        self.assertTrue(expected_cols.issubset(set(df.columns)))
        self.assertEqual(df["machine_id"].nunique(), 5)

    def test_windowing_shapes(self):
        data_cfg = DataConfig(n_machines=3, timesteps_per_machine=80, random_seed=0)
        train_cfg = TrainingConfig(window_size=20, prediction_horizon=5)
        df = generate_synthetic_data(data_cfg)

        X, y = build_windows(df, train_cfg)
        self.assertEqual(X.shape[1], 20)
        self.assertEqual(X.shape[2], len(SENSOR_COLUMNS))
        self.assertEqual(X.shape[0], y.shape[0])


if __name__ == "__main__":
    unittest.main()
