# Predictive Maintenance System (Senior Capstone Project)

A complete end-to-end machine learning project that simulates manufacturing sensor data and trains a TensorFlow LSTM model to predict near-term equipment failure.

## Project Goal

Use historical and live-like sensor telemetry from factory equipment to estimate whether a machine is likely to fail in the next few timesteps. This enables condition-based maintenance and reduces downtime.

## Tech Stack

- **Python** (data + ML pipeline)
- **TensorFlow / Keras** (LSTM model)
- **Pandas + NumPy** (data generation and wrangling)
- **Scikit-learn** (data splitting and scaling)

## Repository Structure

- `src/data_simulator.py` — synthetic sensor/failure data generation
- `src/preprocess.py` — rolling-window feature generation and scaling
- `src/model.py` — LSTM model architecture
- `src/train.py` — full training pipeline and artifact export
- `src/infer.py` — CLI inference for current failure risk prediction
- `tests/test_pipeline.py` — lightweight tests for data and preprocessing
- `models/` — trained model artifacts (after running training)

## Problem Formulation

- Input: a sequence window of machine sensor readings
- Output: probability of failure within a future prediction horizon
- Task type: binary classification (imminent failure vs no imminent failure)

## Sensors Simulated

- Temperature
- Vibration
- Pressure
- RPM
- Humidity

As degradation increases, the synthetic dataset introduces realistic shifts (e.g., temperature and vibration increase while pressure and RPM decrease).

## Quick Start

### 1) Create environment + install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train the model

```bash
python -m src.train
```

This produces:

- `models/predictive_maintenance_lstm.keras`
- `models/scaler.joblib`
- `models/metadata.json`
- `models/synthetic_sensor_data.csv`

### 3) Run inference on recent sensor readings

Prepare a CSV with at least `window_size` rows and columns:
`temperature,vibration,pressure,rpm,humidity`

Then run:

```bash
python -m src.infer --input path/to/recent_readings.csv --model-dir models
```

## Suggested Capstone Report Sections

1. **Introduction & Motivation**
2. **Literature Review (predictive maintenance, PHM, RUL)**
3. **System Design & Architecture**
4. **Dataset Engineering (synthetic + future real IoT data integration)**
5. **Modeling Approach (LSTM baseline, future Transformer comparison)**
6. **Evaluation Metrics (AUC, precision, recall, F1)**
7. **Deployment Plan (edge gateway / cloud REST API)**
8. **Limitations & Future Work**

## Future Enhancements

- Ingest real PLC/SCADA or MQTT stream data
- Add anomaly detection module (autoencoder)
- Add explainability (SHAP for tabular snapshot features)
- Add dashboard (Streamlit/Grafana) for maintenance teams
- Add MLOps pipeline (model versioning + scheduled retraining)

---

This project is intentionally built as a strong capstone baseline so you can demonstrate data engineering, deep learning, and production-oriented ML workflow in one deliverable.
