# SMART Optimization

A TinyML occupancy detection system for classrooms. SMART Optimization reads PIR motion, light, and temperature signals, predicts occupancy with a compact neural network, and drives energy-aware light/fan control logic.

---

## Table of Contents

- [Quick Start](#quick-start)
- [System Summary](#system-summary)
- [Reference Metrics](#reference-metrics)
- [Essential Project Files](#essential-project-files)
- [Who Are You?](#who-are-you)
  - [New Contributor](#new-contributor)
  - [ML Engineer](#ml-engineer)
  - [Edge/Embedded Developer](#edgeembedded-developer)
  - [Frontend Contributor](#frontend-contributor)
  - [Reviewer/Maintainer](#reviewermaintainer)
- [Run Locally](#run-locally)
- [Routes](#routes)
- [License](#license)

---

## Quick Start

| Task | Command |
|---|---|
| Run the app | `python app.py` |
| Retrain the model | `python src/model.py` |
| Compare baselines | `python src/compare_ml_dl.py` |
| Dashboard | `http://localhost:5001/dashboard` |

---

## System Summary

| Property | Value |
|---|---|
| Model type | Feedforward Neural Network (MLP) |
| Input size | 8 engineered features |
| Hidden layers | 32, 16 (ReLU) |
| Output | 1 (sigmoid) |
| Export | TFLite INT8 quantized |
| Typical inference | ~0.5–2 ms per sample |

## Reference Metrics

*Measured on a controlled test set.*

| Metric | Value |
|---|---|
| Accuracy | ~98% |
| Precision | ~0.97 |
| Recall | ~0.96 |
| F1 | ~0.97 |
| ROC-AUC | ~0.99 |

---

## Essential Project Files

All contributors should be familiar with these files:

| File | Purpose |
|---|---|
| `app.py` | Main application (Flask server, routes) |
| `src/model.py` | Training pipeline |
| `src/compare_ml_dl.py` | Model comparison across ML/DL baselines |
| `occupancy_fnn_model.h5` | Trained model (Keras) |
| `occupancy_fnn_int8.tflite` | Quantized model for edge deployment |
| `scaler.pkl` | Feature scaler used at inference time |
| `Sensor_Data_Engineered.csv` | Training dataset |
| `Sensor.cpp` | ESP32 reference firmware |
| `templates/`, `static/` | Frontend assets |

---

## Who Are You?

Jump to the section that matches what you're trying to do:

- **[New Contributor](#new-contributor)** — running and understanding the project
- **[ML Engineer](#ml-engineer)** — training, evaluation, and feature engineering
- **[Edge/Embedded Developer](#edgeembedded-developer)** — firmware and deployment constraints
- **[Frontend Contributor](#frontend-contributor)** — dashboard and simulation UI changes
- **[Reviewer/Maintainer](#reviewermaintainer)** — validation, quality, and release decisions

### New Contributor

Start here:

1. Install dependencies: `pip install -r requirements.txt`
2. Run the server: `python app.py`
3. Open the dashboard: `http://localhost:5001/dashboard`
4. Explore the UI: `templates/`, `static/`

### ML Engineer

Model workflow:

- Retrain pipeline: `python src/model.py`
- Compare alternatives: `python src/compare_ml_dl.py`
- Dataset source: `Sensor_Data_Engineered.csv`
- Outputs to verify: `occupancy_fnn_model.h5`, `occupancy_fnn_int8.tflite`, `scaler.pkl`

### Edge/Embedded Developer

Deployment focus:

- Firmware reference: `Sensor.cpp`
- Use the quantized model: `occupancy_fnn_int8.tflite`
- Check inference budget and memory limits on the target MCU
- Validate with real sensor placement before production use

### Frontend Contributor

UI and visualization focus:

- Flask routes in `app.py`
- Dashboard templates in `templates/`
- Static assets/scripts in `static/`
- Validate manual mode, auto mode, and charts after any UI edits

### Reviewer/Maintainer

Before merge/release:

- [ ] Confirm the app boots cleanly (`python app.py`)
- [ ] Validate model artifacts exist and load correctly
- [ ] Re-check performance claims after retraining
- [ ] Ensure dashboard behavior matches occupancy predictions

---

## Run Locally

```bash
git clone https://github.com/ChandanHegde07/Smart-Energy-Optimization-System-using-TinyML.git
cd Smart-Energy-Optimization-System-using-TinyML
pip install -r requirements.txt
python app.py
```

## Routes

| Route | Description |
|---|---|
| `/` | Landing page |
| `/dashboard` | Main dashboard |
| `/classroom_simulation` | 3D classroom simulation |

## License

MIT. See [`LICENSE`](./LICENSE).

> **Patent notice applies** (Indian Patent Office filing by Sai Vidya Institute of Technology). Review `LICENSE` before commercial use.
