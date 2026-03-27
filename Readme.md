Smart Energy Optimization System
================================

TinyML occupancy detection for classrooms.

Reads PIR motion, light, and temperature data. Runs a compact neural net.
Controls lights/fans based on occupancy prediction. Tracks estimated energy,
CO2, and cost savings.


What this is
------------

A research prototype for edge-friendly occupancy detection.

- Model: feedforward neural net (MLP)
- Deployment target: constrained hardware (ESP32-class)
- Export: int8 quantized TFLite
- Goal: practical inference speed with high classification quality

Not production building-management software.


Model summary
-------------

Architecture:

- Input: 8 engineered features
- Hidden layers: 32, 16 (ReLU)
- Output: 1 (sigmoid)
- Regularization: dropout
- Training: Adam + binary crossentropy

Features:

- Temperature (raw)
- Light level (raw)
- PIR motion (raw)
- 3-sample moving average (light)
- 3-sample moving average (temperature)
- Light delta over 3 samples
- Temperature delta over 3 samples
- Hour of day encoded with sin/cos

Temporal features are required for current performance levels.


Performance
-----------

Test-set metrics (controlled dataset):

- Accuracy: ~98%
- Precision: ~0.97
- Recall: ~0.96
- F1: ~0.97
- ROC-AUC: ~0.99

Approx. inference latency (per sample):

- FNN: 0.5-2 ms (edge-capable)
- XGBoost: 5-10 ms
- Gradient Boosting: 10-20 ms
- KNN: 50-100 ms

Real-world performance depends on sensor quality and placement.


Repository layout
-----------------

- app.py: Flask app and inference endpoints
- src/model.py: training/retraining pipeline
- src/compare_ml_dl.py: baseline comparison scripts
- occupancy_fnn_model.h5: trained model
- occupancy_fnn_int8.tflite: quantized deployment model
- scaler.pkl: fitted feature scaler
- Sensor_Data_Engineered.csv: engineered training dataset
- Sensor.cpp: reference ESP32 firmware
- templates/, static/: dashboard frontend assets


Run
---

Requirements:

- Python 3.8+
- pip

Commands:

```
git clone https://github.com/ChandanHegde07/Smart-Energy-Optimization-System-using-TinyML.git
cd SMART-Optimization
pip install -r requirements.txt
python app.py
```

Server default: `http://localhost:5001`

Routes:

- `/` landing page
- `/dashboard` main interface
- `/classroom_simulation` 3D visualization


Retrain model
-------------

```
python src/model.py
```

This regenerates:

- occupancy_fnn_model.h5
- occupancy_fnn_int8.tflite
- scaler.pkl


Compare models
--------------

```
python src/compare_ml_dl.py
```

Used to benchmark FNN vs classical alternatives for latency and practicality
on constrained hardware.


Dashboard behavior
------------------

- Manual mode: user sets PIR/light/temperature and gets model prediction
- Auto mode: cycles through predefined demo scenarios
- Outputs: estimated kWh saved, CO2 reduction, and cost savings

3D scene is a demonstration layer built with Three.js. Charts use Chart.js.


License
-------

MIT. See `LICENSE`.

Patent notice applies (Indian Patent Office filing by Sai Vidya Institute of
Technology). Review `LICENSE` before commercial use.
