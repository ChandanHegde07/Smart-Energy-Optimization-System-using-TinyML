SMART Optimization
![Last Updated](https://img.shields.io/badge/Last_Updated-March_2026-eeb901?style=flat)
==================

SMART Optimization is a TinyML occupancy detection system for classrooms.
It reads PIR motion, light, and temperature signals, predicts occupancy with a
compact neural network, and drives energy-aware light/fan control logic.


Quick Start
-----------

* Run the app: `python app.py`
* Retrain the model: `python src/model.py`
* Compare baselines: `python src/compare_ml_dl.py`
* Dashboard: `http://localhost:5001/dashboard`


Essential Project Files
-----------------------

All users should know these files:

* Main app: `app.py`
* Training pipeline: `src/model.py`
* Model comparison: `src/compare_ml_dl.py`
* Trained model: `occupancy_fnn_model.h5`
* Quantized model: `occupancy_fnn_int8.tflite`
* Feature scaler: `scaler.pkl`
* Training dataset: `Sensor_Data_Engineered.csv`
* ESP32 reference firmware: `Sensor.cpp`
* Frontend assets: `templates/`, `static/`


System Summary
--------------

* Model type: Feedforward Neural Network (MLP)
* Input size: 8 engineered features
* Hidden layers: 32, 16 (ReLU)
* Output: 1 (sigmoid)
* Export: TFLite int8 quantized
* Typical inference: ~0.5-2 ms per sample

Reference metrics (controlled test set):

* Accuracy: ~98%
* Precision: ~0.97
* Recall: ~0.96
* F1: ~0.97
* ROC-AUC: ~0.99


Who Are You?
============

Find your role below:

* New Contributor - Running and understanding the project
* ML Engineer - Training, evaluation, and feature engineering
* Edge/Embedded Developer - Firmware and deployment constraints
* Frontend Contributor - Dashboard and simulation UI changes
* Reviewer/Maintainer - Validation, quality, and release decisions


For Specific Users
==================

New Contributor
---------------

Start here:

* Install dependencies: `pip install -r requirements.txt`
* Run server: `python app.py`
* Open dashboard: `http://localhost:5001/dashboard`
* Explore templates/UI: `templates/`, `static/`

ML Engineer
-----------

Model workflow:

* Retrain pipeline: `python src/model.py`
* Compare alternatives: `python src/compare_ml_dl.py`
* Dataset source: `Sensor_Data_Engineered.csv`
* Outputs to verify: `occupancy_fnn_model.h5`, `occupancy_fnn_int8.tflite`, `scaler.pkl`

Edge/Embedded Developer
-----------------------

Deployment focus:

* Firmware reference: `Sensor.cpp`
* Use quantized model: `occupancy_fnn_int8.tflite`
* Check inference budget and memory limits on target MCU
* Validate with real sensor placement before production use

Frontend Contributor
--------------------

UI and visualization focus:

* Flask routes in `app.py`
* Dashboard templates in `templates/`
* Static assets/scripts in `static/`
* Validate manual mode, auto mode, and charts after UI edits

Reviewer/Maintainer
-------------------

Before merge/release:

* Confirm app boots cleanly (`python app.py`)
* Validate model artifacts exist and load correctly
* Re-check performance claims after retraining
* Ensure dashboard behavior matches occupancy predictions


Run Locally
-----------

```
git clone https://github.com/ChandanHegde07/Smart-Energy-Optimization-System-using-TinyML.git
cd SMART-Optimization
pip install -r requirements.txt
python app.py
```

Routes:

* `/` landing page
* `/dashboard` main dashboard
* `/classroom_simulation` 3D classroom simulation


License
-------

MIT. See `LICENSE`.

Patent notice applies (Indian Patent Office filing by Sai Vidya Institute of
Technology). Review `LICENSE` before commercial use.
