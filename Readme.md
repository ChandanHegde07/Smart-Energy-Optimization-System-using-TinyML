Smart Energy Optimization System
=================================

This is a TinyML-based occupancy detection system that controls classroom
lights and fans. It uses a small feedforward neural network trained on PIR
motion, ambient light, and temperature sensor data. It works. That's the
point.

If you're here looking for a 47-slide pitch deck explaining "the vision",
go away. This is a README, not a TED talk.


What it does
------------

Reads three sensor inputs. Runs them through a neural net. Turns lights and
fans on or off depending on whether someone's actually in the room. Tracks
how much energy you saved. Done.

The model hits 98% accuracy on the test set. Before you get excited — that
number comes from a controlled dataset. Real-world performance depends on
how badly you've set up your sensors.


The model
---------

MLP. Nothing exotic. 8 input features, two hidden layers (32 and 16
neurons), ReLU activations, sigmoid output, dropout to stop it from
memorizing noise. Adam optimizer, binary crossentropy loss.

Input features:
  - Temperature (raw)
  - Light level (raw)
  - PIR motion (raw)
  - 3-sample moving average of light
  - 3-sample moving average of temperature
  - Light delta over last 3 samples
  - Temperature delta over last 3 samples
  - Hour of day encoded as sin/cos (so midnight and 11pm aren't 23 units apart)

The temporal features matter. If you strip them out and feed raw sensor
readings directly, accuracy drops. Don't do that.

Model gets exported to TFLite int8 quantized format for edge deployment.
Inference is under 2ms per sample on a microcontroller. Compare that to
KNN at 50-100ms. KNN is not going on your ESP32. The FNN is.


Files that matter
-----------------

  app.py                    Flask server, ML inference endpoints
  src/model.py              Training pipeline, run this to retrain
  src/compare_ml_dl.py      Benchmarks FNN vs XGBoost, GBM, KNN
  occupancy_fnn_model.h5    Trained model, use this directly
  scaler.pkl                StandardScaler fitted on training data
  Sensor_Data_Engineered.csv  The training dataset
  Sensor.cpp                ESP32 firmware, for reference

Everything under templates/ and static/ is the web dashboard. Three.js
handles the 3D classroom visualization. Chart.js handles the graphs. Flask
serves it all.


How to run it
-------------

You need Python 3.8 or newer. You need pip. You presumably know how to use
a terminal. If you don't, this project is not for you yet.

  git clone https://github.com/ChandanHegde07/Smart-Energy-Optimization-System-using-TinyML.git
  cd SMART-Optimization
  pip install -r requirements.txt
  python app.py

Server starts on port 5001. Open http://localhost:5001 in a browser.

  /              Landing page
  /dashboard     The actual interface
  /classroom_simulation  3D visualization if you want that

To retrain from scratch:

  python src/model.py

This overwrites occupancy_fnn_model.h5, occupancy_fnn_int8.tflite, and
scaler.pkl. Don't do this unless you have a reason to.

To run the model comparison:

  python src/compare_ml_dl.py

This will tell you what you already know: the FNN is faster and smaller
than the alternatives, and that matters when your target hardware has 256KB
of RAM.


Performance numbers
-------------------

  Accuracy   ~98%
  Precision  ~0.97
  Recall     ~0.96
  F1         ~0.97
  ROC-AUC    ~0.99

Inference speed (per sample):

  FNN               0.5–2ms     runs on edge hardware
  XGBoost           5–10ms      doesn't
  Gradient Boosting 10–20ms     doesn't
  KNN               50–100ms    absolutely not


What the dashboard does
-----------------------

Manual mode: you set PIR, light level, and temperature. The model
predicts occupancy. The UI shows you what lights and fans would do.

Auto mode: cycles through demonstration scenarios automatically.

Either way it tracks cumulative energy saved (kWh), CO2 reduced (kg),
and cost savings (rupees). The calculations are estimates based on typical
classroom power draw. Don't use them for an energy audit.

The 3D simulation is a Three.js scene with a desk and student model. The
lighting in the scene changes based on model predictions. It's a
demonstration tool. It works fine.


What this is NOT
----------------

This is not production-ready building management software. It's a
research prototype demonstrating that a quantized MLP is good enough for
occupancy detection and small enough to deploy to constrained hardware.
The ESP32 firmware is included for reference — wiring your actual
classroom requires you to read the firmware and figure out your own
sensor placement.

Don't email asking for support. Open a GitHub issue. Include logs.


License
-------

MIT. See LICENSE.

There's also a patent notice — the system is covered under an Indian
Patent Office filing by Sai Vidya Institute of Technology. Read the
LICENSE file before you fork this and try to commercialize it.
