# Smart Energy Optimization System using TinyML

> **TinyML-Driven Occupancy and Appliance Control System for Classroom Energy Reduction**
> *Patent Published — Indian Patent Office (IPO) | Assignee: Sai Vidya Institute of Technology (SVIT)*

---

## Abstract

This project presents an intelligent IoT-based lighting control system that leverages TinyML for edge-based decision-making. By processing environmental data locally on an ESP32 microcontroller, the system maximizes energy efficiency while ensuring real-time response and data privacy. Unlike traditional threshold-based automation, this system employs a Feedforward Neural Network (FNN) to analyze motion, ambient light, and temperature, predicting the most energy-efficient lighting state with **98% classification accuracy**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Circuit & Wiring Diagram](#circuit--wiring-diagram)
- [Deep Learning Model](#deep-learning-model)
- [Inference Benchmarks](#inference-benchmarks)
- [Project Structure](#project-structure)
- [Results & Performance](#results--performance)
- [Installation & Setup](#installation--setup)
- [Patent & Intellectual Property](#patent--intellectual-property)

---

## Overview

This project bridges the gap between the Internet of Things (IoT) and on-device Machine Learning. The system collects real-time sensor data — motion (PIR), ambient brightness (LDR), and ambient temperature (DHT11/DHT22) — and feeds it into a quantized neural network running entirely on the ESP32. Inference results drive relay-controlled lighting, with all analytics and visualizations surfaced through a Flask-powered web dashboard.

The core research contribution is demonstrating that a resource-constrained microcontroller (240 MHz dual-core, 520 KB SRAM) can sustain continuous ML inference at production accuracy without offloading computation to the cloud, eliminating round-trip latency and third-party data exposure.

---

## Key Features

- **Edge Intelligence**: All inference executes on the ESP32 — no cloud dependency, no network latency, no privacy exposure.
- **Predictive Analytics**: Tracks cumulative energy saved (kWh), estimated CO₂ reduction (kg), and cost savings (USD) over time.
- **Live Dashboard**: Interactive Chart.js visualizations with a 2-second auto-refresh rate for real-time monitoring.
- **Hybrid Operation**: Seamlessly switch between physical hardware mode and a software simulation mode for development and testing.
- **High Accuracy**: The ML model is specifically trained to suppress false triggers caused by ambient light fluctuations, a common failure mode in LDR-based systems.

---

## Technology Stack

### Hardware & Firmware

| Component | Model | Purpose |
|-----------|-------|---------|
| Microcontroller | ESP32 (Xtensa LX6 dual-core, 240 MHz) | Edge inference, Wi-Fi connectivity |
| Temperature/Humidity | DHT11 / DHT22 | Thermal environment monitoring |
| Motion Sensor | PIR (HC-SR501) | Human presence detection |
| Ambient Light | LDR (GL5528 or equivalent) | Brightness measurement (ADC) |
| Relay Module | 5V Single-Channel | Actuates lighting load |

### Software & ML

| Layer | Technology |
|-------|------------|
| Backend | Python 3.x, Flask, NumPy, Pandas |
| Machine Learning | TensorFlow / TFLite for Microcontrollers, Scikit-learn |
| Frontend | HTML5, CSS3, JavaScript (Chart.js) |
| Firmware | C++, Arduino Framework (PlatformIO), ArduinoJson |
| Serial Logging | Python (PySerial) → CSV |

---

## Circuit & Wiring Diagram

### Pin Mapping (ESP32 DevKit v1)

| Sensor / Module | ESP32 Pin | Notes |
|----------------|-----------|-------|
| DHT11/DHT22 DATA | GPIO 4 | 10kΩ pull-up to 3.3V |
| PIR (HC-SR501) OUT | GPIO 5 | Digital HIGH = motion detected |
| LDR (Voltage Divider) | GPIO 34 (ADC1_CH6) | Use ADC1 pins only; ADC2 conflicts with Wi-Fi |
| Relay IN | GPIO 26 | Active LOW logic; drive via NPN transistor if needed |
| DHT / PIR VCC | 3.3V | HC-SR501 can accept 5V; check sensor datasheet |
| All GND | GND | Common ground |

### LDR Voltage Divider Circuit

```
3.3V
  |
 [LDR]  ← resistance varies with light
  |
  +──── GPIO 34 (ADC Input)
  |
[10kΩ] ← fixed resistor
  |
 GND
```

The ADC reads a voltage proportional to ambient brightness. Lower resistance (bright light) → higher voltage at GPIO 34. Calibrate the ADC range in firmware to map raw values (0–4095) to a normalized lux estimate.

### Relay Wiring (Lighting Load)

```
ESP32 GPIO 26 ──► Relay IN
                  Relay COM ──► Live Wire (AC)
                  Relay NO  ──► Load (Light Fixture)
                  Relay NC  ──► (unused)
```

> ⚠️ **Safety Notice**: Relay switching of mains AC voltage (120V/240V) carries risk of electric shock or fire. Ensure all AC connections are enclosed in a suitable housing and handled only by qualified personnel. Use an optically-isolated relay module for additional protection.

### Schematic Overview (Text Representation)

```
┌─────────────────────────────────────────────────┐
│                   ESP32 DevKit                  │
│                                                 │
│  GPIO 4  ◄──── DHT22 DATA (+ 10kΩ to 3.3V)    │
│  GPIO 5  ◄──── PIR HC-SR501 OUT                │
│  GPIO 34 ◄──── LDR Divider Mid-point           │
│  GPIO 26 ────► Relay Module IN                 │
│  3.3V    ────► DHT22 VCC, LDR Divider Rail     │
│  5V      ────► PIR VCC, Relay VCC              │
│  GND     ────► All Module GNDs                 │
└─────────────────────────────────────────────────┘
```

> 📌 A full KiCad/Fritzing schematic is available in the `hardware/` directory (if included in this repository).

---

## Deep Learning Model

The core decision logic is driven by a Feedforward Neural Network (FNN/MLP) optimized for resource-constrained microcontroller deployment.

### Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | Multi-Layer Perceptron (MLP) |
| Input Features | `[Motion (binary), Light (normalized), Temperature (normalized)]` |
| Hidden Layers | 2 fully-connected layers with ReLU activation |
| Output | Binary classification — Light ON / Light OFF |
| Training Accuracy | 98% |
| Deployment Format | TensorFlow Lite (`.tflite`), post-training quantization |
| Quantization | INT8 (weights + activations) for minimal SRAM footprint |

### Model Pipeline

```
Raw Sensor Data
       │
       ▼
  Preprocessing
  (Normalization, Feature Scaling)
       │
       ▼
  FNN Inference (ESP32 / TFLite Micro)
  ┌──────────────────────────────────┐
  │  Input Layer  [3 neurons]        │
  │  Hidden Layer [16 neurons, ReLU] │
  │  Hidden Layer [8 neurons, ReLU]  │
  │  Output Layer [1 neuron, Sigmoid]│
  └──────────────────────────────────┘
       │
       ▼
  Binary Decision → Relay Control
```

### Training Dataset

- **Collection Method**: Logged via `Logging.py` from a deployed ESP32 over multiple days across varied occupancy and lighting conditions.
- **Features**: Motion (PIR binary), Ambient Light (LDR ADC normalized), Temperature (°C normalized).
- **Labels**: Ground-truth energy-optimal lighting state (manually annotated or rule-bootstrapped).
- **Preprocessing**: Standard scaling (zero mean, unit variance) applied prior to training and serialized for on-device inference.

---

## Inference Benchmarks

Performance measurements obtained on an **ESP32 DevKit v1 (240 MHz, single-core inference task)**.

| Metric | Value |
|--------|-------|
| Model Size (FP32) | ~12 KB |
| Model Size (INT8 Quantized) | ~4 KB |
| Inference Latency (INT8) | ~2–5 ms per sample |
| SRAM Usage (TFLite Micro runtime) | ~20–30 KB |
| Flash Usage | ~1.5 MB (firmware + model) |
| Inference Throughput | ~200–500 inferences/sec (theoretical) |
| End-to-End Latency (sensor read → relay actuation) | < 50 ms |
| Power Consumption (active inference) | ~160 mA @ 3.3V (~528 mW) |
| Power Consumption (light sleep between samples) | ~10 mA @ 3.3V |

> **Note**: Latency measurements are approximate and depend on clock frequency, active peripherals (Wi-Fi enabled adds ~30–80 mA), and Arduino loop overhead. Profiling was performed using `esp_timer_get_time()` bracketing the `interpreter->Invoke()` call.

### Edge vs. Cloud Latency Comparison

| Deployment Mode | Round-Trip Latency | Privacy | Offline Capable |
|----------------|-------------------|---------|-----------------|
| **Edge (ESP32)** | **< 50 ms** | ✅ Full | ✅ Yes |
| Cloud API (Wi-Fi) | 200–800 ms | ❌ Data leaves device | ❌ No |
| Cloud API (4G/LTE) | 400–1500 ms | ❌ Data leaves device | ❌ No |

Edge deployment achieves a **4–30× latency reduction** over cloud-based inference, with the additional benefit of fully offline operation — critical for energy management in environments with unreliable network connectivity.

---

## Project Structure

```
Smart-Energy-Optimization/
├── app.py                # Flask web server & dashboard backend
├── Logging.py            # Serial data logger (ESP32 → CSV via PySerial)
├── Sensor.cpp            # ESP32 Firmware (C++, Arduino Framework)
├── requirements.txt      # Python dependencies
├── model/
│   ├── train_model.py    # FNN training script (TensorFlow/Keras)
│   ├── model.tflite      # Quantized TFLite model for deployment
│   └── scaler.pkl        # Serialized StandardScaler for inference
├── hardware/
│   └── schematic.fzz     # Fritzing wiring diagram (if available)
├── results/
│   ├── activation_functions.png
│   ├── complete_architecture_visualization.png
│   ├── model_architecture.png
│   └── training_results.png
├── data/
│   └── sensor_log.csv    # Collected training/evaluation dataset
└── README.md
```

---

## Results & Performance

The `results/` directory contains comprehensive model evaluation artifacts:

| File | Description |
|------|-------------|
| `training_results.png` | Accuracy and loss convergence curves (train vs. validation) |
| `model_architecture.png` | Visual representation of MLP layer structure |
| `activation_functions.png` | Empirical comparison of ReLU vs. Sigmoid on this classification task |
| `complete_architecture_visualization.png` | End-to-end system diagram from sensor input to relay output |

### Summary Metrics

| Metric | Value |
|--------|-------|
| Classification Accuracy | 98% |
| False Positive Rate (unnecessary light-ON) | < 2% |
| False Negative Rate (missed occupancy) | < 2% |
| Estimated Energy Savings vs. Always-ON | Up to 40–60% (environment dependent) |

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- Node.js (optional, for frontend tooling)
- VS Code with PlatformIO extension **or** Arduino IDE 2.x
- ESP32 board support package installed

### Steps

**1. Clone the Repository**
```bash
git clone https://github.com/your-username/Smart-Energy-Optimization.git
cd Smart-Energy-Optimization
```

**2. Install Python Dependencies**
```bash
pip install -r requirements.txt
```

**3. (Optional) Retrain the Model**
```bash
python model/train_model.py
# Outputs: model/model.tflite, model/scaler.pkl
```

**4. Flash Firmware to ESP32**

Open `Sensor.cpp` in PlatformIO (recommended) or Arduino IDE. Ensure the following libraries are installed:
- `ArduinoJson`
- `DHT sensor library` (Adafruit)
- `TensorFlowLite_ESP32`

Then upload to your ESP32.

**5. Launch the Dashboard**
```bash
python app.py
```
Navigate to `http://localhost:5000` in your browser.

**6. (Simulation Mode)**

If no hardware is connected, the system defaults to software simulation mode, generating synthetic sensor readings for demonstration purposes.

---

## Patent & Intellectual Property

> [!IMPORTANT]
> This technology is Patent Published through the Indian Patent Office (IPO).
>
> **Patent Title**: TinyML-Driven Occupancy and Appliance Control System for Classroom Energy Reduction
>
> **Assignee**: Sai Vidya Institute of Technology (SVIT)
>
> **Notice**: Unauthorized reproduction or commercial distribution of this methodology is strictly prohibited and may result in legal action.

For licensing inquiries or academic collaboration, please contact the **SVIT IP Cell**.


## License

This project is protected under patent law. All rights reserved by Sai Vidya Institute of Technology (SVIT). See [LICENSE](LICENSE) for academic use terms.
