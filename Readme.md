# Smart Energy Optimization System using TinyML

An intelligent IoT-based lighting control system that leverages TinyML for edge-based decision-making. By processing environmental data locally on an ESP32, the system maximizes energy efficiency while ensuring real-time response and data privacy.

## Overview

This project bridges the gap between IoT and Machine Learning. Unlike traditional "if-then" automation, this system utilizes a Feedforward Neural Network (FNN) to analyze motion, ambient light, and temperature, predicting the most energy-efficient lighting state with 98% accuracy.

## Key Features

- **Edge Intelligence**: Inference happens directly on the ESP32—no cloud latency or privacy concerns.
- **Predictive Analytics**: Tracks cumulative energy saved (kWh), CO₂ reduction, and cost savings (USD).
- **Live Dashboard**: Interactive visualizations powered by Chart.js with a 2-second auto-refresh rate.
- **Hybrid Operation**: Seamlessly switch between physical hardware mode and a software simulation mode.
- **High Accuracy**: Robust ML model trained to prevent "false triggers" from ambient light fluctuations.

## Technology Stack

### Hardware & Firmware

| Component | Model | Purpose |
|-----------|-------|---------|
| Microcontroller | ESP32 | Dual-core processing & Wi-Fi connectivity |
| Temp/Humidity | DHT11/DHT22 | Monitors thermal environment |
| Motion Sensor | PIR (HC-SR501) | Human presence detection |
| Light Sensor | LDR | Ambient brightness measurement |

### Software & ML

- **Backend**: Python, Flask, NumPy, Pandas
- **Machine Learning**: TensorFlow (Lite for Microcontrollers), Scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript (Chart.js)
- **Firmware**: C++, Arduino Framework, ArduinoJson

## Deep Learning Model

The core logic is driven by a Feedforward Neural Network (FNN) optimized for resource-constrained hardware.

| Metric | Value |
|--------|-------|
| Architecture | Multi-layer Perceptron (MLP) |
| Input Features | [Motion, Light, Temperature] |
| Accuracy | 98% |
| Deployment | Serialized and quantized for TinyML edge execution |

## Project Structure

```
Smart-Energy-Optimization/
├── Logging.py           # Serial data logger (ESP32 → CSV)
├── Sensor.cpp           # ESP32 Firmware (C++)
├── requirements.txt      # Python dependencies
├── results/              # ML Visualizations
│   ├── activation_functions.png
│   ├── complete_architecture_visualization.png
│   ├── model_architecture.png
│   └── training_results.png
└── README.md
```

## Results & Performance

The `results/` directory contains comprehensive performance evaluations:

- **Training Results**: Accuracy and Loss curves demonstrating model convergence.
- **Activation Functions**: A comparison of ReLU vs Sigmoid performance on the edge.
- **Edge vs Cloud**: Data showing the latency benefits of FNN edge deployment.

## Patent & Intellectual Property

> [!IMPORTANT]
> This technology is Patent Published through the Indian Patent Office (IPO).
>
> **Patent Title**: TinyML - Driven Occupancy and Appliance Control System for Classroom Energy Reduction
>
> **Assignee**: Sai Vidya Institute of Technology (SVIT)
>
> **Notice**: Unauthorized reproduction or commercial distribution of this methodology is strictly prohibited and may result in legal action.

For licensing or academic collaboration, please contact the SVIT IP Cell.

## Installation & Setup

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/your-username/Smart-Energy-Optimization.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Flash Hardware**:
   Upload `Sensor.cpp` to your ESP32 using VS Code (PlatformIO) or Arduino IDE.

4. **Run Dashboard**:
   ```bash
   python app.py
   ```
