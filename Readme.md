# TinyML - Driven Occupancy and Appliance Control System for Classroom Energy Reduction

A real-time IoT dashboard for intelligent lighting control using TinyML, ESP32, and environmental sensors. The system employs machine learning to make smart lighting decisions based on motion detection, ambient light levels, and room temperature, enabling significant energy savings.

---

## Overview

This project demonstrates an energy-efficient lighting control system that uses TinyML (Tiny Machine Learning) for edge-based decision making. By processing sensor data directly on the ESP32 microcontroller, the system achieves real-time responses while minimizing energy consumption and maintaining user privacy.

---

## Key Features

- **Real-time Sensor Monitoring**: Live visualization of PIR motion, LDR light levels, and temperature data
- **TinyML Integration**: Intelligent decision-making using machine learning models with **98% accuracy**
- **Energy Analytics**: Track energy savings, CO₂ reduction, and cost savings
- **Interactive Charts**: Chart.js visualizations with historical data trends
- **Auto-refresh**: Configurable live data updates every 2 seconds
- **Simulation Mode**: Test without hardware using realistic sensor data generation

---

## Hardware Components

| Component | Model | Purpose |
|-----------|-------|---------|
| Microcontroller | ESP32 | Main processing unit with Wi-Fi capability |
| Temperature Sensor | DHT11/DHT22 | Monitors room temperature and humidity |
| Motion Sensor | PIR (HC-SR501) | Detects human presence |
| Light Sensor | LDR (Light Dependent Resistor) | Measures ambient brightness levels |

---

## Technology Stack

### Backend

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Flask | Web framework for API and dashboard |
| NumPy | Numerical computations |
| Pandas | Data manipulation and analysis |
| Scikit-learn | Machine learning model training |
| TensorFlow | Deep learning framework for FNN model |
| Pickle | Model serialization |

### Frontend

| Technology | Purpose |
|------------|---------|
| HTML5 | Structure and markup |
| CSS3 | Styling and layout |
| JavaScript | Interactive functionality |
| Chart.js | Data visualizations |

### Hardware Libraries

| Library | Purpose |
|---------|---------|
| ArduinoJson | JSON parsing for ESP32 |
| DHT Sensor Library | Temperature/humidity readings |

---

## Dashboard Components

### Sensor Monitoring
- **PIR Motion Sensor**: Detects human presence (Binary: 0/1)
- **LDR Light Sensor**: Measures ambient brightness (Range: 0-100)
- **Temperature Sensor**: Monitors room temperature (°C)

### Analytics & Statistics
- Cumulative energy saved (kWh)
- CO₂ emissions reduced (kg)
- Cost savings (USD)
- Number of unnecessary light activations prevented
- Temperature statistics (min/max/average)

### Visualizations
- Energy savings trend chart
- Temperature monitoring graph
- AI decision history
- Real-time status indicators

---

## Deep Learning Model

The system uses a Feedforward Neural Network (FNN) trained on historical sensor data to predict optimal lighting states.

### Model Performance

| Metric | Value |
|--------|-------|
| Model Type | Feedforward Neural Network (FNN) |
| Accuracy | **98%** |
| Input Features | Motion (PIR), Light Level (LDR), Temperature |
| Output | Light Control Decision (ON/OFF) |

The TinyML model runs directly on the ESP32, enabling low-latency inference and reducing dependence on cloud connectivity.

---

## Usage

### Hardware Mode
Connect your ESP32 with sensors to the Flask server endpoint. The dashboard will display live sensor data and ML-driven lighting decisions.

### Simulation Mode
Run the system without physical hardware using built-in sensor simulation for testing and demonstration purposes.

---

# Patent Information

## Patent Status
Published (Indian Patent Office)

## Project Details
This project includes proprietary technology and methodologies disclosed in a patent published by the Indian Patent Office (IPO) in the name of **Sai Vidya Institute of Technology (SVIT)**.

**Technology**: TinyML - Driven Occupancy and Appliance Control System for Classroom Energy Reduction

## Intellectual Property Rights
All intellectual property rights related to the TinyML - Driven Occupancy and Appliance Control System for Classroom Energy Reduction are protected under applicable patent and intellectual property laws.

## Important Notice
**Unauthorized use, reproduction, or distribution of the patented/patent-published technology may constitute infringement and may be subject to legal action.**

## Licensing and Commercial Use
For licensing inquiries or commercial use, please contact:
* Project maintainers
* Sai Vidya Institute of Technology (SVIT)


