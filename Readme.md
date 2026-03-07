# Smart Energy Optimization System using TinyML

> **AI-Driven Occupancy Detection & Energy Management Dashboard**
> *An Intelligent System for Classroom Energy Reduction using Feedforward Neural Networks*

---

## Abstract

This project presents an intelligent energy management system that leverages machine learning for occupancy-based lighting and fan control. Built with a Flask-powered web dashboard and a Feedforward Neural Network (FNN) model, the system analyzes environmental data — motion (PIR), ambient light (LDR), and temperature — to predict optimal appliance states with **98% classification accuracy**.

The system features an interactive 3D classroom simulation using Three.js, real-time analytics for energy savings, CO₂ reduction, and cost tracking, making it ideal for demonstrating TinyML concepts in educational environments.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Deep Learning Model](#deep-learning-model)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Results & Performance](#results--performance)
- [License](#license)

---

## Overview

This project bridges the gap between IoT and machine learning by creating a complete simulation environment for occupancy-based energy optimization. The system:

1. **Collects** environmental sensor data (motion, light, temperature)
2. **Processes** data through feature engineering with temporal features
3. **Predicts** occupancy using an FNN model
4. **Controls** lighting and fans based on occupancy and environmental conditions
5. **Visualizes** results through an interactive dashboard and 3D simulation

The research contribution demonstrates that a lightweight FNN can achieve high accuracy for occupancy detection while remaining deployable to resource-constrained edge devices.

---

## Key Features

- **Interactive Dashboard**: Real-time monitoring with Chart.js visualizations showing energy saved, CO₂ reduced, and cost savings
- **3D Classroom Simulation**: Immersive Three.js-powered simulation with 3D models of a desk and student
- **Dual Mode Operation**: Manual input mode for testing scenarios and auto mode for continuous demonstration
- **FNN Model**: Feedforward Neural Network with feature engineering (temporal features like light_mean_3, temp_mean_3)
- **Smart Controls**: Automatic fan activation based on temperature thresholds when room is occupied
- **Energy Analytics**: Tracks cumulative energy savings, CO₂ reduction, and estimated cost savings over time
- **Model Comparison**: Includes comparison scripts to benchmark FNN against traditional ML algorithms (XGBoost, Gradient Boosting, KNN)

---

## Technology Stack

### Backend & ML

| Component | Technology |
|-----------|------------|
| Framework | Flask 3.0.3 |
| ML Library | TensorFlow 2.17.0 |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Model Format | Keras H5, TensorFlow Lite |

### Frontend & UI

| Component | Technology |
|-----------|------------|
| HTML/CSS | Modern responsive design |
| Charts | Chart.js 4.4.0 |
| 3D Graphics | Three.js with GLTF models |
| Icons | Font Awesome 6.4.0 |

### Project Files

| File | Purpose |
|------|---------|
| `app.py` | Flask web server with ML inference endpoints |
| `src/model.py` | FNN model training script |
| `src/compare_ml_dl.py` | Model comparison with ML algorithms |
| `src/evaluate_with_labels.py` | Model evaluation with labeled data |
| `templates/*.html` | Frontend templates |
| `static/` | CSS, JS, and 3D models |

---

## Project Structure

```
SMART-Optimization/
├── app.py                        # Flask application & ML inference
├── Logging.py                    # Serial data logger (legacy)
├── Sensor.cpp                    # ESP32 firmware (for reference)
├── requirements.txt              # Python dependencies
├── occupancy_fnn_model.h5        # Trained FNN model
├── scaler.pkl                    # Feature scaler
├── Sensor_Data_Engineered.csv    # Training dataset
│
├── src/                          # ML source code
│   ├── model.py                  # FNN training pipeline
│   ├── compare_ml_dl.py          # FNN vs ML comparison
│   ├── evaluate_with_labels.py  # Model evaluation
│   ├── model_architecture.py    # Architecture visualization
│   ├── model_compare.py         # Model comparison utilities
│   ├── test.py                   # Testing utilities
│   └── test_data.py              # Test data generation
│
├── templates/                    # HTML templates
│   ├── index.html                # Main dashboard
│   ├── home.html                 # Landing page
│   └── classroom_simulation.html # 3D simulation
│
├── static/                       # Static assets
│   ├── css/style.css             # Dashboard styles
│   ├── js/script.js             # Frontend logic
│   └── models/                   # 3D GLTF models
│       ├── desk/                 # 3D desk model
│       └── student/              # 3D student model
│
├── results/                      # Model visualizations
│   ├── training_results.png
│   ├── model_architecture.png
│   ├── activation_functions.png
│   └── fnn_vs_ml_edge_deployment.png
│
├── reports/                      # Evaluation reports
│   ├── 01_probability_hist.png
│   ├── 02_prediction_pie.png
│   └── 03_probability_timeline.png
│
├── reports_labeled/              # Labeled evaluation reports
│   ├── cm.png                    # Confusion matrix
│   ├── pr.png                    # Precision-Recall
│   ├── roc.png                   # ROC curve
│   ├── reliability.png
│   └── threshold_sweep.png
│
└── LICENSE                       # MIT License with patent notice
```

---

## Deep Learning Model

The core decision logic is driven by a Feedforward Neural Network (FNN/MLP) optimized for occupancy classification.

### Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | Multi-Layer Perceptron (MLP) |
| Input Features | 8 features (Temperature, Light, Light_mean_3, Light_diff_3, Temp_mean_3, Temp_diff_3, hour_sin, hour_cos) |
| Hidden Layer 1 | 32 neurons, ReLU activation |
| Dropout | 0.2 (to prevent overfitting) |
| Hidden Layer 2 | 16 neurons, ReLU activation |
| Output | 1 neuron, Sigmoid activation (binary classification) |
| Training Accuracy | ~98% |
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Crossentropy |

### Feature Engineering

The model uses advanced temporal features for better prediction:

- **Current values**: Temperature, Light, PIR
- **Moving averages**: 3-sample mean for light and temperature
- **Differences**: Change in light and temperature over last 3 samples
- **Time features**: Cyclical encoding of hour (sin/cos)

### Model Pipeline

```
Raw Sensor Data (PIR, LDR, Temperature)
        │
        ▼
   Feature Engineering
   - Moving averages (3 samples)
   - Temporal differences
   - Hour encoding (sin/cos)
        │
        ▼
   Feature Scaling (StandardScaler)
        │
        ▼
   FNN Inference
   ┌──────────────────────────────────┐
   │  Input Layer  [8 neurons]      │
   │  Hidden Layer [32 neurons, ReLU]│
   │  Dropout 0.2                    │
   │  Hidden Layer [16 neurons, ReLU]│
   │  Dropout 0.2                    │
   │  Output Layer [1 neuron, Sigmoid]│
   └──────────────────────────────────┘
        │
        ▼
   Binary Decision → Light/Fan Control
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Steps

**1. Clone the Repository**

```bash
git clone https://github.com/ChandanHegde07/Smart-Energy-Optimization-System-using-TinyML.git
cd SMART-Optimization
```

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. (Optional) Train the Model**

If you want to retrain the FNN model:

```bash
python src/model.py
```

This will generate:
- `occupancy_fnn_model.h5` - Trained Keras model
- `occupancy_fnn_int8.tflite` - Quantized TFLite model
- `scaler.pkl` - Feature scaler

**4. Run the Application**

```bash
python app.py
```

The server will start on `http://localhost:5001`

**5. Access the Dashboard**

- **Landing Page**: http://localhost:5001/
- **Main Dashboard**: http://localhost:5001/dashboard
- **3D Simulation**: http://localhost:5001/classroom_simulation

---

## Usage Guide

### Dashboard Mode

1. Open the dashboard at `/dashboard`
2. Toggle between **Auto Mode** (demonstration scenarios) and **Manual Mode** (custom input)
3. Adjust sensor values using the control panel:
   - **PIR Motion**: 0 (No Motion) or 1 (Motion Detected)
   - **LDR Light**: 0-1000 (brightness level)
   - **Temperature**: 0-50°C
4. View real-time predictions, energy savings, and charts
5. The system automatically calculates:
   - Energy saved (kWh)
   - CO₂ reduced (kg)
   - Cost savings (₹)

### 3D Classroom Simulation

1. Navigate to `/classroom_simulation`
2. Interact with the 3D scene featuring:
   - Classroom environment with desk and student models
   - Real-time sensor overlays
   - Dynamic lighting based on occupancy predictions
3. Monitor live sensor data and predictions

### Model Comparison

To compare FNN against traditional ML algorithms:

```bash
python src/compare_ml_dl.py
```

This generates comprehensive visualizations comparing:
- Accuracy, Precision, Recall, F1-Score
- ROC Curves
- Inference Speed
- Edge Deployment suitability

---

## Results & Performance

### Model Performance

| Metric | Value |
|--------|-------|
| Classification Accuracy | ~98% |
| Precision | ~0.97 |
| Recall | ~0.96 |
| F1-Score | ~0.97 |
| ROC-AUC | ~0.99 |

### Inference Speed

| Model | Speed (ms/sample) | Edge Deployable |
|-------|-------------------|-----------------|
| FNN | ~0.5-2ms | ✓ Yes |
| XGBoost | ~5-10ms | ✗ No |
| Gradient Boosting | ~10-20ms | ✗ No |
| K-Nearest Neighbors | ~50-100ms | ✗ No |

### Key Visualizations

The `results/` directory contains:

| File | Description |
|------|-------------|
| `training_results.png` | Training/validation accuracy and loss curves |
| `model_architecture.png` | Neural network architecture diagram |
| `activation_functions.png` | ReLU vs Sigmoid comparison |
| `fnn_vs_ml_edge_deployment.png` | FNN vs ML algorithms comparison |

---

## License

MIT License  
Copyright (c) 2025 Sai Vidya Institute of Technology

**Patent Notice**: This software is protected by a patent published by the Indian Patent Office (IPO). The TinyML-Driven Occupancy and Appliance Control System for Classroom Energy Reduction is covered under applicable patent and intellectual property laws.

See [LICENSE](LICENSE) for full details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Acknowledgments

- Sai Vidya Institute of Technology (SVIT)
- TensorFlow and Flask communities
