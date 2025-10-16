"""
model.py - Feed-Forward Neural Network for Occupancy Detection
=================================================================
This script trains a compact FNN model suitable for ESP32 deployment.
It includes:
- Model architecture optimized for TinyML
- Training with early stopping and class balancing
- Evaluation metrics and visualizations
- Model export to TensorFlow Lite with int8 quantization
- Threshold tuning for optimal precision-recall trade-off
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
import pickle
import os

# CONFIGURATION
ENGINEERED_DATA_PATH = 'Sensor_Data_Engineered.csv'
MODEL_SAVE_PATH = 'occupancy_fnn_model.h5'
TFLITE_MODEL_PATH = 'occupancy_fnn_int8.tflite'
SCALER_PATH = 'scaler.pkl'

# Model hyperparameters
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 15  

FEATURES = [
    'Temperature', 'Light',
    'Light_mean_3', 'Light_diff_3',
    'Temp_mean_3', 'Temp_diff_3',
    'hour_sin', 'hour_cos'
]
TARGET = 'PIR'

# 1. LOAD AND PREPARE DATA
print("="*70)
print("OCCUPANCY DETECTION - FNN MODEL TRAINING")
print("="*70)

print("\n[1/6] Loading feature-engineered data...")
try:
    df = pd.read_csv(ENGINEERED_DATA_PATH, parse_dates=['date'])
    print(f"✓ Loaded {len(df)} samples")
except FileNotFoundError:
    print(f"Error: Could not find '{ENGINEERED_DATA_PATH}'")
    print("Please run train.ipynb first to generate the engineered dataset.")
    exit(1)

print("\n[2/6] Splitting data chronologically (70/15/15)...")
train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.85)

X = df[FEATURES].values
y = df[TARGET].values

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:val_size], y[train_size:val_size]
X_test, y_test = X[val_size:], y[val_size:]

print(f"  Training:   {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test:       {len(X_test)} samples")

print("\n[3/6] Scaling features...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved to '{SCALER_PATH}'")

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"  Class weights: {class_weight_dict}")

# 2. BUILD THE FNN MODEL
print("\n[4/6] Building Feed-Forward Neural Network...")

def build_fnn_model(input_dim):
    """
    Build a compact FNN optimized for TinyML deployment.
    
    Architecture:
    - Input layer (8 features)
    - Dense 32 neurons + ReLU
    - Dense 16 neurons + ReLU
    - Output 1 neuron + Sigmoid (binary classification)
    
    Total parameters: ~800 (very small for ESP32)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name='input'),
        layers.Dense(32, activation='relu', name='hidden1'),
        layers.Dropout(0.2, name='dropout1'),  # Prevent overfitting
        layers.Dense(16, activation='relu', name='hidden2'),
        layers.Dropout(0.2, name='dropout2'),
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='OccupancyFNN')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    return model

model = build_fnn_model(input_dim=len(FEATURES))
model.summary()

# 3. TRAIN THE MODEL
print("\n[5/6] Training the model...")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Save the trained model
model.save(MODEL_SAVE_PATH)
print(f"\n✓ Model saved to '{MODEL_SAVE_PATH}'")

# 4. EVALUATE THE MODEL
print("\n[6/6] Evaluating model performance...")

y_val_pred_probs = model.predict(X_val, verbose=0).ravel()

precision, recall, thresholds = precision_recall_curve(y_val, y_val_pred_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[min(best_threshold_idx, len(thresholds)-1)]

print(f"\n✓ Optimal decision threshold: {best_threshold:.3f}")

y_val_pred = (y_val_pred_probs >= best_threshold).astype(int)
y_test_pred_probs = model.predict(X_test, verbose=0).ravel()
y_test_pred = (y_test_pred_probs >= best_threshold).astype(int)

print("\n" + "="*70)
print("VALIDATION SET PERFORMANCE")
print("="*70)
print(classification_report(y_val, y_val_pred, 
                          target_names=['Unoccupied', 'Occupied'],
                          digits=3))

print("\n" + "="*70)
print("TEST SET PERFORMANCE (Final Evaluation)")
print("="*70)
print(classification_report(y_test, y_test_pred,
                          target_names=['Unoccupied', 'Occupied'],
                          digits=3))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix (Test Set):")
print(f"                 Predicted")
print(f"               Unocc  Occ")
print(f"Actual Unocc    {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       Occ      {cm[1,0]:4d}  {cm[1,1]:4d}")

# 5. VISUALIZE TRAINING HISTORY
print("\nGenerating training visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss')
axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
axes[0, 0].set_title('Model Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0, 1].set_title('Model Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Precision-Recall Curve
axes[1, 0].plot(recall, precision, label=f'AP={average_precision_score(y_val, y_val_pred_probs):.3f}')
axes[1, 0].scatter(recall[best_threshold_idx], precision[best_threshold_idx], 
                  c='red', s=100, label=f'Best (thresh={best_threshold:.3f})')
axes[1, 0].set_title('Precision-Recall Curve')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Prediction Distribution
axes[1, 1].hist(y_val_pred_probs[y_val == 0], bins=50, alpha=0.5, label='Unoccupied', color='red')
axes[1, 1].hist(y_val_pred_probs[y_val == 1], bins=50, alpha=0.5, label='Occupied', color='green')
axes[1, 1].axvline(best_threshold, color='black', linestyle='--', label=f'Threshold={best_threshold:.3f}')
axes[1, 1].set_title('Prediction Probability Distribution')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
print("✓ Training visualizations saved to 'training_results.png'")
plt.show()

# 6. EXPORT TO TENSORFLOW LITE (INT8)
print("\n" + "="*70)
print("EXPORTING MODEL FOR ESP32 DEPLOYMENT")
print("="*70)

def representative_dataset():
    """Generate representative data for quantization."""
    for i in range(min(500, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

# Convert to TensorFlow Lite with int8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save the quantized model
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

tflite_size_kb = len(tflite_model) / 1024
print(f"\n✓ TensorFlow Lite model saved to '{TFLITE_MODEL_PATH}'")
print(f"  Model size: {tflite_size_kb:.2f} KB")

# 7. SAVE DEPLOYMENT PARAMETERS
deployment_params = {
    'features': FEATURES,
    'scaler_means': scaler.mean_.tolist(),
    'scaler_stds': scaler.scale_.tolist(),
    'threshold': float(best_threshold),
    'model_size_kb': float(tflite_size_kb)
}

with open('deployment_params.pkl', 'wb') as f:
    pickle.dump(deployment_params, f)

print("\n" + "="*70)
print("ESP32 DEPLOYMENT PARAMETERS")
print("="*70)
print("\nFeature Order:")
for i, feat in enumerate(FEATURES):
    print(f"  {i}: {feat}")

print("\nScaler Means (copy to ESP32):")
print("  float MEANS[] = {", end="")
print(", ".join([f"{m:.6f}" for m in scaler.mean_]), end="")
print("};")

print("\nScaler Stds (copy to ESP32):")
print("  float STDS[] = {", end="")
print(", ".join([f"{s:.6f}" for s in scaler.scale_]), end="")
print("};")

print(f"\nDecision Threshold:")
print(f"  float THRESHOLD = {best_threshold:.6f};")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nFiles generated:")
print(f"  1. {MODEL_SAVE_PATH} - Full Keras model")
print(f"  2. {TFLITE_MODEL_PATH} - Quantized model for ESP32 ({tflite_size_kb:.2f} KB)")
print(f"  3. {SCALER_PATH} - Feature scaler")
print(f"  4. deployment_params.pkl - All deployment parameters")
print(f"  5. training_results.png - Training visualization")
print("\nNext step: Upload '{TFLITE_MODEL_PATH}' to your ESP32")
