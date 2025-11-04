"""
fnn_vs_ml_comparison.py - Clean Comparison: FNN vs Top 3 ML Models
==================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                            recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    print("Warning: XGBoost not installed. Skipping XGBoost comparison.")
    HAS_XGBOOST = False

# CONFIGURATION
ENGINEERED_DATA_PATH = 'Sensor_Data_Engineered.csv'
FNN_MODEL_PATH = 'occupancy_fnn_model.h5'
SCALER_PATH = 'scaler.pkl'

FEATURES = [
    'Temperature', 'Light',
    'Light_mean_3', 'Light_diff_3',
    'Temp_mean_3', 'Temp_diff_3',
    'hour_sin', 'hour_cos'
]
TARGET = 'PIR'

print("="*80)
print("FNN vs TOP 3 ML ALGORITHMS - CLEAN COMPARISON")
print("="*80)

# 1. LOAD DATA
print("\n[1/5] Loading data and preparing splits...")
try:
    df = pd.read_csv(ENGINEERED_DATA_PATH, parse_dates=['date'])
    print(f"✓ Loaded {len(df)} samples")
except FileNotFoundError:
    print(f"Error: Could not find '{ENGINEERED_DATA_PATH}'")
    exit(1)

X = df[FEATURES].values
y = df[TARGET].values

# Stratified split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"  Test set: {len(X_test)} samples")

# 2. LOAD SCALER AND SCALE DATA
print("\n[2/5] Loading scaler and scaling features...")
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler loaded")
except FileNotFoundError:
    print("Scaler not found, creating new one...")
    scaler = StandardScaler()
    scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. LOAD FNN MODEL
print("\n[3/5] Loading trained FNN model...")
try:
    fnn_model = keras.models.load_model(FNN_MODEL_PATH)
    print(f"✓ FNN model loaded ({fnn_model.count_params()} parameters)")
except FileNotFoundError:
    print(f"Error: Could not find '{FNN_MODEL_PATH}'")
    exit(1)

# 4. TRAIN TOP 3 ML MODELS
print("\n[4/5] Training top 3 traditional ML models...")

# Compute class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Define models
ml_models = {}

if HAS_XGBOOST:
    scale_pos_weight = class_weights[1] / class_weights[0]
    ml_models['XGBoost'] = XGBClassifier(
        n_estimators=100, max_depth=5, 
        scale_pos_weight=scale_pos_weight, 
        random_state=42, eval_metric='logloss'
    )

ml_models['Gradient Boosting'] = GradientBoostingClassifier(
    n_estimators=100, max_depth=5, random_state=42
)

ml_models['K-Nearest Neighbors'] = KNeighborsClassifier(n_neighbors=5)

# Evaluate all models
results = []

# Evaluate FNN
print("\nEvaluating FNN...")
start_time = time.time()
fnn_pred_probs = fnn_model.predict(X_test_scaled, verbose=0).ravel()
fnn_inference_time = (time.time() - start_time) / len(X_test) * 1000
fnn_pred = (fnn_pred_probs >= 0.5).astype(int)

results.append({
    'Model': 'FNN',
    'Accuracy': accuracy_score(y_test, fnn_pred),
    'Precision': precision_score(y_test, fnn_pred, zero_division=0),
    'Recall': recall_score(y_test, fnn_pred, zero_division=0),
    'F1-Score': f1_score(y_test, fnn_pred, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, fnn_pred_probs),
    'Speed (ms)': fnn_inference_time,
    'y_pred': fnn_pred,
    'y_pred_proba': fnn_pred_probs
})

print(f"  ✓ Accuracy: {results[0]['Accuracy']:.4f} | F1: {results[0]['F1-Score']:.4f}")

# Evaluate ML models
for name, model in ml_models.items():
    print(f"\nTraining {name}...")
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    inference_time = (time.time() - start_time) / len(X_test) * 1000
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
        'Speed (ms)': inference_time,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    })
    
    print(f"  ✓ Accuracy: {results[-1]['Accuracy']:.4f} | F1: {results[-1]['F1-Score']:.4f}")

results_df = pd.DataFrame(results)

# 5. CREATE CLEAN VISUALIZATIONS
print("\n[5/5] Generating clean visualizations...")

# Set modern style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with 6 subplots (2 rows x 3 columns)
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('white')

# Color scheme
colors = ['#9D4EDD', '#FF6B6B', '#4ECDC4', '#45B7D1']
color_map = {row['Model']: colors[idx] for idx, row in results_df.iterrows()}

# 1. PERFORMANCE COMPARISON (Grouped Bar Chart)
ax1 = plt.subplot(2, 3, 1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.2

for i, (idx, row) in enumerate(results_df.iterrows()):
    values = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']]
    offset = (i - len(results_df)/2 + 0.5) * width
    bars = ax1.bar(x + offset, values, width, label=row['Model'], 
                   color=colors[i], alpha=0.85, edgecolor='black', linewidth=1.2)

ax1.set_xlabel('Metrics', fontweight='bold', fontsize=11)
ax1.set_ylabel('Score', fontweight='bold', fontsize=11)
ax1.set_title('Performance Metrics Comparison', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=10)
ax1.legend(fontsize=9, framealpha=0.9)
ax1.set_ylim([0.92, 1.0])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 2. ROC CURVES
ax2 = plt.subplot(2, 3, 2)
for idx, row in results_df.iterrows():
    fpr, tpr, _ = roc_curve(y_test, row['y_pred_proba'])
    ax2.plot(fpr, tpr, label=f"{row['Model']} (AUC={row['ROC-AUC']:.3f})", 
            color=colors[idx], linewidth=2.5, alpha=0.8)

ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.4, label='Random')
ax2.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
ax2.set_title('ROC Curves', fontweight='bold', fontsize=12)
ax2.legend(fontsize=9, loc='lower right', framealpha=0.9)
ax2.grid(alpha=0.3, linestyle='--')

# 3. INFERENCE SPEED
ax3 = plt.subplot(2, 3, 3)
models_sorted = results_df.sort_values('Speed (ms)', ascending=True)
bars = ax3.barh(models_sorted['Model'], models_sorted['Speed (ms)'], 
               color=[colors[results_df[results_df['Model']==m].index[0]] for m in models_sorted['Model']], 
               alpha=0.85, edgecolor='black', linewidth=1.2)

for i, (bar, val) in enumerate(zip(bars, models_sorted['Speed (ms)'])):
    ax3.text(val + 0.0002, i, f'{val:.4f} ms', va='center', fontsize=9, fontweight='bold')

ax3.set_xlabel('Inference Time (ms)', fontweight='bold', fontsize=11)
ax3.set_title('Inference Speed', fontweight='bold', fontsize=12)
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# 4. CONFUSION MATRIX - FNN
ax4 = plt.subplot(2, 3, 4)
cm_fnn = confusion_matrix(y_test, results_df.iloc[0]['y_pred'])
sns.heatmap(cm_fnn, annot=True, fmt='d', cmap='Purples', ax=ax4,
            xticklabels=['Unoccupied', 'Occupied'],
            yticklabels=['Unoccupied', 'Occupied'],
            cbar_kws={'label': 'Count'},
            annot_kws={'fontsize': 11, 'fontweight': 'bold'},
            linewidths=0.5, linecolor='gray')
ax4.set_xlabel('Predicted', fontweight='bold', fontsize=10)
ax4.set_ylabel('Actual', fontweight='bold', fontsize=10)
ax4.set_title(f"FNN - Acc: {results_df.iloc[0]['Accuracy']:.4f}", 
             fontweight='bold', fontsize=11, color=colors[0])

# 5. CONFUSION MATRIX - BEST ML MODEL
ax5 = plt.subplot(2, 3, 5)
best_ml_idx = results_df.iloc[1:]['F1-Score'].idxmax()
cm_best = confusion_matrix(y_test, results_df.loc[best_ml_idx]['y_pred'])
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax5,
            xticklabels=['Unoccupied', 'Occupied'],
            yticklabels=['Unoccupied', 'Occupied'],
            cbar_kws={'label': 'Count'},
            annot_kws={'fontsize': 11, 'fontweight': 'bold'},
            linewidths=0.5, linecolor='gray')
ax5.set_xlabel('Predicted', fontweight='bold', fontsize=10)
ax5.set_ylabel('Actual', fontweight='bold', fontsize=10)
best_name = results_df.loc[best_ml_idx]['Model']
ax5.set_title(f"{best_name} - Acc: {results_df.loc[best_ml_idx]['Accuracy']:.4f}", 
             fontweight='bold', fontsize=11, color=colors[best_ml_idx])

# 6. SUMMARY TABLE
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create clean summary table
summary_data = []
for idx, row in results_df.iterrows():
    summary_data.append([
        row['Model'],
        f"{row['Accuracy']:.4f}",
        f"{row['F1-Score']:.4f}",
        f"{row['ROC-AUC']:.4f}",
        f"{row['Speed (ms)']:.4f}"
    ])

table = ax6.table(cellText=summary_data,
                 colLabels=['Model', 'Accuracy', 'F1', 'ROC-AUC', 'Speed'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.3, 0.175, 0.175, 0.175, 0.175])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.8)

# Style header
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(weight='bold', color='white', fontsize=10)
    cell.set_edgecolor('white')

# Style rows
for i in range(1, len(summary_data) + 1):
    model_name = summary_data[i-1][0]
    row_color = colors[i-1]
    
    for j in range(5):
        cell = table[(i, j)]
        if j == 0:
            cell.set_facecolor(row_color)
            cell.set_alpha(0.3)
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('#CCCCCC')

ax6.set_title('Performance Summary', fontweight='bold', fontsize=12, pad=15)

# Main title
fig.suptitle('FNN vs Traditional ML: Performance Comparison', 
            fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('fnn_vs_ml_clean_comparison.png', dpi=150, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print("\n✓ Clean visualization saved to 'fnn_vs_ml_clean_comparison.png'")
plt.show()

# PRINT COMPARISON
print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)
print("\n" + "-"*80)
print(f"{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'ROC-AUC':<12} {'Speed(ms)':<12}")
print("-"*80)
for idx, row in results_df.iterrows():
    print(f"{row['Model']:<20} {row['Accuracy']:<12.4f} {row['F1-Score']:<12.4f} "
          f"{row['ROC-AUC']:<12.4f} {row['Speed (ms)']:<12.4f}")
print("-"*80)

# Best model
best_idx = results_df['F1-Score'].idxmax()
best_model = results_df.loc[best_idx, 'Model']
best_f1 = results_df.loc[best_idx, 'F1-Score']

print(f"\nBEST MODEL: {best_model}")
print(f"   F1-Score: {best_f1:.4f}")
print(f"   Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
print(f"   ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']:.4f}")

# FNN analysis
fnn_row = results_df.iloc[0]
print(f"\nFNN MODEL:")
print(f"   Accuracy: {fnn_row['Accuracy']*100:.2f}%")
print(f"   F1-Score: {fnn_row['F1-Score']:.4f}")
print(f"   Inference: {fnn_row['Speed (ms)']:.4f} ms/sample")

if best_model == 'FNN':
    print("\nFNN outperforms all traditional ML models!")
else:
    diff = abs(best_f1 - fnn_row['F1-Score']) * 100
    print(f"\nFNN within {diff:.2f}% of best ML model")
    print("FNN advantage: Deployable to ESP32 edge devices")

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80)
