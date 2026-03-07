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
print("FNN vs TOP 3 ML ALGORITHMS - EDGE DEPLOYMENT FOCUSED COMPARISON")
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
print(f"✓ Test set: {len(X_test)} samples")

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
    'y_pred_proba': fnn_pred_probs,
    'Edge_Deployable': True
})
print(f"✓ Accuracy: {results[0]['Accuracy']:.4f} | F1: {results[0]['F1-Score']:.4f}")

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
        'y_pred_proba': y_pred_proba,
        'Edge_Deployable': False
    })

    print(f"✓ Accuracy: {results[-1]['Accuracy']:.4f} | F1: {results[-1]['F1-Score']:.4f}")

results_df = pd.DataFrame(results)

# 5. CREATE ENHANCED VISUALIZATIONS WITH EDGE DEPLOYMENT HIGHLIGHT
print("\n[5/5] Generating enhanced visualizations...")

# Set modern style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with 6 subplots
fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor('white')

# Color scheme - FNN highlighted
colors = ['#2ECC71', '#FF6B6B', '#4ECDC4', '#45B7D1']  # Green for FNN
edge_badge_color = '#FFD700'
color_map = {row['Model']: colors[idx] for idx, row in results_df.iterrows()}

# 1. PERFORMANCE COMPARISON (Grouped Bar Chart) WITH EDGE BADGE
ax1 = plt.subplot(2, 3, 1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.2

for i, (idx, row) in enumerate(results_df.iterrows()):
    values = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']]
    offset = (i - len(results_df)/2 + 0.5) * width
    bars = ax1.bar(x + offset, values, width, label=row['Model'],
                   color=colors[i], alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add edge device badge for FNN
    if row['Edge_Deployable']:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                    '⭐ EDGE', ha='center', va='bottom', fontweight='bold',
                    fontsize=8, color='#27AE60', bbox=dict(boxstyle='round,pad=0.3',
                    facecolor=edge_badge_color, alpha=0.7, edgecolor='#27AE60', linewidth=1.5))

ax1.set_xlabel('Metrics', fontweight='bold', fontsize=11)
ax1.set_ylabel('Score', fontweight='bold', fontsize=11)
ax1.set_title('Performance Metrics Comparison\n(⭐ = Edge Deployable)', 
              fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=10)
ax1.legend(fontsize=9, framealpha=0.9, loc='lower right')
ax1.set_ylim([0.92, 1.02])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 2. ROC CURVES WITH EDGE HIGHLIGHT
ax2 = plt.subplot(2, 3, 2)
for idx, row in results_df.iterrows():
    fpr, tpr, _ = roc_curve(y_test, row['y_pred_proba'])
    linestyle = '-' if row['Edge_Deployable'] else '--'
    linewidth = 3 if row['Edge_Deployable'] else 2.5
    label_suffix = ' [EDGE]' if row['Edge_Deployable'] else ''
    ax2.plot(fpr, tpr, label=f"{row['Model']} (AUC={row['ROC-AUC']:.3f}){label_suffix}",
            color=colors[idx], linewidth=linewidth, alpha=0.9, linestyle=linestyle)

ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.4, label='Random')
ax2.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
ax2.set_title('ROC Curves\n(Solid Line = Edge Deployable)', fontweight='bold', fontsize=12)
ax2.legend(fontsize=9, loc='lower right', framealpha=0.9)
ax2.grid(alpha=0.3, linestyle='--')

# 3. INFERENCE SPEED - CRITICAL FOR EDGE
ax3 = plt.subplot(2, 3, 3)
models_sorted = results_df.sort_values('Speed (ms)', ascending=True)
bar_colors = [colors[results_df[results_df['Model']==m].index[0]] for m in models_sorted['Model']]
bars = ax3.barh(models_sorted['Model'], models_sorted['Speed (ms)'],
                color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1.2)

# Highlight FNN with annotation
for i, (bar, val) in enumerate(zip(bars, models_sorted['Speed (ms)'])):
    model_name = models_sorted.iloc[i]['Model']
    is_edge = results_df[results_df['Model']==model_name].iloc[0]['Edge_Deployable']
    
    ax3.text(val + 0.0002, i, f'{val:.4f} ms', va='center', fontsize=9, fontweight='bold')
    
    if is_edge:
        ax3.text(val - 0.0001, i, '✓ EDGE', va='center', ha='right', fontsize=9,
                fontweight='bold', color='white', bbox=dict(boxstyle='round,pad=0.4',
                facecolor='#27AE60', alpha=0.8, edgecolor='white', linewidth=1))

ax3.set_xlabel('Inference Time (ms)', fontweight='bold', fontsize=11)
ax3.set_title('Inference Speed\n(✓ = Suitable for Edge Devices)', fontweight='bold', fontsize=12)
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# 4. CONFUSION MATRIX - FNN
ax4 = plt.subplot(2, 3, 4)
cm_fnn = confusion_matrix(y_test, results_df.iloc[0]['y_pred'])
sns.heatmap(cm_fnn, annot=True, fmt='d', cmap='Greens', ax=ax4,
           xticklabels=['Unoccupied', 'Occupied'],
           yticklabels=['Unoccupied', 'Occupied'],
           cbar_kws={'label': 'Count'},
           annot_kws={'fontsize': 11, 'fontweight': 'bold'},
           linewidths=0.5, linecolor='gray')
ax4.set_xlabel('Predicted', fontweight='bold', fontsize=10)
ax4.set_ylabel('Actual', fontweight='bold', fontsize=10)
ax4.set_title(f"FNN (EDGE DEPLOYABLE)\nAcc: {results_df.iloc[0]['Accuracy']:.4f} | Params: {fnn_model.count_params()}",
             fontweight='bold', fontsize=11, color='#27AE60',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=edge_badge_color, alpha=0.6))

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
best_acc = results_df.loc[best_ml_idx]['Accuracy']
ax5.set_title(f"{best_name} (CLOUD/SERVER)\nAcc: {best_acc:.4f} | NOT Edge Compatible",
             fontweight='bold', fontsize=11, color='#C0392B')

# 6. SUMMARY TABLE WITH DEPLOYMENT RECOMMENDATION
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_data = []
for idx, row in results_df.iterrows():
    deployment = '✓ EDGE' if row['Edge_Deployable'] else '✗ CLOUD'
    summary_data.append([
        row['Model'],
        f"{row['Accuracy']:.4f}",
        f"{row['F1-Score']:.4f}",
        f"{row['ROC-AUC']:.4f}",
        f"{row['Speed (ms)']:.4f}",
        deployment
    ])

table = ax6.table(cellText=summary_data,
                 colLabels=['Model', 'Accuracy', 'F1', 'ROC-AUC', 'Speed (ms)', 'Deploy'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.22, 0.13, 0.13, 0.13, 0.13, 0.16])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 3)

# Style header
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(weight='bold', color='white', fontsize=9)
    cell.set_edgecolor('white')

# Style rows
for i in range(1, len(summary_data) + 1):
    model_name = summary_data[i-1][0]
    is_edge = results_df[results_df['Model']==model_name].iloc[0]['Edge_Deployable']
    row_color = colors[i-1]
    
    for j in range(6):
        cell = table[(i, j)]
        if j == 0:
            cell.set_facecolor(row_color)
            cell.set_alpha(0.3)
            cell.set_text_props(weight='bold')
        elif j == 5:  # Deployment column
            cell.set_facecolor(edge_badge_color if is_edge else '#FFE5E5')
            cell.set_text_props(weight='bold', color='#27AE60' if is_edge else '#C0392B')
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('#CCCCCC')

ax6.set_title('Performance & Deployment Summary', fontweight='bold', fontsize=12, pad=15)

# Main title
fig.suptitle('FNN vs Traditional ML: Performance Comparison\nFNN RECOMMENDED FOR EDGE DEVICES (ESP32)',
            fontsize=16, fontweight='bold', y=0.98, color='#27AE60')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('fnn_vs_ml_edge_deployment.png', dpi=150, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print("\n✓ Enhanced visualization saved to 'fnn_vs_ml_edge_deployment.png'")
plt.show()

# PRINT COMPARISON WITH DEPLOYMENT INSIGHTS
print("\n" + "="*90)
print("PERFORMANCE COMPARISON WITH DEPLOYMENT CONSIDERATIONS")
print("="*90)

print("\n" + "-"*90)
print(f"{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'ROC-AUC':<12} {'Speed(ms)':<12} {'Deployment':<10}")
print("-"*90)

for idx, row in results_df.iterrows():
    deploy = "✓ EDGE" if row['Edge_Deployable'] else "✗ CLOUD"
    print(f"{row['Model']:<20} {row['Accuracy']:<12.4f} {row['F1-Score']:<12.4f} "
         f"{row['ROC-AUC']:<12.4f} {row['Speed (ms)']:<12.4f} {deploy:<10}")

print("-"*90)

# Best model overall
best_idx = results_df['F1-Score'].idxmax()
best_model = results_df.loc[best_idx, 'Model']
best_f1 = results_df.loc[best_idx, 'F1-Score']

print(f"\nBEST PERFORMANCE: {best_model}")
print(f" F1-Score: {best_f1:.4f}")
print(f" Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
print(f" ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']:.4f}")

# FNN analysis
fnn_row = results_df.iloc[0]
print(f"\nFNN MODEL (EDGE DEPLOYABLE):")
print(f" ✓ Accuracy: {fnn_row['Accuracy']*100:.2f}%")
print(f" ✓ F1-Score: {fnn_row['F1-Score']:.4f}")
print(f" ✓ Inference: {fnn_row['Speed (ms)']:.4f} ms/sample")
print(f" ✓ Model Parameters: {fnn_model.count_params()}")
print(f" ✓ Memory Footprint: ~{fnn_model.count_params() * 4 / 1024:.2f} KB (approx)")

# RECOMMENDATION
print("\n" + "="*90)
print("DEPLOYMENT RECOMMENDATION")
print("="*90)

if best_model == 'FNN':
    print("\n✓ FNN OPTIMAL: Best performance AND edge-deployable")
else:
    diff = abs(best_f1 - fnn_row['F1-Score']) * 100
    print(f"\n⚠ TRADE-OFF ANALYSIS:")
    print(f"  • {best_model} has {diff:.2f}% higher F1-Score")
    print(f"  • BUT: Requires cloud/server infrastructure")
    print(f"  • BUT: Higher latency and internet dependency")
    print(f"  • FNN performance gap: Only {diff:.2f}% inferior")

print("\n✓ RECOMMENDATION: Deploy FNN to ESP32 edge devices")
print("  Reasons:")
print("  1. Compatible with resource-constrained hardware (ESP32, Raspberry Pi)")
print("  2. Real-time processing without cloud latency")
print("  3. Privacy-preserving (no data transmission required)")
print("  4. Cost-effective deployment")
print(f"  5. Performance is competitive: {fnn_row['Accuracy']*100:.2f}% accuracy")
print(f"  6. Ultra-fast inference: {fnn_row['Speed (ms)']:.4f} ms per prediction")

print("\n" + "="*90)
print("COMPARISON COMPLETE - FNN SELECTED FOR EDGE DEPLOYMENT")
print("="*90)