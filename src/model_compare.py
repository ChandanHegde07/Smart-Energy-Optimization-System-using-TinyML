import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split  # Better splitting
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    print("Warning: XGBoost not installed. Install with: pip install xgboost")
    HAS_XGBOOST = False


# CONFIGURATION
ENGINEERED_DATA_PATH = 'Sensor_Data_Engineered.csv'

FEATURES = [
    'Temperature', 'Light',
    'Light_mean_3', 'Light_diff_3',
    'Temp_mean_3', 'Temp_diff_3',
    'hour_sin', 'hour_cos'
]
TARGET = 'PIR'


# 1. LOAD AND PREPARE DATA
print("="*70)
print("OCCUPANCY DETECTION - ML ALGORITHM COMPARISON")
print("="*70)

print("\n[1/5] Loading feature-engineered data...")
try:
    df = pd.read_csv(ENGINEERED_DATA_PATH, parse_dates=['date'])
    print(f"✓ Loaded {len(df)} samples")
except FileNotFoundError:
    print(f"Error: Could not find '{ENGINEERED_DATA_PATH}'")
    print("Please run train.ipynb first to generate the engineered dataset.")
    exit(1)

# Check class distribution
print("\n[2/5] Checking class distribution...")
class_counts = df[TARGET].value_counts()
print(f"  Class 0 (Unoccupied): {class_counts.get(0, 0)} samples")
print(f"  Class 1 (Occupied): {class_counts.get(1, 0)} samples")
print(f"  Imbalance ratio: {class_counts.max() / class_counts.min():.2f}:1")

# FIXED: Use stratified split instead of chronological to ensure both classes in test set
print("\n[3/5] Splitting data with stratification (70/15/15)...")
X = df[FEATURES].values
y = df[TARGET].values

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Second split: 15% validation, 15% test (from the 30% temp)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"  Training:   {len(X_train)} samples (Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)})")
print(f"  Validation: {len(X_val)} samples (Class 0: {sum(y_val==0)}, Class 1: {sum(y_val==1)})")
print(f"  Test:       {len(X_test)} samples (Class 0: {sum(y_test==0)}, Class 1: {sum(y_test==1)})")

# Verify both classes present
if len(np.unique(y_test)) < 2:
    print("ERROR: Test set contains only one class. Adjusting split strategy...")
    exit(1)

print("\n[4/5] Scaling features...")
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"  Class weights: {class_weight_dict}")


# 2. DEFINE ML ALGORITHMS
print("\n[5/5] Initializing ML algorithms...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    'Naive Bayes': GaussianNB()
}

if HAS_XGBOOST:
    scale_pos_weight = class_weights[1] / class_weights[0]
    models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=5, 
                                      scale_pos_weight=scale_pos_weight, 
                                      random_state=42, eval_metric='logloss')

print(f"✓ Initialized {len(models)} algorithms")


# 3. TRAIN AND EVALUATE ALL MODELS
print("\n" + "="*70)
print("TRAINING AND EVALUATING MODELS")
print("="*70)

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    try:
        # Start timer
        start_time = time.time()
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Training time
        train_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC AUC with error handling
        roc_auc = None
        if y_pred_proba is not None:
            try:
                # Check if both classes are present
                if len(np.unique(y_test)) == 2:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                else:
                    print(f"  ⚠ Warning: Only one class in test set, ROC AUC not calculated")
            except ValueError as e:
                print(f"  ⚠ Warning: Could not calculate ROC AUC - {str(e)}")
                roc_auc = None
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Train Time (s)': train_time,
            'Inference Time (ms)': inference_time,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'model_obj': model
        })
        
        print(f"  ✓ Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Train Time: {train_time:.2f}s")
        
    except Exception as e:
        print(f"  ✗ Error training {name}: {str(e)}")
        continue

# Create results DataFrame
results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("PERFORMANCE COMPARISON - ALL MODELS")
print("="*70)
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 
                   'Train Time (s)', 'Inference Time (ms)']].to_string(index=False))


# 4. VISUALIZATIONS
print("\n" + "="*70)
print("GENERATING COMPARISON VISUALIZATIONS")
print("="*70)

sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 12))

# 1. Accuracy Comparison
ax1 = plt.subplot(3, 3, 1)
results_sorted = results_df.sort_values('Accuracy', ascending=True)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results_sorted)))
ax1.barh(results_sorted['Model'], results_sorted['Accuracy'], color=colors)
ax1.set_xlabel('Accuracy')
ax1.set_title('Model Accuracy Comparison', fontweight='bold')
ax1.set_xlim([0, 1])
for i, v in enumerate(results_sorted['Accuracy']):
    ax1.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=8)

# 2. F1-Score Comparison
ax2 = plt.subplot(3, 3, 2)
results_sorted = results_df.sort_values('F1-Score', ascending=True)
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(results_sorted)))
ax2.barh(results_sorted['Model'], results_sorted['F1-Score'], color=colors)
ax2.set_xlabel('F1-Score')
ax2.set_title('Model F1-Score Comparison', fontweight='bold')
ax2.set_xlim([0, 1])
for i, v in enumerate(results_sorted['F1-Score']):
    ax2.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=8)

# 3. Precision vs Recall
ax3 = plt.subplot(3, 3, 3)
ax3.scatter(results_df['Recall'], results_df['Precision'], s=200, alpha=0.6, 
           c=range(len(results_df)), cmap='coolwarm')
for idx, row in results_df.iterrows():
    ax3.annotate(row['Model'], (row['Recall'], row['Precision']), 
                fontsize=8, ha='right', va='bottom')
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision vs Recall Trade-off', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 1.05])
ax3.set_ylim([0, 1.05])

# 4. ROC Curves (only for models with ROC AUC)
ax4 = plt.subplot(3, 3, 4)
has_roc = False
for idx, row in results_df.iterrows():
    if row['y_pred_proba'] is not None and row['ROC-AUC'] is not None:
        try:
            fpr, tpr, _ = roc_curve(y_test, row['y_pred_proba'])
            ax4.plot(fpr, tpr, label=f"{row['Model']} (AUC={row['ROC-AUC']:.3f})", linewidth=2)
            has_roc = True
        except:
            continue
if has_roc:
    ax4.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curves', fontweight='bold')
    ax4.legend(loc='lower right', fontsize=7)
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'ROC curves not available', ha='center', va='center')
    ax4.set_title('ROC Curves', fontweight='bold')

# 5. Training Time
ax5 = plt.subplot(3, 3, 5)
results_sorted = results_df.sort_values('Train Time (s)', ascending=True)
colors = plt.cm.autumn(np.linspace(0.3, 0.9, len(results_sorted)))
ax5.barh(results_sorted['Model'], results_sorted['Train Time (s)'], color=colors)
ax5.set_xlabel('Training Time (seconds)')
ax5.set_title('Training Time Comparison', fontweight='bold')
for i, v in enumerate(results_sorted['Train Time (s)']):
    ax5.text(v + 0.01, i, f'{v:.2f}s', va='center', fontsize=8)

# 6. Inference Time
ax6 = plt.subplot(3, 3, 6)
results_sorted = results_df.sort_values('Inference Time (ms)', ascending=True)
colors = plt.cm.summer(np.linspace(0.3, 0.9, len(results_sorted)))
ax6.barh(results_sorted['Model'], results_sorted['Inference Time (ms)'], color=colors)
ax6.set_xlabel('Inference Time (ms per sample)')
ax6.set_title('Inference Speed Comparison', fontweight='bold')
for i, v in enumerate(results_sorted['Inference Time (ms)']):
    ax6.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)

# 7. Metrics Heatmap
ax7 = plt.subplot(3, 3, 7)
metrics_df = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].set_index('Model')
sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax7, cbar_kws={'label': 'Score'})
ax7.set_title('Performance Metrics Heatmap', fontweight='bold')
ax7.set_ylabel('')

# 8. Confusion Matrix for Best Model
ax8 = plt.subplot(3, 3, 8)
best_model_idx = results_df['F1-Score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_y_pred = results_df.loc[best_model_idx, 'y_pred']
cm = confusion_matrix(y_test, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax8, 
            xticklabels=['Unoccupied', 'Occupied'],
            yticklabels=['Unoccupied', 'Occupied'])
ax8.set_xlabel('Predicted')
ax8.set_ylabel('Actual')
ax8.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')

# 9. Radar Chart for Top 3
ax9 = plt.subplot(3, 3, 9, projection='polar')
top3_models = results_df.nlargest(3, 'F1-Score')
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for idx, row in top3_models.iterrows():
    values = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']]
    values += values[:1]
    ax9.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
    ax9.fill(angles, values, alpha=0.15)

ax9.set_theta_offset(np.pi / 2)
ax9.set_theta_direction(-1)
ax9.set_xticks(angles[:-1])
ax9.set_xticklabels(categories, fontsize=8)
ax9.set_ylim(0, 1)
ax9.set_title('Top 3 Models - Performance Radar', fontweight='bold', pad=20)
ax9.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax9.grid(True)

plt.tight_layout()
plt.savefig('ml_algorithm_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Comparison visualizations saved to 'ml_algorithm_comparison.png'")
plt.show()


# 5. DETAILED REPORT
print("\n" + "="*70)
print(f"DETAILED REPORT - BEST MODEL: {best_model_name}")
print("="*70)
print(classification_report(y_test, best_y_pred, 
                          target_names=['Unoccupied', 'Occupied'],
                          digits=4))

# 6. RANKING
print("\n" + "="*70)
print("RANKING BY F1-SCORE")
print("="*70)
ranking = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].sort_values('F1-Score', ascending=False)
ranking.index = range(1, len(ranking) + 1)
print(ranking.to_string())

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\n✓ Best Model: {best_model_name}")
print(f"✓ Best F1-Score: {results_df.loc[best_model_idx, 'F1-Score']:.4f}")
print(f"✓ Visualization saved: ml_algorithm_comparison.png")
