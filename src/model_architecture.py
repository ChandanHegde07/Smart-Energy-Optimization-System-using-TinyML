import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import os

# Path to your trained model
MODEL_PATH = 'occupancy_fnn_model.h5'

print("="*70)
print("MODEL ARCHITECTURE VISUALIZATION")
print("="*70)

# Load the trained model
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"\n✓ Model loaded successfully from '{MODEL_PATH}'")
except:
    print(f"\nError: Could not find '{MODEL_PATH}'")
    print("Make sure you've trained the model first!")
    exit(1)

# Display model summary in console
print("\nModel Summary:")
print("-" * 70)
model.summary()

# Generate architecture diagram
print("\nGenerating architecture diagram...")
plot_model(
    model,
    to_file='model_architecture.png',
    show_shapes=True,           # Show input/output shapes
    show_layer_names=True,      # Show layer names
    rankdir='TB',               # Top to Bottom layout
    expand_nested=False,
    dpi=150,
    show_layer_activations=True # Show activation functions
)
print("✓ Architecture diagram saved as 'model_architecture.png'")

print("\nGenerating activation function plots...")

# Define activation functions
def relu(x):
    """ReLU: Returns max(0, x)"""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid: Returns 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-x))

# Generate input values
x = np.linspace(-10, 10, 1000)

# Calculate activation outputs
relu_output = relu(x)
sigmoid_output = sigmoid(x)

# Create visualization with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ---------------------- RELU PLOT ----------------------
axes[0].plot(x, relu_output, linewidth=3, color='#2E86AB', label='ReLU(x)')
axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('Input (x)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Output', fontsize=12, fontweight='bold')
axes[0].set_title('ReLU Activation Function\n(Used in Hidden Layers)', 
                  fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].set_ylim(-1, 10)

# Add annotation
axes[0].annotate('f(x) = max(0, x)', 
                xy=(5, 5), xytext=(2, 7),
                fontsize=11, color='#2E86AB',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

axes[0].text(-8, 8.5, 'Output = 0\nfor x < 0', 
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0].text(5, 2, 'Output = x\nfor x > 0', 
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# ---------------------- SIGMOID PLOT ----------------------
axes[1].plot(x, sigmoid_output, linewidth=3, color='#A23B72', label='Sigmoid(x)')
axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
axes[1].axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
axes[1].axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Decision boundary (0.5)')
axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('Input (x)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Output (Probability)', fontsize=12, fontweight='bold')
axes[1].set_title('Sigmoid Activation Function\n(Used in Output Layer)', 
                  fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].set_ylim(-0.1, 1.1)

# Add annotation
axes[1].annotate('f(x) = 1/(1 + e^(-x))', 
                xy=(0, 0.5), xytext=(3, 0.3),
                fontsize=11, color='#A23B72',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

axes[1].text(-8, 0.9, 'Unoccupied\n(< 0.5)', 
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
axes[1].text(5, 0.9, 'Occupied\n(> 0.5)', 
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
print("✓ Activation functions saved as 'activation_functions.png'")
plt.show()

print("\n" + "="*70)
print("DETAILED LAYER INFORMATION")
print("="*70)

print("\n{:<15} {:<20} {:<25} {:<15}".format(
    "Layer Name", "Type", "Output Shape", "Activation"))
print("-" * 75)

for layer in model.layers:
    layer_type = layer.__class__.__name__
    
    # FIXED: Get output shape correctly
    try:
        output_shape = str(layer.output.shape)
    except:
        output_shape = "N/A"
    
    # Get activation function
    if hasattr(layer, 'activation'):
        activation = layer.activation.__name__
    else:
        activation = "None"
    
    print("{:<15} {:<20} {:<25} {:<15}".format(
        layer.name, 
        layer_type, 
        output_shape, 
        activation
    ))

print("\nGenerating combined architecture + activation visualization...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

# Architecture text representation
ax_arch = fig.add_subplot(gs[0:2, 0])
ax_arch.axis('off')

architecture_text = """
FEED-FORWARD NEURAL NETWORK ARCHITECTURE
─────────────────────────────────────────

INPUT LAYER (8 features)
   ↓
   Temperature, Light, Engineered Features
   
HIDDEN LAYER 1 (32 neurons)
   ↓  ReLU Activation
   Dense + Dropout(0.2)
   
HIDDEN LAYER 2 (16 neurons)
   ↓  ReLU Activation  
   Dense + Dropout(0.2)
   
OUTPUT LAYER (1 neuron)
   ↓  Sigmoid Activation
   Binary Classification: [0, 1]
   
PARAMETERS: ~800 (lightweight for ESP32)
"""

ax_arch.text(0.1, 0.5, architecture_text, 
            fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# ReLU plot
ax_relu = fig.add_subplot(gs[0, 1])
ax_relu.plot(x, relu_output, linewidth=2.5, color='#2E86AB')
ax_relu.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax_relu.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax_relu.grid(True, alpha=0.3)
ax_relu.set_xlabel('Input (x)', fontsize=10)
ax_relu.set_ylabel('Output', fontsize=10)
ax_relu.set_title('ReLU (Hidden Layers)', fontsize=12, fontweight='bold')
ax_relu.set_ylim(-1, 8)

# Sigmoid plot
ax_sigmoid = fig.add_subplot(gs[1, 1])
ax_sigmoid.plot(x, sigmoid_output, linewidth=2.5, color='#A23B72')
ax_sigmoid.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
ax_sigmoid.grid(True, alpha=0.3)
ax_sigmoid.set_xlabel('Input (x)', fontsize=10)
ax_sigmoid.set_ylabel('Output (Probability)', fontsize=10)
ax_sigmoid.set_title('Sigmoid (Output Layer)', fontsize=12, fontweight='bold')
ax_sigmoid.set_ylim(-0.1, 1.1)

# Comparison table
ax_table = fig.add_subplot(gs[2, :])
ax_table.axis('off')

comparison_data = [
    ['Property', 'ReLU', 'Sigmoid'],
    ['Formula', 'max(0, x)', '1/(1 + e^(-x))'],
    ['Output Range', '[0, ∞)', '[0, 1]'],
    ['Used In', 'Hidden Layers', 'Output Layer'],
    ['Purpose', 'Learn Complex Patterns', 'Probability Output'],
    ['Gradient', 'Constant (1) for x>0', 'Vanishes at extremes'],
]

table = ax_table.table(cellText=comparison_data, 
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.2, 0.4, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style alternating rows
for i in range(1, len(comparison_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E8F5E9')

plt.savefig('complete_architecture_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Combined visualization saved as 'complete_architecture_visualization.png'")
plt.show()

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  1. model_architecture.png - Keras architecture diagram")
print("  2. activation_functions.png - ReLU and Sigmoid plots")
print("  3. complete_architecture_visualization.png - Combined view")
print("\nAll visualizations clearly show ReLU (hidden layers) and Sigmoid (output)!")

