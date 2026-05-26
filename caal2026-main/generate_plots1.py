"""
Generating all M2 benchmark visualisation images from benchmark_results.txt
and all M1 result images from the saved model checkpoint.

Run from repo root:
    python generate_plots.py
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from model import GalaxyClassifierS4D
from model.functions import load_data

os.makedirs("images", exist_ok=True)

# ── M2 Benchmark plots ────────────────────────────────────────────────────────

opt_levels  = ['-O0', '-O1', '-O2', '-O3', '-Ofast']
total_times = [7957.893, 3344.294, 3201.373, 3332.582, 1773.894]
speedups    = [t / total_times[0] for t in total_times]
speedups    = [total_times[0] / t for t in total_times]

layers = ['hilbert_scan', 'uproject', 's4d_layer_1', 'gelu_1',
          's4d_layer_2', 'gelu_2', 'take_last', 'fc_head', 'softmax']

# per-layer times at each opt level (ms)
layer_times = {
    '-O0':    [0.031, 2.044, 3968.805, 9.995, 3967.559, 9.453, 0.001, 0.002, 0.002],
    '-O1':    [0.014, 0.381, 1664.504, 8.084, 1663.787, 7.519, 0.000, 0.001, 0.003],
    '-O2':    [0.015, 0.508, 1584.863, 8.054, 1600.478, 7.448, 0.000, 0.003, 0.003],
    '-O3':    [0.014, 0.755, 1661.813, 8.404, 1654.028, 7.563, 0.000, 0.001, 0.003],
    '-Ofast': [0.008, 0.765,  882.346, 1.276,  888.230, 1.267, 0.000, 0.001, 0.001],
}

# 1. Inference time vs optimization level
fig, ax1 = plt.subplots(figsize=(9, 5))
bars = ax1.bar(opt_levels, total_times, color='#4C72B0', alpha=0.8)
ax1.set_ylabel('Mean Inference Time (ms)')
ax1.set_xlabel('Optimization Level')
ax1.set_title('Inference Time vs Compiler Optimization Level')
for bar, t in zip(bars, total_times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{t:.0f}ms', ha='center', va='bottom', fontsize=9)
ax2 = ax1.twinx()
ax2.plot(opt_levels, speedups, 'o--', color='#C44E52', linewidth=2, markersize=8, label='Speedup vs -O0')
ax2.set_ylabel('Speedup vs -O0', color='#C44E52')
ax2.tick_params(axis='y', labelcolor='#C44E52')
for i, (opt, s) in enumerate(zip(opt_levels, speedups)):
    ax2.annotate(f'{s:.2f}x', (i, s), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('images/timing_vs_optimization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: images/timing_vs_optimization.png")

# 2. Per-layer timing breakdown at O2
fig, ax = plt.subplots(figsize=(10, 6))
times_O2 = layer_times['-O2']
pcts = [t / sum(times_O2) * 100 for t in times_O2]
colors = ['#C44E52' if 's4d' in l else '#4C72B0' for l in layers]
bars = ax.barh(layers, pcts, color=colors)
ax.set_xlabel('% of Total Inference Time')
ax.set_title('Per-Layer Timing Breakdown (-O2)')
for bar, pct, t in zip(bars, pcts, times_O2):
    ax.text(pct + 0.3, bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}% ({t:.1f}ms)', va='center', fontsize=8)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#C44E52', label='S4D layers (bottleneck)'),
                   Patch(facecolor='#4C72B0', label='Other layers')]
ax.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.savefig('images/per_layer_timing.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: images/per_layer_timing.png")

# 3. Memory footprint breakdown
fig, ax = plt.subplots(figsize=(7, 5))
categories = ['Model\nParameters', 'Activation\nBuffers', 'S4D Kernel\nBuffer']
sizes_kb = [82.5, 3088.3, 16.0]
colors_mem = ['#4C72B0', '#C44E52', '#55A868']
bars = ax.bar(categories, sizes_kb, color=colors_mem)
ax.set_ylabel('Memory (KB)')
ax.set_title(f'Memory Footprint Breakdown\nTotal: {sum(sizes_kb):.1f} KB ({sum(sizes_kb)/1024:.2f} MB)')
for bar, s in zip(bars, sizes_kb):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{s:.1f} KB', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('images/memory_footprint.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: images/memory_footprint.png")

# 4. C vs Python timing comparison
fig, ax = plt.subplots(figsize=(8, 5))
implementations = ['Python\n(PyTorch)', 'C (-O0)', 'C (-O1)', 'C (-O2)', 'C (-Ofast)']
# measure python time
import time
DEVICE = 'cpu'
COLORED = False
model = GalaxyClassifierS4D(colored=COLORED)
checkpoint = torch.load('model_params/galaxys4-29070.pth', map_location='cpu')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
x = torch.randn(1, 1, 64, 64)
# warmup
for _ in range(5):
    with torch.no_grad():
        model(x)
t0 = time.time()
for _ in range(100):
    with torch.no_grad():
        model(x)
python_ms = (time.time() - t0) / 100 * 1000
print(f"Python inference time: {python_ms:.1f} ms")

times_comparison = [python_ms, 7957.893, 3344.294, 3201.373, 1773.894]
colors_comp = ['#55A868', '#C44E52', '#4C72B0', '#4C72B0', '#4C72B0']
bars = ax.bar(implementations, times_comparison, color=colors_comp)
ax.set_ylabel('Mean Inference Time (ms)')
ax.set_title('C vs Python/PyTorch Inference Time Comparison')
for bar, t in zip(bars, times_comparison):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{t:.0f}ms', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('images/c_vs_pytorch.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: images/c_vs_pytorch.png")

# 5. Instruction count (from assembly line counts as proxy)
fig, ax = plt.subplots(figsize=(7, 5))
import subprocess
counts = {}
for opt, fname in [('O0', 'c_implementation/nn_O0.s'),
                   ('O2', 'c_implementation/nn_O2.s'),
                   ('O3', 'c_implementation/nn_O3.s')]:
    try:
        result = subprocess.run(['wc', '-l', fname], capture_output=True, text=True)
        counts[opt] = int(result.stdout.strip().split()[0])
    except:
        counts[opt] = 0

ax.bar(list(counts.keys()), list(counts.values()), color='#4C72B0')
ax.set_xlabel('Optimization Level')
ax.set_ylabel('Assembly Line Count')
ax.set_title('Assembly Size vs Optimization Level\n(Static instruction count proxy)')
for opt, count in counts.items():
    ax.text(list(counts.keys()).index(opt), count + 5, str(count), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('images/instruction_count.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: images/instruction_count.png")

# ── M1 plots ──────────────────────────────────────────────────────────────────

CLASSES = ['Smooth Round', 'Smooth Cigar', 'Edge-on Disk', 'Unbarred Spiral']
RNG_SEED = 29070
BATCH_SIZE = 64

print("\nLoading test data for M1 plots...")
X_test, y_test_onehot, y_test = load_data(
    root='./data', download=True, train=False, colored=COLORED
)
test_loader = DataLoader(TensorDataset(X_test, y_test_onehot), batch_size=BATCH_SIZE)

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        probs  = model(xb, return_logits=False)
        preds  = probs.argmax(dim=1)
        labels = yb.argmax(dim=1)
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

all_probs  = torch.cat(all_probs)
all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()
accuracy   = (all_preds == all_labels).mean()
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Training curves
checkpoint = torch.load('model_params/galaxys4-29070.pth', map_location='cpu')
history = checkpoint.get('history', None) if isinstance(checkpoint, dict) else None

if history:
    epochs = range(1, len(history['loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='#C44E52')
    ax1.plot(epochs, history['loss'], color='#C44E52', linewidth=2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='#C44E52')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='#4C72B0')
    ax2.plot(epochs, history['val_accuracy'], color='#4C72B0', linewidth=2, linestyle='--', label='Val Accuracy')
    ax2.axhline(y=0.65, color='gray', linestyle=':', linewidth=1.2, label='65% target')
    ax2.tick_params(axis='y', labelcolor='#4C72B0')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    plt.title('Training Loss and Validation Accuracy')
    fig.tight_layout()
    plt.savefig('images/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: images/training_curves.png")

# Confusion matrix
cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, ax=ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted Class')
ax.set_ylabel('True Class')
ax.set_title('Confusion Matrix (Normalised by True Class)')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: images/confusion_matrix.png")

# Per-class metrics
precision = precision_score(all_labels, all_preds, average=None)
recall    = recall_score(all_labels, all_preds, average=None)
f1        = f1_score(all_labels, all_preds, average=None)
x         = np.arange(len(CLASSES))
width     = 0.25
fig, ax   = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width, precision, width, label='Precision', color='#4C72B0')
b2 = ax.bar(x,         recall,    width, label='Recall',    color='#55A868')
b3 = ax.bar(x + width, f1,        width, label='F1-Score',  color='#C44E52')
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Score')
ax.set_title('Per-Class Precision, Recall and F1-Score')
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=15, ha='right')
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y', alpha=0.3)
macro_p, macro_r, macro_f = precision.mean(), recall.mean(), f1.mean()
ax.text(0.98, 0.02, f'Macro avg - P: {macro_p:.2f}  R: {macro_r:.2f}  F1: {macro_f:.2f}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
plt.tight_layout()
plt.savefig('images/per_class_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: images/per_class_metrics.png")

# Sample predictions
np.random.seed(RNG_SEED)
correct_idx   = np.where(all_preds == all_labels)[0]
incorrect_idx = np.where(all_preds != all_labels)[0]
incorrect_conf = all_probs[incorrect_idx].max(dim=1).values.numpy()
top_incorrect  = incorrect_idx[np.argsort(-incorrect_conf)[:3]]
selected = np.concatenate([np.random.choice(correct_idx, 6, replace=False), top_incorrect])
np.random.shuffle(selected)
selected = selected[:9]
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
for ax, idx in zip(axes.flat, selected):
    img     = X_test[idx].squeeze().numpy()
    true    = all_labels[idx]
    pred    = all_preds[idx]
    conf    = all_probs[idx][pred].item()
    correct = (true == pred)
    ax.imshow(img, cmap='magma')
    ax.axis('off')
    color = '#2ecc71' if correct else '#e74c3c'
    ax.set_title(f'True:  {CLASSES[true]}\nPred:  {CLASSES[pred]}\nConf: {conf:.1%}',
                 fontsize=8, color=color, pad=4)
plt.tight_layout()
plt.savefig('images/sample_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: images/sample_predictions.png")

print(f"\nAll plots saved to images/")
print(f"Test accuracy: {accuracy*100:.2f}%")
