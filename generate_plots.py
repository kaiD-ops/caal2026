import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("images", exist_ok=True)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11

# ── 1. Inference time vs optimization level ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
flags   = ['-O0', '-O1', '-O2', '-O3', '-Ofast']
times   = [6.211, 1.602, 1.885, 1.802, 1.610]
colors  = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
bars = ax.bar(flags, times, color=colors, edgecolor='white', linewidth=0.8)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{t:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xlabel('GCC Optimization Flag')
ax.set_ylabel('Inference Time (seconds)')
ax.set_title('Inference Time vs GCC Optimization Level\n(sample_09, x86-64, gcc 13.3.0)')
ax.set_ylim(0, 7.5)
ax.axhline(y=6.211, color='red', linestyle='--', alpha=0.3, label='-O0 baseline')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('images/timing_vs_optimization.png')
plt.close()
print("Saved: images/timing_vs_optimization.png")

# ── 2. Per-layer timing breakdown (estimated from profiling) ─────────────
fig, ax = plt.subplots(figsize=(9, 5))
layers  = ['Hilbert\nScan', 'Linear\n(uproject)', 'S4D\nLayer 1', 'GELU 1',
           'S4D\nLayer 2', 'GELU 2', 'TakeLast\nTimestep', 'FC\nHead', 'Softmax']
# S4D dominates: ~95% of time split between two S4D layers
total = 1.885
pcts  = [0.001, 0.002, 0.468, 0.001, 0.468, 0.001, 0.0001, 0.0001, 0.0001]
times_layer = [p * total for p in pcts]
colors2 = ['#95a5a6','#3498db','#e74c3c','#2ecc71','#e74c3c','#2ecc71','#9b59b6','#f39c12','#1abc9c']
bars2 = ax.bar(layers, times_layer, color=colors2, edgecolor='white', linewidth=0.8)
for bar, t in zip(bars2, times_layer):
    if t > 0.01:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{t:.3f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xlabel('Layer')
ax.set_ylabel('Estimated Time (seconds)')
ax.set_title('Per-Layer Inference Time Breakdown (-O2)\nS4D layers dominate due to O(L²·H) complexity')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('images/per_layer_timing.png')
plt.close()
print("Saved: images/per_layer_timing.png")

# ── 3. Instruction count vs optimization level ───────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
flags3  = ['-O0', '-O2', '-O3']
lines   = [1187, 1091, 1704]
instrs  = [29,   36,   38]
x = np.arange(len(flags3))
w = 0.35
b1 = ax.bar(x - w/2, lines,  w, label='Assembly lines', color='#3498db', edgecolor='white')
b2 = ax.bar(x + w/2, instrs, w, label='Multiply instructions', color='#e74c3c', edgecolor='white')
for bar, v in zip(b1, lines):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
            str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, v in zip(b2, instrs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
            str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(flags3)
ax.set_xlabel('GCC Optimization Flag')
ax.set_ylabel('Count')
ax.set_title('Assembly Lines and Multiply Instructions\nvs GCC Optimization Level')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('images/instruction_count.png')
plt.close()
print("Saved: images/instruction_count.png")

# ── 4. Memory footprint breakdown ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
labels4 = ['Model parameters\n(82.5 KB)', 'buf_proj\n(1024 KB)', 
           'buf_s4d1+s4d2\n(2048 KB)', 'Other buffers\n(32.3 KB)']
sizes4  = [82.5, 1024, 2048, 32.3]
colors4 = ['#3498db', '#e74c3c', '#e74c3c', '#95a5a6']
explode = (0.05, 0, 0.05, 0)
wedges, texts, autotexts = ax.pie(sizes4, labels=labels4, colors=colors4,
                                   explode=explode, autopct='%1.1f%%',
                                   startangle=140, pctdistance=0.75)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax.set_title('Memory Footprint Breakdown\nTotal: ~3.1 MB', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images/memory_footprint.png')
plt.close()
print("Saved: images/memory_footprint.png")

# ── 5. C vs PyTorch timing comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
impls   = ['PyTorch\n(estimated)', 'C -O0', 'C -O1', 'C -O2', 'C -O3', 'C -Ofast']
times5  = [0.4, 6.211, 1.602, 1.885, 1.802, 1.610]
colors5 = ['#27ae60','#e74c3c','#3498db','#9b59b6','#f39c12','#1abc9c']
bars5 = ax.bar(impls, times5, color=colors5, edgecolor='white', linewidth=0.8)
for bar, t in zip(bars5, times5):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f'{t:.3f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xlabel('Implementation')
ax.set_ylabel('Inference Time (seconds)')
ax.set_title('C Implementation vs PyTorch\nInference Time Comparison')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 7.5)
green_patch = mpatches.Patch(color='#27ae60', label='PyTorch (BLAS + multithreaded)')
blue_patch  = mpatches.Patch(color='#3498db', label='C naive (single-threaded)')
ax.legend(handles=[green_patch, blue_patch])
plt.tight_layout()
plt.savefig('images/c_vs_pytorch.png')
plt.close()
print("Saved: images/c_vs_pytorch.png")

print("\nAll 5 plots saved to images/")
