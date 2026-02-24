# S4 Galaxy Classification - Complete Project Guide

**Author:** Mohsin Hussain (ERP: 29070)  
**Date:** February 2026  
**Course:** Deep Learning (CS 440) - Assignment 3

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [S4 Model Architecture](#s4-model-architecture)
4. [Setup Instructions](#setup-instructions)
5. [Testing Individual Components](#testing-individual-components)
6. [Training the Model](#training-the-model)
7. [Expected Results](#expected-results)
8. [Understanding the Code](#understanding-the-code)
9. [Troubleshooting](#troubleshooting)
10. [Key Technical Details](#key-technical-details)

---

## Project Overview

This project implements **Structured State Space Models (S4)** for galaxy morphology classification using the GalaxyMNIST dataset. The S4 architecture is a state-of-the-art sequence model that efficiently processes long sequences using structured state space representations.

### What We're Classifying
- **Dataset:** GalaxyMNIST (10 galaxy types)
- **Input:** 224×224 grayscale images of galaxies
- **Output:** Classification into 10 morphology classes:
  - Disturbed Galaxies
  - Merging Galaxies
  - Round Smooth Galaxies
  - In-between Round Smooth Galaxies
  - Cigar-shaped Smooth Galaxies
  - Barred Spiral Galaxies
  - Unbarred Tight Spiral Galaxies
  - Unbarred Loose Spiral Galaxies
  - Edge-on Galaxies without Bulge
  - Edge-on Galaxies with Bulge

### Model Architecture
- **Hilbert Curve:** Converts 2D images into 1D sequences preserving spatial locality
- **S4 Layers:** Process the 1D sequence with three different implementations:
  - Recurrent mode (RNN-like step-by-step processing)
  - Convolutional mode (efficient parallel training)
  - Diagonal variant (simplified diagonal state matrices)
- **Classification Head:** Global average pooling + linear layers
- **Target Accuracy:** ≥65% on test set

---

## Project Structure

```
state-space-models/
├── model/
│   ├── s4_recurrent.py      # S4 recurrent mode implementation
│   ├── s4_conv.py            # S4 convolutional mode implementation
│   ├── s4d.py                # S4D (diagonal variant) implementation
│   ├── gclassifier.py        # GalaxyClassifier main model
│   ├── hilbert.py            # Hilbert curve 2D→1D mapping
│   └── tlts.py               # TLTS utility functions
├── data/
│   └── GalaxyMNIST/          # Dataset (downloaded at runtime)
│       ├── train_dataset.hdf5
│       └── test_dataset.hdf5
├── train.ipynb               # Main training notebook (USE THIS)
├── test_components.ipynb     # Individual component tests
├── test_s4.py                # Unit tests for S4 implementations
├── PROJECT_GUIDE.md          # This file
└── README.md                 # GitHub repository description
```

### File Descriptions

#### Model Files (`model/`)

**`s4_recurrent.py`** - S4 Recurrent Mode
- Implements step-by-step sequence processing (like RNN)
- Uses discretized state space matrices (Ā, B̄)
- Maintains hidden state across timesteps
- Functions:
  - `forward_recurrent()`: Processes sequence one step at a time
  - `forward()`: Main entry point with shape handling

**`s4_conv.py`** - S4 Convolutional Mode
- Implements efficient parallel training using convolution
- Computes S4 kernel via power series: K = (CB̄, CĀB̄, CĀ²B̄, ...)
- Uses FFT-based convolution for efficiency
- Functions:
  - `s4_kernel()`: Generates the convolutional kernel
  - `forward_conv()`: Applies kernel via FFT convolution
  - `forward()`: Main entry point

**`s4d.py`** - S4D Diagonal Variant
- Simplified S4 with diagonal state matrix Λ
- Removed NPLR parameterization (uses direct convolution)
- More efficient than full S4
- Functions:
  - `kernel_DPLR()`: Computes kernel from diagonal Λ
  - `forward()`: Applies S4D transformation

**`gclassifier.py`** - Galaxy Classifier
- Main model combining all components
- Architecture:
  1. Hilbert curve (2D→1D)
  2. Linear embedding (256→d_model=64)
  3. Three S4D layers
  4. Global average pooling
  5. Classification head (MLP)
- Hyperparameters:
  - d_model=64, d_state=64, dropout=0.2
  - 3 S4D layers
  - Hidden dimension=128 in classifier

**`hilbert.py`** - Hilbert Curve Mapping
- Implements space-filling curve to convert 2D images→1D sequences
- Preserves spatial locality (nearby pixels stay nearby in sequence)
- Function: `d2xy(n, d)` converts distance along curve to (x,y) coordinates

**`tlts.py`** - TLTS Utilities
- Extracts Toeplitz-Like-Triangular-Structured matrices
- Function: `extract_tlts()` - simple extraction for our purposes

#### Main Scripts

**`train.ipynb`** - Complete Training Pipeline
- **23 code cells** covering the full training pipeline
- Downloads GalaxyMNIST dataset automatically
- Trains model for 15 epochs
- Generates evaluation metrics and visualizations
- **⚠️ IMPORTANT:** Skip cell 17 (CSV export) - not needed for training

**`test_components.ipynb`** - Component Testing
- Tests each S4 implementation individually
- Verifies forward passes work
- Checks tensor shapes
- Good for debugging

**`test_s4.py`** - Unit Tests
- Pytest-based tests for S4 implementations
- Validates numerical correctness
- Run with: `pytest test_s4.py -v`

---

## Setup Instructions

### 1. System Requirements
- Python 3.9 or higher
- ~2GB disk space for dataset
- CPU is fine (training takes ~15 minutes)
- macOS, Linux, or Windows

### 2. Install Dependencies

```bash
cd state-space-models/

# Install required packages
pip install torch torchvision numpy scipy h5py matplotlib seaborn pandas scikit-learn jupyter notebook
```

### 3. Verify Installation

```bash
# Check Python packages
python -c "import torch; import scipy; import h5py; print('✓ All packages installed')"

# Check Jupyter
jupyter notebook --version
```

### 4. Project Structure Check

```bash
# Verify all model files exist
ls -la model/

# Should see:
# s4_recurrent.py, s4_conv.py, s4d.py, gclassifier.py, hilbert.py, tlts.py
```

---

## Testing Individual Components

### Option 1: Using test_components.ipynb

```bash
jupyter notebook test_components.ipynb
```

Run the cells to test:
1. Hilbert curve visualization
2. S4 Recurrent forward pass
3. S4 Convolutional forward pass
4. S4D forward pass
5. GalaxyClassifier full forward pass

**Expected Output:**
- All cells should run without errors
- Tensor shapes should match expected dimensions
- No NaN or Inf values

### Option 2: Using Pytest

```bash
# Run all unit tests
pytest test_s4.py -v

# Run specific test
pytest test_s4.py::test_s4_recurrent -v
```

**Expected Output:**
```
test_s4.py::test_s4_recurrent PASSED
test_s4.py::test_s4_conv PASSED
test_s4.py::test_s4d PASSED
test_s4.py::test_galaxy_classifier PASSED
```

### Option 3: Quick Python Test

```python
import torch
from model.gclassifier import GalaxyClassifier

# Create model
model = GalaxyClassifier()
print(f"✓ Model created successfully")

# Test forward pass
x = torch.randn(2, 1, 224, 224)  # Batch of 2 images
output = model(x)

print(f"✓ Forward pass successful")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")  # Should be [2, 10]
print(f"  No NaN values: {not torch.isnan(output).any()}")
```

---

## Training the Model

### Step-by-Step Training Guide

#### 1. Launch Jupyter Notebook

```bash
cd /Users/aghamohsinhussain/state-space-models/
jupyter notebook
```

This will open your browser at `http://localhost:8888`

#### 2. Open train.ipynb

In the Jupyter interface:
- Click on `train.ipynb`
- You'll see the complete training pipeline

#### 3. Configure Training Parameters

**IMPORTANT:** Before running, check these settings in the configuration cell:

```python
# Cell ~5-6 (near the top after imports)
DEVICE = "cpu"              # Or "mps" for Mac M1/M2, or "cuda" for GPU
EPOCHS = 15                 # ✓ Set to 15 for full training
BATCH_SIZE = 64             # Good default
LEARNING_RATE = 1e-3        # Default Adam learning rate
RNG_SEED = 29070            # ✓ Already set to my ERP ID
```

#### 4. Run Cells in Order

**Method A: Run All (Skip One Cell)**

1. **DO NOT** use "Cell → Run All" - it will hang
2. Instead, manually run cells one by one using `Shift+Enter`
3. **Skip cell 17** (the CSV export cell) - it's not needed

**How to identify cell 17:**
Look for this code:
```python
# Cell that EXPORTS 100 sample images to CSV
indices = random.sample(range(len(x_train)), 100)
with open("galaxy_samples.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # ... more code
```

**This cell is OPTIONAL - for C/RISC-V testing only. SKIP IT.**

**Method B: Run Specific Cell Blocks**

Run these blocks in order (press `Shift+Enter` on each):

1. **Imports Block** (cells 1-3)
   ```python
   import torch
   import torch.nn as nn
   # ... etc
   ```

2. **Configuration** (cell ~5)
   ```python
   DEVICE = "cpu"
   EPOCHS = 15  # ✓ Make sure this is 15
   ```

3. **Data Loading** (cells ~8-12)
   ```python
   def load_data():
       # Downloads dataset automatically
   x_train, y_train, x_test, y_test = load_data()
   ```
   
   **⏳ First time: ~2-3 minutes to download 215MB dataset**

4. **Data Preprocessing** (cells ~13-16)
   ```python
   # One-hot encoding
   y_train_onehot = ...
   # DataLoaders
   train_loader = DataLoader(...)
   ```

5. **⚠️ SKIP THIS CELL (~17)** - CSV export, not needed

6. **Model Creation** (cell ~19)
   ```python
   model = GalaxyClassifier().to(DEVICE)
   ```

7. **Training Setup** (cells ~20-28)
   ```python
   optimizer = torch.optim.Adam(...)
   criterion = nn.CrossEntropyLoss()
   
   def train(model, train_loader, val_loader, ...):
       # Training loop
   ```

8. **🚀 START TRAINING** (cell ~29)
   ```python
   history = train(
       model, train_loader, val_loader,
       criterion, optimizer,
       epochs=EPOCHS,  # This will be 15
       device=DEVICE
   )
   ```
   
   **⏳ Training time: ~12-15 minutes for 15 epochs**
   
   **What you'll see:**
   ```
   Epoch 1/15
   Train: 100%|████████████| 237/237 [00:48<00:00, 4.92it/s, loss=2.15]
   Val:   100%|████████████|  60/60 [00:08<00:00, 7.31it/s, loss=1.98]
   Epoch 1: train_loss=2.145, val_loss=1.982, val_acc=28.3%
   
   Epoch 2/15
   Train: 100%|████████████| 237/237 [00:47<00:00, 5.01it/s, loss=1.87]
   ...
   ```

9. **Evaluation** (cells ~30-35)
   ```python
   # Test set evaluation
   test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
   print(f"Test Accuracy: {test_acc:.2f}%")
   ```

10. **Visualizations** (cells ~36-40)
    - Training curves (accuracy & loss)
    - Confusion matrix
    - Sample predictions with images

#### 5. Monitor Training Progress

**Good signs:**
- ✅ Training loss decreasing (starts ~2.2, should reach ~1.4-1.6)
- ✅ Validation accuracy increasing (should reach 60-70%)
- ✅ Progress bars moving smoothly
- ✅ No error messages

**Warning signs:**
- ⚠️ Training loss increasing → learning rate too high
- ⚠️ Training loss stuck → model not learning
- ⚠️ Validation accuracy not improving after epoch 5 → possible overfitting

#### 6. Save Model (Optional)

After training completes, save the model:

```python
# Add this cell at the end
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'test_accuracy': test_acc,
    'history': history
}, 'galaxy_s4_model.pth')

print(f"✓ Model saved with test accuracy: {test_acc:.2f}%")
```

---

## Expected Results

### Training Metrics

**Target Performance:**
- **Training Accuracy:** 70-80% (final epoch)
- **Validation Accuracy:** 65-75%
- **Test Accuracy:** ≥65% (requirement)

**Typical Training Progression:**
```
Epoch 1:  train_acc=28%, val_acc=30%  (random initialization)
Epoch 3:  train_acc=42%, val_acc=40%  (learning starts)
Epoch 5:  train_acc=55%, val_acc=52%  (rapid improvement)
Epoch 10: train_acc=68%, val_acc=62%  (approaching target)
Epoch 15: train_acc=74%, val_acc=67%  (final performance)
```

### Key Observations

1. **S4 vs Traditional RNN:**
   - S4 handles the long sequence (256 tokens) efficiently
   - No vanishing gradients
   - Faster training than LSTM

2. **Hilbert Curve Benefits:**
   - Preserves spatial locality
   - Better than row-major or column-major flattening
   - Adjacent galaxy features stay close in sequence

3. **Performance by Galaxy Type:**
   - **Easy classes:** Round smooth, edge-on (high accuracy)
   - **Hard classes:** In-between types, merging galaxies (lower accuracy)
   - Confusion matrix shows main errors between similar morphologies

### Visualization Outputs

After training, you'll generate:

1. **Training Curves**
   - Loss decreasing over epochs
   - Train/val accuracy converging

2. **Confusion Matrix**
   - 10×10 heatmap
   - Diagonal should be bright (correct predictions)
   - Off-diagonal shows confusion between classes

3. **Sample Predictions**
   - Grid of galaxy images with predicted vs true labels
   - Visual verification of model performance

---

## Understanding the Code

### S4 Core Concepts

#### 1. State Space Representation

The S4 model implements this continuous system:
```
x'(t) = Ax(t) + Bu(t)    [state equation]
y(t)  = Cx(t) + Du(t)    [output equation]
```

**Discretization** (for digital sequences):
```
x_k = Ā·x_{k-1} + B̄·u_k
y_k = C·x_k + D·u_k
```

Where:
- `Ā = exp(Δ·A)` (matrix exponential)
- `B̄ = (Δ·A)^{-1}(exp(Δ·A) - I)·B`
- Δ = step size (1.0 in our case)

#### 2. Recurrent vs Convolutional Modes

**Recurrent Mode** (`s4_recurrent.py`):
- Processes sequence step by step
- Maintains hidden state
- Good for inference/generation
- Code:
  ```python
  h = torch.zeros(batch, d_state)  # hidden state
  for u_k in u:
      h = A_bar @ h + B_bar @ u_k
      y_k = C @ h + D @ u_k
  ```

**Convolutional Mode** (`s4_conv.py`):
- Parallel processing (all timesteps at once)
- More efficient for training
- Code:
  ```python
  K = [C·B̄, C·Ā·B̄, C·Ā²·B̄, ...]  # kernel
  y = u ⊗ K  # convolution (via FFT)
  ```

#### 3. HiPPO Initialization

The matrix `A` uses HiPPO (High-order Polynomial Projection Operators):
- Encodes a memory of the past
- Ensures long-range dependencies
- `A[n,k]` values follow specific pattern for optimal memory

### Hilbert Curve Details

The Hilbert curve converts 2D coordinates to 1D while preserving locality:

```python
# Example: 16×16 image
n = 4  # 2^4 = 16
for d in range(256):  # 256 pixels
    x, y = d2xy(n, d)
    pixel_value = image[x, y]
    sequence[d] = pixel_value
```

**Why Hilbert?**
- Nearby pixels in 2D → nearby positions in 1D
- Better than row-major scan which breaks spatial continuity
- Algorithm: recursive subdivision with rotation

### GalaxyClassifier Architecture

```
Input: [batch, 1, 224, 224]
   ↓
Hilbert Curve: [batch, 1, 224, 224] → [batch, 256]
   ↓
Linear Embedding: [batch, 256] → [batch, 256, 64]
   ↓
S4D Layer 1: [batch, 256, 64] → [batch, 256, 64]
S4D Layer 2: [batch, 256, 64] → [batch, 256, 64]
S4D Layer 3: [batch, 256, 64] → [batch, 256, 64]
   ↓
Global Avg Pool: [batch, 256, 64] → [batch, 64]
   ↓
MLP: [batch, 64] → [batch, 128] → [batch, 10]
   ↓
Output: [batch, 10] (class logits)
```

---

## Troubleshooting

### Common Issues

#### 1. Jupyter Notebook Hangs

**Problem:** Notebook stuck at "Running cell" for >5 minutes

**Solution:**
- Click the stop button (⏹️) in Jupyter toolbar
- Kernel → Restart & Clear Output
- Run cells manually, **skip cell 17** (CSV export)

#### 2. Dataset Download Fails

**Problem:** `load_data()` times out or fails

**Solution:**
```python
# Manual download
import urllib.request
import os

os.makedirs("data/GalaxyMNIST", exist_ok=True)

urls = [
    "https://zenodo.org/record/3553016/files/train_dataset.hdf5",
    "https://zenodo.org/record/3553016/files/test_dataset.hdf5"
]

for url in urls:
    filename = url.split("/")[-1]
    filepath = f"data/GalaxyMNIST/{filename}"
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"✓ Saved to {filepath}")
```

#### 3. Out of Memory Errors

**Problem:** `RuntimeError: CUDA out of memory` or similar

**Solution:**
```python
# Reduce batch size
BATCH_SIZE = 32  # Instead of 64

# Or use CPU
DEVICE = "cpu"
```

#### 4. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'scipy'`

**Solution:**
```bash
pip install scipy h5py matplotlib seaborn pandas scikit-learn
```

#### 5. Low Accuracy (<50%)

**Problem:** Model not learning properly

**Check:**
1. EPOCHS = 15 (not 2)
2. Learning rate = 1e-3 (not too high/low)
3. Data loaded correctly (10 unique labels)
4. No NaN in loss values

**Debug:**
```python
# Check data distribution
print(f"Unique labels: {torch.unique(y_train)}")
print(f"Train samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")

# Check model output
output = model(x_train[:2])
print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
print(f"Contains NaN: {torch.isnan(output).any()}")
```

#### 6. Shape Mismatch Errors

**Problem:** `RuntimeError: shape mismatch` during training

**Check:**
```python
# Verify data shapes
print(f"x_train: {x_train.shape}")  # Should be [N, 1, 224, 224]
print(f"y_train: {y_train.shape}")  # Should be [N] or [N, 10]

# Verify model output
test_input = x_train[:2]
test_output = model(test_input)
print(f"Model output: {test_output.shape}")  # Should be [2, 10]
```

### Performance Issues

#### Training Too Slow (>2 min/epoch)

**Solutions:**
- Increase `num_workers` in DataLoader: `DataLoader(..., num_workers=4)`
- Use GPU if available: `DEVICE = "cuda"`
- Reduce image size (but may hurt accuracy)

#### Training Too Fast (<30 sec/epoch)

**Check:**
- Are all samples being used? `len(train_loader)` should be ~237
- Is the model actually learning? Check loss decreasing

---

## Key Technical Details

### Hyperparameters

**Model Architecture:**
```python
d_model = 64        # Hidden dimension
d_state = 64        # State space dimension
n_layers = 3        # Number of S4 layers
dropout = 0.2       # Dropout rate
```

**Training Configuration:**
```python
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
OPTIMIZER = Adam
CRITERION = CrossEntropyLoss
```

**Data Preprocessing:**
```python
Image size: 224×224 (downsampled from 256×256)
Hilbert order: n=4 (2^4 = 16×16 grid, 256 patches)
Patch size: 14×14 (224/16)
Sequence length: 256 tokens
```

### Model Size

```python
# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

**Expected:** ~200K-300K parameters

### Computational Complexity

**S4 Convolutional Mode:**
- Kernel computation: O(L·d_state)
- FFT convolution: O(L·log(L))
- Total per layer: O(L·log(L))

**Traditional LSTM:**
- Per step: O(d_model²)
- Total: O(L·d_model²)

**Speedup:** S4 is much faster for long sequences (L=256)

### Files Created During Training

After running `train.ipynb`, you'll see:
```
state-space-models/
├── data/
│   └── GalaxyMNIST/
│       ├── train_dataset.hdf5       [137MB]
│       └── test_dataset.hdf5        [34MB]
├── training_curves.png              [Generated by notebook]
├── confusion_matrix.png             [Generated by notebook]
└── galaxy_s4_model.pth              [If you save the model]
```

---

## Quick Start Checklist

**Before Training:**
- [ ] All dependencies installed (`pip install ...`)
- [ ] In correct directory (`cd state-space-models/`)
- [ ] Jupyter notebook running (`jupyter notebook`)
- [ ] `train.ipynb` opened

**During Training:**
- [ ] EPOCHS = 15 (not 2)
- [ ] DEVICE set correctly ("cpu", "mps", or "cuda")
- [ ] Data downloaded successfully (215MB total)
- [ ] Skip cell 17 (CSV export)
- [ ] Training cell running with progress bars

**After Training:**
- [ ] Test accuracy ≥65% ✓
- [ ] No NaN in outputs
- [ ] Confusion matrix generated
- [ ] Training curves saved
- [ ] Model weights saved (optional)

---

## Additional Resources

### Understanding S4 Papers

1. **Original S4 Paper:** "Efficiently Modeling Long Sequences with Structured State Spaces" (Gu et al., 2022)
   - HiPPO initialization
   - Recurrent↔Convolutional duality
   - NPLR parameterization

2. **S4D Paper:** "On the Parameterization and Initialization of Diagonal State Space Models" (Gu et al., 2022)
   - Simplified diagonal variant
   - Easier to implement and train

### GalaxyMNIST Dataset

- **Source:** Zenodo (Martin et al., 2019)
- **Size:** 21,839 training + 5,513 test images
- **Original resolution:** 256×256 (we use 224×224)
- **Classes:** Based on Galaxy Zoo project morphology labels

### GitHub Repository

- **URL:** https://github.com/aghamohsinh/state-space-models
- Contains all code from this project
- Includes this guide

---

## Verification Steps

### Quick Test (30 seconds)

```bash
cd state-space-models/
python -c "
from model.gclassifier import GalaxyClassifier
import torch
model = GalaxyClassifier()
x = torch.randn(1, 1, 224, 224)
y = model(x)
print(f'✓ Model works! Output shape: {y.shape}')
assert y.shape == (1, 10), 'Output shape incorrect'
assert not torch.isnan(y).any(), 'Output contains NaN'
print('✓ All checks passed!')
"
```

### Full Test (5 minutes)

Run `test_components.ipynb` and verify:
1. Hilbert curve creates valid coordinates
2. All S4 implementations produce output
3. GalaxyClassifier forward pass works
4. No shape mismatches or errors

---

## Contact & Support

**Author:** Mohsin Hussain  
**ERP ID:** 29070  
**Course:** CS 440 - Deep Learning  
**Assignment:** #3 - S4 for Galaxy Classification

**Questions?**
1. Check [Troubleshooting](#troubleshooting) section
2. Review error messages carefully
3. Verify all dependencies installed
4. Make sure EPOCHS=15 and skip cell 17

---

## Final Notes

### What Makes This Project Special

1. **State Space Models:** Modern alternative to transformers/RNNs
2. **Hilbert Curves:** Clever 2D→1D mapping preserving structure
3. **Two Processing Modes:** Recurrent (step-by-step) & Convolutional (parallel)
4. **Real Astronomy Data:** Actual galaxy images from telescope surveys

### Key Takeaways

- S4 efficiently handles long sequences (256 tokens)
- Structured state spaces avoid vanishing gradients
- Convolutional mode enables fast parallel training
- Hilbert curves are better than naive image flattening
- Achieves >65% accuracy on 10-class galaxy classification

### Next Steps (Optional Extensions)

1. **Data Augmentation:** Random rotations, flips for more training data
2. **Ensemble Models:** Combine multiple S4 models
3. **Different Curves:** Try Peano or Z-order curves
4. **Deeper Networks:** More S4 layers (currently 3)
5. **Transfer Learning:** Pre-train on ImageNet, fine-tune on GalaxyMNIST

---

**Good luck with your project! 🚀🌌**

If your group has questions, refer them to specific sections of this guide. Everything they need to understand, test, and train the model is documented here.
