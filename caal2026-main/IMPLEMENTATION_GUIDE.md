# Milestone 1: S4-Based Galaxy Morphology Classification

Implementation of Structured State Space (S4) models for galaxy morphology classification using the GalaxyMNIST dataset.

## Project Status: ✅ Implementation Complete

All core components have been implemented:
- ✅ S4 Recurrent formulation (`model/s4_recurrent.py`)
- ✅ S4 Convolutional formulation (`model/s4_conv.py`)
- ✅ S4D Diagonal parameterization (`model/s4d.py`) - Modified for direct convolution
- ✅ Hilbert curve scanning (`model/hilbert.py`)
- ✅ TakeLastTimestep layer (`model/tlts.py`)
- ✅ GalaxyClassifier forward pass (`model/gclassifier.py`)
- ✅ Training notebook with visualization (`train.ipynb`)
- ✅ Numerical validation script (`validate_s4.py`)
- ✅ LaTeX report template (`report.tex`)

## Environment Setup

### Requirements
- Python 3.9+ (tested on 3.9.6)
- PyTorch 2.8.0+ (with MPS support for Apple Silicon or CUDA for NVIDIA GPUs)
- Additional dependencies: numpy, matplotlib, seaborn, einops, scikit-learn, torchinfo, h5py

### Installation

1. **Clone the repository** (if not already done):
```bash
git clone https://github.com/syedtaha22/state-space-models
cd state-space-models
```

2. **Install PyTorch**:
```bash
# For CUDA (NVIDIA GPUs):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# For CPU or Apple Silicon (MPS):
pip install torch torchvision
```

3. **Install dependencies**:
```bash
pip install numpy matplotlib seaborn einops scikit-learn torchinfo h5py tqdm
```

4. **Install GalaxyMNIST**:
```bash
pip install git+https://github.com/mwalmsley/galaxy_mnist.git@c1fe9853a00bc34b2ff082585c6bb1654d34d239
```

## Project Structure

```
state-space-models/
├── model/
│   ├── __init__.py
│   ├── s4_recurrent.py      # S4 recurrent formulation (NEW)
│   ├── s4_conv.py            # S4 convolutional formulation (NEW)
│   ├── s4d.py                # S4D diagonal (MODIFIED for direct conv)
│   ├── hilbert.py            # Hilbert curve scanning (COMPLETED)
│   ├── tlts.py               # TakeLastTimestep layer (COMPLETED)
│   ├── gclassifier.py        # Galaxy classifier (COMPLETED)
│   └── functions.py          # Utility functions
├── train.ipynb               # Training notebook (COMPLETED)
├── validate_s4.py            # S4 validation script (NEW)
├── report.tex                # LaTeX report template (NEW)
├── utils.py                  # Utility functions
├── main.py                   # CLI training script
└── README.md                 # This file

data/                         # Created automatically when training
model_params/                 # Model weights saved here after training
```

## Usage

### 1. Validate S4 Implementations

Test that Recurrent and Convolutional S4 produce equivalent outputs:

```bash
python3 validate_s4.py
```

This will:
- Test numerical equivalence (outputs should match within 1e-4)
- Benchmark timing for different sequence lengths
- Verify both implementations are working correctly

**Expected output:**
```
S4 IMPLEMENTATION VALIDATION SUITE
============================================================
Testing Equivalence: L=100, d_model=16, d_state=64
============================================================
✓ Recurrent output shape: torch.Size([2, 100, 16])
✓ Convolutional output shape: torch.Size([2, 100, 16])

Numerical Difference:
  Max absolute difference:  X.XXe-05
  Mean absolute difference: X.XXe-06
  Tolerance threshold:      1.00e-04

✅ PASSED: Outputs match within tolerance!
```

### 2. Train the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook train.ipynb
```

**Key steps in the notebook:**
1. Load GalaxyMNIST dataset (10,000 images, 4 classes)
2. Split into train/validation/test sets
3. Initialize `GalaxyClassifierS4D` model
4. Train for 10-20 epochs (adjust `EPOCHS` variable)
5. Evaluate on test set
6. Visualize results (training curves, confusion matrix)
7. Export model weights

**Important: Set your ERP ID**  
Before training, update the `RNG_SEED` variable:
```python
RNG_SEED = YOUR_ERP_ID  # Replace 42 with your actual ERP ID
```

### 3. Monitor Training

The notebook includes progress bars and will display:
- Training loss per epoch
- Validation accuracy per epoch
- Final test accuracy
- Confusion matrix showing per-class performance

**Target Performance:** ≥ 65% test accuracy

### 4. Export Model

After training, the notebook exports:
- Model parameters to `model_params/` directory
- PyTorch state dict: `model_params/galaxys4-{RNG_SEED}.pth`
- CSV format weights for C/RISC-V deployment

### 5. Complete the Report

Edit `report.tex` to document your findings:

```bash
# Edit the LaTeX file
code report.tex  # or use your preferred editor

# Compile to PDF
pdflatex report.tex
bibtex report  # if you add more references
pdflatex report.tex
pdflatex report.tex
```

**Fill in the following sections:**
- [ ] Abstract: Update with your actual test accuracy
- [ ] Section 3.2: Numerical validation results
- [ ] Section 4: Training curves (insert figures)
- [ ] Section 4.3: Confusion matrix (insert figure)
- [ ] Section 4: Fill in performance tables
- [ ] Section 5: Discussion of results
- [ ] Appendix B: Paste model summary from notebook

## Implementation Details

### Architecture

The galaxy classifier uses the following pipeline:

```
Input: (B, C, 64, 64) RGB/Grayscale Images
    ↓
Hilbert Scan: Convert to (B, 4096, C) sequence
    ↓
Linear Projection: (B, 4096, C) → (B, 4096, 64)
    ↓
S4D Layer 1 + GELU: (B, 4096, 64) → (B, 4096, 64)
    ↓
S4D Layer 2 + GELU: (B, 4096, 64) → (B, 4096, 64)
    ↓
TakeLastTimestep: (B, 4096, 64) → (B, 64)
    ↓
Fully Connected: (B, 64) → (B, 4)
    ↓
Softmax: (B, 4) probability distribution
```

### Hyperparameters

Fixed hyperparameters (as per requirements):
- `d_model = 64` (model/feature dimension)
- `d_state = 64` (state space dimension)
- `image_size = 64` (GalaxyMNIST native resolution)
- `num_classes = 4` (galaxy types)

Tunable hyperparameters:
- `batch_size = 64` (can adjust based on GPU memory)
- `learning_rate = 0.0015` (can tune for convergence)
- `epochs = 10-20` (train longer for better accuracy)

### S4 Formulations

**Recurrent (Sequential Processing):**
- Complexity: O(L·N²) per batch
- Use case: Autoregressive generation, step-by-step inference
- Implementation: `model/s4_recurrent.py`

**Convolutional (Parallel Processing):**
- Complexity: O(L²·N) direct, O(L log L·N) with FFT
- Use case: Training (faster), fixed-length sequences
- Implementation: `model/s4_conv.py`

**Diagonal (S4D) (Efficient Parameterization):**
- Complexity: O(L log L·N) parameters reduced from O(N²) to O(N)
- Use case: Production deployment, memory efficiency
- Implementation: `model/s4d.py` (modified for direct conv1d)

### Hilbert Curve

The Hilbert curve preserves spatial locality when converting 2D images to 1D sequences:

```python
# Row-major flattening (naive):
# Pixels (0,0), (0,1), ..., (0,63), (1,0), ...
# Adjacent pixels in 2D can be ~64 steps apart in 1D!

# Hilbert curve flattening:
# Follows a continuous space-filling path
# Adjacent pixels in 2D are typically 1-2 steps apart in 1D
# Better for sequence models to capture spatial structure
```

## Key Implementations

### 1. Hilbert d2xy (model/hilbert.py)
Converts 1D distance to 2D coordinates using iterative quadrant traversal:
```python
def _d2xy(self, n: int, d: int) -> tuple:
    """
    Convert Hilbert curve distance to (x, y) coordinates.
    
    Algorithm:
    - Iterate log₂(n) times
    - Determine quadrant: rx = 1 & (d//2), ry = 1 & (d^rx)
    - Apply rotation/reflection if ry==0
    - Accumulate offsets and double square size
    """
    # See implementation for full code
```

### 2. TakeLastTimestep (model/tlts.py)
Extracts the final hidden state for classification:
```python
def forward(self, x):
    # x: (B, L, D) -> (B, D)
    return x[:, -1, :]
```

### 3. GalaxyClassifier (model/gclassifier.py)
Connects all components into end-to-end model:
```python
def forward(self, x, return_logits=False):
    # 1. Hilbert scan: (B,C,64,64) -> (B,4096,C)
    # 2. Project: (B,4096,C) -> (B,4096,64)
    # 3. S4D layers with GELU activations
    # 4. Take last timestep: (B,4096,64) -> (B,64)
    # 5. Classify: (B,64) -> (B,4) logits or probs
```

## Troubleshooting

### Import Errors
```bash
# If you see "ModuleNotFoundError: No module named 'galaxy_mnist'"
pip install git+https://github.com/mwalmsley/galaxy_mnist.git

# If you see "ModuleNotFoundError: No module named 'einops'"
pip install einops
```

### GPU/CUDA Issues
```python
# Check if CUDA is available:
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check if MPS (Apple Silicon) is available:
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# If neither works, the model will fallback to CPU (slower but still works)
```

### Memory Issues
If training crashes with out-of-memory errors:
- Reduce `BATCH_SIZE` from 64 to 32 or 16
- Use grayscale images (`COLORED = False`) instead of RGB
- Use smaller `d_state` (but this changes the model architecture)

### Validation Script Errors
If `validate_s4.py` shows large differences:
- Check that both models have identical parameter initialization
- Verify discretization formulas are correctly implemented
- Try increasing tolerance to 1e-3 (some numerical error is expected)

## Testing Your Implementation

### Quick Functionality Test
```bash
# Test all imports and basic forward passes
python3 -c "
import torch
from model.s4_recurrent import S4Recurrent
from model.s4_conv import S4Convolutional
from model.gclassifier import GalaxyClassifierS4D

print('Testing S4Recurrent...')
m1 = S4Recurrent(d_model=4, d_state=16)
x = torch.randn(1, 10, 4)
y1, _ = m1(x)
print(f'✓ Output shape: {y1.shape}')

print('Testing S4Convolutional...')
m2 = S4Convolutional(d_model=4, d_state=16)
y2, _ = m2(x)
print(f'✓ Output shape: {y2.shape}')

print('Testing GalaxyClassifierS4D...')
m3 = GalaxyClassifierS4D(num_classes=4, colored=False)
x_img = torch.randn(1, 1, 64, 64)
y3 = m3(x_img)
print(f'✓ Output shape: {y3.shape}')

print('All tests passed!')
"
```

### Expected Model Performance

Based on the requirements:
- **Minimum target:** 65% test accuracy
- **Typical range:** 65-75% (depends on training epochs and hyperparameters)
- **Reference:** CNNs on GalaxyMNIST typically achieve 75-85%

If your accuracy is below 50%, check:
- Are you training for enough epochs? (Try 15-20)
- Is the learning rate too high/low? (Try 0.001 to 0.003)
- Did you normalize the images? (Should be in [0, 1] range)
- Is overfitting occurring? (Check if validation accuracy plateaus or drops)

## Citation

If you use this code, please cite the original S4 papers:

```bibtex
@inproceedings{gu2022efficiently,
  title={Efficiently Modeling Long Sequences with Structured State Spaces},
  author={Gu, Albert and Goel, Karan and R{\'e}, Christopher},
  booktitle={International Conference on Learning Representations},
  year={2022}
}

@inproceedings{gu2022parameterization,
  title={On the Parameterization and Initialization of Diagonal State Space Models},
  author={Gu, Albert and Gupta, Ankit and Goel, Karan and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## License

This project is for educational purposes as part of CS/ECE coursework.

## Authors

- Student Name: [YOUR NAME]
- ERP ID: [YOUR ERP ID]
- Course: [CS/ECE XXX]
- Semester: [SEMESTER/YEAR]

## Acknowledgments

- Original S4 implementation by Albert Gu et al.
- GalaxyMNIST dataset by Mike Walmsley et al.
- State-space-models repository by Syed Taha
