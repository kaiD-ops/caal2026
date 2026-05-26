# Repository Structure

## Overview
This repository contains a machine learning project focused on S4 (Selective State Space Sequential) models with implementations in Python, C, and RISC-V assembly.

## Root Level Files

### Documentation
- `README.md` - Main project documentation
- `README1.md` - Additional documentation
- `IMPLEMENTATION_GUIDE.md` - Guide for implementation details
- `report.tex` - LaTeX report document
- `m2_echotheory.tex` - LaTeX document on echo theory

### Core Python Scripts
- `main.py` - Main entry point for the project
- `train.ipynb` - Jupyter notebook for training
- `train_standalone.py` - Standalone training script
- `retrain.py` - Script for retraining models
- `run_test.py` - Test execution script
- `utils.py` - Utility functions
- `monitor.py` - Monitoring utilities
- `status.py` - Status reporting

### Data & Analysis
- `generate_test_data.py` - Generate test datasets
- `generate_plots.py` - Plot generation script
- `generate_plots1.py` - Alternative plot generation
- `export_weights.py` - Export model weights
- `validate_s4.py` - S4 model validation
- `validate_c_inference.py` - C implementation validation

### Testing & Benchmarking
- `benchmark.c` - C benchmark code
- `run_benchmark.sh` - Benchmark execution script
- `benchmark_results.txt` - Benchmark results
- `validation_results.txt` - Validation results
- `requirements.txt` - Python dependencies

## Directories

### `/model` - Python Model Implementation
Contains the core machine learning model implementation:
- `__init__.py` - Package initialization
- `s4d.py` - S4D model implementation
- `s4_conv.py` - S4 convolution layer
- `s4_recurrent.py` - S4 recurrent layer
- `functions.py` - Core functions
- `gclassifier.py` - Galaxy classifier
- `hilbert.py` - Hilbert curve utilities
- `interface.py` - Interface definitions
- `gui.py` - GUI components
- `tlts.py` - TLTS (Time-Length Time-Series) utilities

### `/model_params` - Model Parameters & Weights
- `galaxys4-29070.pth` - Trained model checkpoint
- `weights.txt` - Model weights in text format

### `/c_implementation` - C Implementation
Optimized C implementation of the model:
- `main.c` - C entry point
- `nn.c` - Neural network implementation
- `nn.h` - Neural network headers
- `test.c` - C tests
- `Makefile` - Build configuration
- `galaxy_bench_O0`, `galaxy_bench_O1`, `galaxy_bench_O2`, `galaxy_bench_O3`, `galaxy_bench_Ofast` - Compiled binaries with different optimization levels
- `nn_O0.s`, `nn_O2.s`, `nn_O3.s` - Assembly output for different optimization levels

### `/riscv` - RISC-V Assembly Implementation
RISC-V ISA implementation of the model:
- `main.s` - Main assembly file
- `s4d.s` - S4D layer assembly
- `linear.s` - Linear layer assembly
- `softmax.s` - Softmax assembly
- `gelu.s` - GELU activation assembly
- `hilbert.s` - Hilbert curve assembly
- `math.s` - Math operations assembly
- `take_last.s` - Take-last operation assembly
- `layers.s` - Layer definitions
- `build.sh` - Build script
- `Makefile` - Build configuration
- `validate_riscv.py` - RISC-V implementation validation
- `static_instruction_count.txt` - Static instruction metrics
- `dynamic_instruction_count.txt` - Dynamic instruction metrics

#### `/riscv/build` - Build Artifacts
- `asm/` - Assembled output
  - `main.s`, `math.s`
- `dis/` - Disassembled code
  - `main.dis`, `math.dis`
- `exe/` - Executable files
- `hex/` - Hex files
- `logs/` - Build logs
  - `math.txt`

#### `/riscv/tests` - Test Assembly Files
- `test_s4d_full.s` - Full S4D tests
- `test_s4d_layers.s` - S4D layer tests
- `test_s4d_small.s` - Small S4D tests
- `test_s4d_fixed.s` - Fixed S4D tests
- `test_gelu.s` - GELU tests
- `test_gelu_small.s` - Small GELU tests
- `test_linear.s` - Linear layer tests
- `test_softmax.s` - Softmax tests
- `test_hilbert.s` - Hilbert tests
- `test_math.s` - Math operation tests
- `test_take_last.s` - Take-last tests
- `benchmark.s` - Benchmark tests
- `read_s4_hex.s`, `read_s4_hex2.s`, `read_s4_hex2_v2.s` - S4 hex reading tests

#### `/riscv/veer` - VeeR Processor Configuration
- `link.ld` - Linker script
- `whisper.json` - VeeR configuration

### `/milestone4` - Milestone 4: Vectorized Implementation
Vectorized assembly implementations:
- `main_vec.s` - Vectorized main
- `s4d_vec.s` - Vectorized S4D
- `linear_vec.s` - Vectorized linear layer
- `softmax_vec.s` - Vectorized softmax
- `gelu_vec.s` - Vectorized GELU
- `hilbert_vec.s` - Vectorized Hilbert
- `math_vec.s` - Vectorized math operations
- `take_last_vec.s` - Vectorized take-last
- `build_vec.sh` - Build script for vectorized code
- `count_instructions.py` - Instruction counting utility
- `validate_vec.py` - Vectorized implementation validation
- `README_m4.md` - Milestone 4 documentation

### `/test_data` - Test Datasets
Sample test data with labels:
- `metadata.txt` - Metadata file
- `sample_00_label.txt` through `sample_11_label.txt` - 12 labeled samples

### `/images` - Image Assets
(Directory for storing images/figures)

## Project Workflow

1. **Model Development** (`/model`) - Python implementation of S4D model
2. **Training** (`train.py`, `train_standalone.py`) - Train the model
3. **Optimization** - Create optimized implementations:
   - C implementation (`/c_implementation`)
   - RISC-V implementation (`/riscv`)
   - Vectorized implementation (`/milestone4`)
4. **Validation** - Verify implementations match Python version
5. **Benchmarking** - Performance metrics and comparisons

## Key Technologies

- **Python**: Model definition, training, and validation
- **C**: Optimized CPU implementation with multiple O-levels
- **RISC-V Assembly**: Low-level RISC-V ISA implementation
- **LaTeX**: Documentation and reports
- **Jupyter Notebooks**: Interactive development and exploration
