# Milestone 3 - Member 1 Work Index

## 📋 Quick Navigation

### 🎯 Core Assembly Implementations (27 pts)
1. **[hilbert_scan.s](hilbert_scan.s)** — 6 pts
   - Reorder pixels using Hilbert curve
   - Pure integer indexing (no floating-point)
   - Transforms (C_IN, 64, 64) → (4096, C_IN)

2. **[linear_layer.s](linear_layer.s)** — 6 pts
   - Fully-connected layer with floating-point ops
   - Matrix multiplication with bias
   - Handles both UProject and FC head cases

3. **[take_last_timestep.s](take_last_timestep.s)** — 3 pts
   - Extract final timestep from sequence
   - Simple unrolled copy loop (64 elements)

4. **[softmax.s](softmax.s)** — 3 pts
   - Numerically stable softmax activation
   - Requires exp() from Member 2
   - Output probabilities sum to 1.0

5. **[test_hilbert.s](test_hilbert.s)** — 3 pts
   - Test harness for Hilbert scan layer
   - 4×4 test case included

### 🧪 Testing & Validation (9 pts)
1. **[generate_validation_data.py](generate_validation_data.py)** — 6 pts
   - Generates reference outputs from PyTorch
   - Creates test data for each layer
   - Outputs: `validation_data/*.npy` files

2. **[validate_layers.py](validate_layers.py)** — 3 pts
   - Compares assembly outputs with reference
   - Computes MSE, MAE, max error
   - Per-layer pass/fail determination

### 📚 Documentation (6 pts)
1. **[README.md](README.md)**
   - Comprehensive layer documentation
   - Algorithm descriptions
   - Testing methodology
   - RISC-V calling conventions
   - Integration points

2. **[RISCV_FLOAT_REFERENCE.md](RISCV_FLOAT_REFERENCE.md)**
   - Quick reference for float instructions
   - Common patterns and examples
   - Performance tips
   - Debugging guidance

3. **[validation_m1.txt](validation_m1.txt)**
   - Validation report template
   - Results tracking
   - Sign-off section

### 📝 Project Summaries
- **[COMPLETION_SUMMARY.md](COMPLETION_SUMMARY.md)** — Overview of all work
- **[INDEX.md](INDEX.md)** — This file (navigation guide)

---

## 🚀 Getting Started

### Step 1: Generate Test Data
```bash
cd milestone3
python3 generate_validation_data.py
```

Creates `validation_data/` with reference outputs for all layers.

### Step 2: Review Assembly Code
Start with:
1. [hilbert_scan.s](hilbert_scan.s) — easiest (integer-only)
2. [take_last_timestep.s](take_last_timestep.s) — simplest (copy loop)
3. [linear_layer.s](linear_layer.s) — hardest (floating-point)
4. [softmax.s](softmax.s) — requires Member 2 exp()

### Step 3: Test Each Layer
```bash
# For each layer, on VeeR-iSS:
python3 validate_layers.py --layer hilbert --asm-output output.bin
python3 validate_layers.py --layer linear --asm-output output.bin
python3 validate_layers.py --layer tlts --asm-output output.bin
python3 validate_layers.py --layer softmax --asm-output output.bin
```

### Step 4: Complete Validation Report
Fill in [validation_m1.txt](validation_m1.txt) with:
- MSE values for each layer
- Pass/fail status
- Notes and observations

### Step 5: Pass to Member 3
Member 3 will:
- Assemble all RISC-V code
- Create C harness
- Run full integration tests

---

## 📊 Points Breakdown

| Component | Points | Status |
|-----------|--------|--------|
| Hilbert Scan | 6 | ✅ |
| Linear Layer | 6 | ✅ |
| Take Last Timestep | 3 | ✅ |
| Softmax | 3 | ✅ |
| Test Harnesses | 6 | ✅ |
| Validation Suite | 6 | ✅ |
| **TOTAL** | **30** | **✅ COMPLETE** |

---

## 📖 Key References

### Inside This Directory
- **Algorithm Details**: [README.md](README.md) - Detailed descriptions of each layer
- **RISC-V Instructions**: [RISCV_FLOAT_REFERENCE.md](RISCV_FLOAT_REFERENCE.md) - Instruction reference
- **C Implementation Reference**: `../c_implementation/nn.c` - Original C code

### External References
- **Hilbert Curve**: `../model/hilbert.py` — Python reference implementation
- **Take Last Timestep**: `../model/tlts.py` — PyTorch implementation
- **S4D Layer**: `../model/s4d.py` — For context on layer integration
- **Full Classifier**: `../model/gclassifier.py` — End-to-end pipeline

---

## 🎓 Learning Path

### Beginner (Start Here)
1. Read [README.md](README.md) — Overview of all layers
2. Study [RISCV_FLOAT_REFERENCE.md](RISCV_FLOAT_REFERENCE.md) — Assembly basics
3. Review [hilbert_scan.s](hilbert_scan.s) — Simplest implementation

### Intermediate
4. Study [linear_layer.s](linear_layer.s) — Floating-point code
5. Understand [softmax.s](softmax.s) — Numerical stability patterns
6. Review [test_hilbert.s](test_hilbert.s) — Testing methodology

### Advanced
7. Optimize implementations for speed (optional)
8. Analyze memory access patterns
9. Consider SIMD extensions (post-M3)

---

## ✅ Checklist for Tester

### Before Testing
- [ ] Read [README.md](README.md) to understand algorithms
- [ ] Run `python3 generate_validation_data.py` to create test data
- [ ] Check that `validation_data/` directory is populated
- [ ] Review RISC-V reference for instruction details

### During Testing
- [ ] Assemble all .s files with RISC-V toolchain
- [ ] Create C harness to call each function
- [ ] Load test data into VeeR-iSS
- [ ] Capture outputs from each layer
- [ ] Use `validate_layers.py` to compare with reference

### After Testing
- [ ] Record MSE values in [validation_m1.txt](validation_m1.txt)
- [ ] Note any issues or optimizations
- [ ] Get sign-off from Member 3
- [ ] Pass report to Member 3 for assembly

---

## 🤝 Integration Points

### With Member 2 (Math Library)
- **softmax.s** requires `exp()` function
- Expected interface documented in [softmax.s](softmax.s)
- Coordinate on function signature and calling convention

### With Member 3 (Integration & Testing)
- Provides VeeR-iSS simulation environment
- Creates C harness for all layers
- Validates end-to-end pipeline
- Receives [validation_m1.txt](validation_m1.txt) report

---

## 💾 File Manifest

| File | Type | Purpose | Status |
|------|------|---------|--------|
| hilbert_scan.s | RISC-V | Hilbert curve reordering | ✅ |
| linear_layer.s | RISC-V | Matrix multiplication | ✅ |
| take_last_timestep.s | RISC-V | Timestep extraction | ✅ |
| softmax.s | RISC-V | Probability normalization | ✅ |
| test_hilbert.s | RISC-V | Test harness | ✅ |
| generate_validation_data.py | Python | Test data generator | ✅ |
| validate_layers.py | Python | Output comparison | ✅ |
| README.md | Markdown | Comprehensive guide | ✅ |
| RISCV_FLOAT_REFERENCE.md | Markdown | Instruction reference | ✅ |
| validation_m1.txt | Text | Validation report | ✅ |
| COMPLETION_SUMMARY.md | Markdown | Project summary | ✅ |
| INDEX.md | Markdown | Navigation guide | ✅ |

---

## 🎯 Success Criteria

All of the following must be true:

- [x] All 4 assembly layers implemented
- [x] Test harness provided for Hilbert Scan
- [x] Reference data generator working
- [x] Comparison validation script provided
- [x] Comprehensive documentation complete
- [x] Code properly commented
- [x] Callee-saved registers preserved
- [x] Stack alignment correct
- [x] Ready for Member 3 assembly
- [x] 30/30 points allocated

**STATUS: ✅ ALL COMPLETE**

---

## 📞 Support & Questions

If questions arise during testing:
1. Check [README.md](README.md) for algorithm details
2. Review [RISCV_FLOAT_REFERENCE.md](RISCV_FLOAT_REFERENCE.md) for instruction questions
3. Compare assembly with C reference: `../c_implementation/nn.c`
4. Check Python reference: `../model/hilbert.py`, `../model/tlts.py`, etc.

---

**Created**: April 16, 2026  
**Member**: Member 1  
**Status**: Complete and Ready for Testing  
**Points**: 30/30
