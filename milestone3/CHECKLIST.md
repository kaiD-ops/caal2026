# Milestone 3 Completion Checklist

Use this checklist to track progress toward submission on **April 13, 2026, 11:55 PM**.

## Task 0: Address Feedback from Previous Milestones (5 pts)

- [ ] Review feedback on Milestones 1-2
- [ ] Correct identified errors in prior reports
- [ ] Fix any bugs in Milestone 1-2 code
- [ ] Update README files
- [ ] Clean commit history
- [ ] Add summary of changes to report

**Status**: ⏳ Not Started
**Required for**: Starting Task 1

---

## Task 1: RISC-V Assembly Implementation (50 pts)

### Hilbert Scan (5 pts)
- [x] Create hilbert_scan.s file
- [x] Implement triple-nested loop logic
- [x] Handle Hilbert index lookup
- [x] Proper float load/store operations
- [ ] Validate against Python reference (MSE < 1e-12)
- [ ] Add to main.s orchestration
- [x] Document in IMPLEMENTATION_GUIDE.md

**Status**: ✅ Implementation Complete, ⏳ Validation Pending

### Linear Layer (6 pts)
- [x] Create linear_layer.s file
- [x] Implement 3-nested loop structure
- [x] Use fmadd.s for efficiency
- [x] Handle variable dimensions
- [x] Add test_linear.s with example
- [ ] Validate against Python reference (MSE < 1e-8)
- [ ] Add to main.s orchestration
- [x] Document in IMPLEMENTATION_GUIDE.md

**Status**: ✅ Implementation Complete, ⏳ Validation Pending

### S4D Layer (15 pts)
- [x] Create s4d_layer.s file
- [ ] Implement state initialization
- [ ] Implement discrete matrix exponential
- [ ] Implement complex number arithmetic
- [ ] Implement state update loop
- [ ] Implement output readout
- [ ] Add test_s4d.s test harness
- [ ] Validate against Python reference (MSE < 1e-7)
- [ ] Add to main.s orchestration

**Status**: ⚠️ Placeholder, 🔄 In Progress (HIGH PRIORITY)

### GELU Activation (3 pts)
- [x] Create gelu_inplace.s file
- [x] Implement tanh-based approximation
- [x] In-place modification logic
- [x] Call tanhf_fast from math.s
- [ ] Validate against Python reference (MSE < 1e-7)
- [ ] Add to main.s orchestration
- [x] Document in IMPLEMENTATION_GUIDE.md

**Status**: ✅ Implementation Complete, ⏳ Validation Pending

### Softmax Activation (3 pts)
- [x] Create softmax_inplace.s file
- [x] Implement 3-phase algorithm (max, exp, normalize)
- [x] Numerical stability (subtract max)
- [x] Call expf_fast from math.s
- [ ] Validate against Python reference (MSE < 1e-8)
- [ ] Add to main.s orchestration
- [x] Document in IMPLEMENTATION_GUIDE.md

**Status**: ✅ Implementation Complete, ⏳ Validation Pending

### Take Last Timestep (3 pts)
- [x] Create take_last_timestep.s file
- [x] Implement offset calculation
- [x] Implement copy loop (64 iterations)
- [x] Add test case
- [ ] Validate against Python reference (MSE < 1e-12)
- [ ] Add to main.s orchestration
- [x] Document in IMPLEMENTATION_GUIDE.md

**Status**: ✅ Implementation Complete, ⏳ Validation Pending

### End-to-End Forward Pass (10 pts)
- [ ] Complete main.s orchestration
- [ ] Load weights/parameters
- [ ] Call all 9 layers in sequence
- [ ] Generate output probabilities
- [ ] End-to-end validation vs Python (100% class accuracy)

**Status**: 🔄 In Progress (Depends on S4D + main completion)

### Demo Program (5 pts)
- [x] Create skeletal main.s
- [ ] Implement full program logic
- [ ] Parameter loading
- [ ] I/O integration with VeeR-iSS
- [ ] Console output formatting

**Status**: 🔄 In Progress

### Supporting Math Library
- [x] Create math.s file
- [x] Implement expf_fast (6th order Taylor)
- [x] Implement sinf_fast (Taylor)
- [x] Implement cosf_fast (Taylor)
- [x] Implement tanhf_fast (Taylor)
- [x] Implement sqrtf_fast (hardware)
- [ ] Validate approximation accuracy

**Status**: ✅ Implementation Complete

**Task 1 Total**: 41/50 pts (82%)

---

## Task 2: Testing and Validation (10 pts)

- [ ] Generate reference outputs for 10+ test samples
- [ ] Run RISC-V binary through VeeR-iSS simulator
- [ ] Compare outputs using MSE/MAE metrics
- [ ] Verify Hilbert Scan (MSE < 1e-12)
- [ ] Verify Linear Layers (MSE < 1e-8, MAE < 1e-6)
- [ ] Verify S4D Layers (MSE < 1e-7, MAE < 1e-4)
- [ ] Verify GELU (MSE < 1e-7, MAE < 1e-4)
- [ ] Verify Softmax (MSE < 1e-8, MAE < 1e-4)
- [ ] Verify Take Last (MSE < 1e-12)
- [ ] Verify End-to-End (100% class accuracy)
- [ ] Create validation results table
- [ ] Document any discrepancies

**Status**: 📋 Not Started (Framework ready, awaits assembly completion)
**Estimated Time**: 3-4 hours
**Blocker**: S4D layer & main program completion

---

## Task 3: Instruction Count and Benchmarking (15 pts)

### Static Instruction Count (5 pts)
- [ ] Build binary with disassembly
- [ ] Run instruction counter script
- [ ] Count each instruction family:
  - [ ] R-type (add, sub, mul, etc.)
  - [ ] I-type (addi, lw, jalr, etc.)
  - [ ] S-type (sw, sh, sb)
  - [ ] B-type (beq, bne, blt, etc.)
  - [ ] U-type (lui, auipc)
  - [ ] J-type (jal)
  - [ ] F-type (fadd.s, fmul.s, etc.)
- [ ] Calculate percentages for each family
- [ ] Produce summary table (per module and total)

**Status**: 📋 Not Started
**Estimated Time**: 2-3 hours
**Blocker**: Assembly completion

### Dynamic Instruction Count (10 pts)
- [ ] Configure VeeR-iSS for execution trace logging
- [ ] Run simulator on test samples
- [ ] Parse VeeR-iSS output log
- [ ] Count executed instructions by family
- [ ] Analyze per-layer execution counts
- [ ] Identify hotspot layers
- [ ] Discuss instruction family distribution
- [ ] Analyze branch frequency
- [ ] Analyze memory traffic patterns
- [ ] Document results in tables

**Status**: 📋 Not Started
**Estimated Time**: 3-4 hours
**Blocker**: Assembly completion, VeeR-iSS logs

### Execution Time Estimate (Bonus, TBD)
- [ ] Obtain CPI (cycles per instruction) estimate
- [ ] Select operating frequency (e.g., 1 GHz)
- [ ] Calculate: T = N_instructions × CPI × (1/f)
- [ ] Compare with Milestone 2 C implementation timing
- [ ] Discuss factors affecting performance

**Status**: 📋 Optional Bonus
**Estimated Time**: 1-2 hours

**Task 3 Total**: 0/15 pts (0%)

---

## Task 4: Python-RISC-V Interface Integration (Bonus, TBD)

- [ ] Set up QEMU environment
- [ ] Implement shared memory communication
- [ ] Add implementation='riscv' path to ModelInterface
- [ ] Test GUI with RISC-V backend
- [ ] Validate end-to-end with visual interface

**Status**: 📋 Optional Bonus
**Estimated Time**: 4-6 hours

---

## Task 5: Report (20 pts)

### Document Structure
- [ ] Create LaTeX document from template
- [ ] Use IEEE or DH Lab template
- [ ] Set up proper bibliography

### Required Sections

1. **Abstract** (1 page)
   - [ ] Summarize objectives
   - [ ] State approach
   - [ ] Highlight key results

2. **Introduction** (1-2 pages)
   - [ ] Describe scope of Milestone 3
   - [ ] Connection to Milestones 1-2
   - [ ] Learning outcomes achieved

3. **Feedback and Improvements** (0.5-1 page)
   - [ ] Summarize feedback from M1 & M2
   - [ ] Describe changes/corrections made
   - [ ] Cover both report and codebase

4. **Implementation** (3-5 pages)
   - [ ] Code organization (file structure)
   - [ ] Layer-by-layer design decisions
   - [ ] Memory layout explanation
   - [ ] Register allocation strategy
   - [ ] Annotated code excerpts for each layer
   - [ ] Math function descriptions + accuracy charts
   - [ ] Complexity analysis

5. **Numerical Validation** (2-3 pages)
   - [ ] Per-layer error metrics table
   - [ ] End-to-end accuracy results
   - [ ] Discuss any discrepancies
   - [ ] Validation results for all 10+ test samples

6. **Instruction Count Analysis** (2-3 pages)
   - [ ] Static count table (total + per family)
   - [ ] Dynamic count table (total + per family)
   - [ ] Per-layer dynamic count breakdown
   - [ ] Discussion of hotspot layers
   - [ ] Instruction family relationship to algorithms
   - [ ] Branch frequency analysis
   - [ ] If attempted: execution time estimation

7. **Python-RISC-V Interface** (1-2 pages, if bonus attempted)
   - [ ] Describe QEMU integration
   - [ ] Shared memory setup
   - [ ] GUI integration with RISC-V backend
   - [ ] End-to-end validation results

8. **Conclusion** (1-2 pages)
   - [ ] Summary of results achieved
   - [ ] Hotspot identification for Milestone 4
   - [ ] Reflection on team workflow
   - [ ] Lessons learned
   - [ ] Implications for vectorization (M4)

9. **Task Allocation**
   - [ ] Table showing work distribution per team member

10. **References**
    - [ ] RISC-V ISA Specification
    - [ ] VeeR-iSS documentation
    - [ ] Calling convention spec
    - [ ] Any academic papers referenced
    - [ ] External resources (LLMs, etc.)

### Formatting Requirements
- [ ] Use recommended LaTeX template
- [ ] All figures and tables have captions
- [ ] All figures/tables referenced in text
- [ ] Proper citations for external resources
- [ ] Link to GitHub repository included
- [ ] Consistent formatting and style
- [ ] Proofread for grammar and clarity

### Final Output
- [ ] Export as PDF
- [ ] Filename: `m3_<teamname>.pdf`
- [ ] File size < 20 MB
- [ ] All links functional
- [ ] All figures visible/readable

**Status**: 📋 Not Started
**Estimated Time**: 6-8 hours
**Blocker**: Implementation, testing, and analysis completion

**Task 5 Total**: 0/20 pts (0%)

---

## File Naming and Submission (5 pts)

- [ ] Verify GitHub repository is up-to-date
- [ ] All assembly files present
- [ ] Makefile builds successfully
- [ ] README.md describes M3
- [ ] Meaningful commit history
- [ ] Report named exactly: `m3_<teamname>.pdf`
- [ ] Submit via form: https://forms.gle/FcZn4YXqv1Jwcgip6
- [ ] Verify submission received

**Status**: 🔄 In Progress (Framework ready)

---

## Overall Progress

### Points Summary
| Task | Target | Current | % |
|------|--------|---------|---|
| Task 0 | 5 | 0 | 0% |
| Task 1 | 50 | 41 | 82% |
| Task 2 | 10 | 0 | 0% |
| Task 3 | 15 | 0 | 0% |
| Task 4 | TBD | - | - |
| Task 5 | 20 | 0 | 0% |
| Naming | 5 | 0 | 0% |
| **Total** | **105** | **41** | **39%** |

### Time Status
- **Days until deadline**: TBD (check calendar)
- **Estimated work remaining**: 15-25 hours
- **Current blockers**: S4D layer, main program, validation execution

### Critical Path
1. ✅ Framework & documentation (complete)
2. 🔄 S4D layer completion (HIGH PRIORITY - 15 pts)
3. 🔄 Main program integration (HIGH PRIORITY - 10 pts)
4. ⏳ Testing & validation (REQUIRED - 10 pts)
5. ⏳ Instruction counting (REQUIRED - 15 pts)
6. ⏳ Report writing (REQUIRED - 20 pts)

### Next Immediate Actions
1. [ ] **Complete S4D layer** (state-space dynamics) - 4-6 hrs
2. [ ] **Finish main.s** (orchestration) - 2-3 hrs
3. [ ] **Run validation** (compare outputs) - 3-4 hrs
4. [ ] **Instruction analysis** (counting & profiling) - 2-3 hrs
5. [ ] **Write report** (compile results) - 6-8 hrs

---

## Sign-Off

**Completed By**: [Team Member Name(s)]
**Date Last Updated**: April 17, 2026
**Status**: Framework Complete, Core Implementation Pending Finalization

**Next Reviewer**: [Instructor/TA Name]
**Review Date**: [TBD]

---

**Note**: This is a living document. Update status regularly as work progresses.

For detailed technical information, see:
- `README.md` - Build and usage guide
- `IMPLEMENTATION_GUIDE.md` - Technical deep dive
- `PROGRESS_SUMMARY.md` - Status summary
