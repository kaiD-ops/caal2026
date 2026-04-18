# ============================================================================
# Makefile for RISC-V S4D Galaxy Classifier - Milestone 3
# ============================================================================
#
# Targets:
#   make build    - Compile and link the binary
#   make clean    - Remove build artifacts
#   make help     - Show this help message
#
# Prerequisites:
#   - RISC-V GNU toolchain (riscv64-unknown-elf-gcc, riscv64-unknown-elf-as, etc.)
#   - VeeR-iSS simulator (for execution)
#

# ============================================================================
# Configuration
# ============================================================================

# RISC-V toolchain prefix (riscv32 for VeeR 32-bit)
CROSS_COMPILE ?= riscv32-unknown-elf-

# Tool commands
CC  = $(CROSS_COMPILE)gcc
AS  = $(CROSS_COMPILE)as
LD  = $(CROSS_COMPILE)ld
AR  = $(CROSS_COMPILE)ar
OBJDUMP = $(CROSS_COMPILE)objdump
READELF = $(CROSS_COMPILE)readelf

# Directories
SRC_DIR     = src
INCLUDE_DIR = include
BUILD_DIR   = build
TEST_DIR    = tests
BIN_DIR     = bin

# Compilation flags
# RV32IMF: 32-bit RISC-V with integer, multiply/divide, and floating-point
CFLAGS   = -march=rv32imf -mabi=ilp32f -static
ASFLAGS  = -march=rv32imf --warn
LDFLAGS  = -T linker.ld -Map $(BUILD_DIR)/program.map

# Source files
MATH_SRC       = $(SRC_DIR)/math.s
HILBERT_SRC    = $(SRC_DIR)/hilbert_scan.s
HILBERT_TEST_SRC = $(SRC_DIR)/hilbert_scan_test.s
LINEAR_SRC     = $(SRC_DIR)/linear_layer.s
TAKELT_SRC     = $(SRC_DIR)/take_last_timestep.s
SOFTMAX_SRC    = $(SRC_DIR)/softmax_inplace.s
GELU_SRC       = $(SRC_DIR)/gelu_inplace.s
S4D_SRC        = $(SRC_DIR)/s4d_layer.s
MAIN_SRC       = $(SRC_DIR)/main.s

ALL_SRC = $(MATH_SRC) $(HILBERT_SRC) $(HILBERT_TEST_SRC) $(LINEAR_SRC) $(TAKELT_SRC) \
          $(SOFTMAX_SRC) $(GELU_SRC) $(S4D_SRC) $(MAIN_SRC)

# Object files
MATH_OBJ       = $(BUILD_DIR)/math.o
HILBERT_OBJ    = $(BUILD_DIR)/hilbert_scan.o
HILBERT_TEST_OBJ = $(BUILD_DIR)/hilbert_scan_test.o
LINEAR_OBJ     = $(BUILD_DIR)/linear_layer.o
TAKELT_OBJ     = $(BUILD_DIR)/take_last_timestep.o
SOFTMAX_OBJ    = $(BUILD_DIR)/softmax_inplace.o
GELU_OBJ       = $(BUILD_DIR)/gelu_inplace.o
S4D_OBJ        = $(BUILD_DIR)/s4d_layer.o
MAIN_OBJ       = $(BUILD_DIR)/main.o

ALL_OBJ = $(MATH_OBJ) $(HILBERT_OBJ) $(HILBERT_TEST_OBJ) $(LINEAR_OBJ) $(TAKELT_OBJ) \
          $(SOFTMAX_OBJ) $(GELU_OBJ) $(S4D_OBJ) $(MAIN_OBJ)

# Output executable
TARGET = $(BIN_DIR)/s4d_classifier

# ============================================================================
# Targets
# ============================================================================

.PHONY: all clean help disassemble readelf test build build-main-only

# Ensure directories exist first
all: create_dirs $(TARGET)

# Create build directories
create_dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR) $(TEST_DIR)

# Assemble math.s
$(MATH_OBJ): $(MATH_SRC) | create_dirs
	$(AS) $(ASFLAGS) -Iinclude -o $@ $<

# Assemble hilbert_scan.s
$(HILBERT_OBJ): $(HILBERT_SRC) | create_dirs
	$(AS) $(ASFLAGS) -Iinclude -o $@ $<

# Assemble hilbert_scan_test.s
$(HILBERT_TEST_OBJ): $(HILBERT_TEST_SRC) | create_dirs
	$(AS) $(ASFLAGS) -Iinclude -o $@ $<

# Assemble linear_layer.s
$(LINEAR_OBJ): $(LINEAR_SRC) | create_dirs
	$(AS) $(ASFLAGS) -Iinclude -o $@ $<

# Assemble take_last_timestep.s
$(TAKELT_OBJ): $(TAKELT_SRC) | create_dirs
	$(AS) $(ASFLAGS) -Iinclude -o $@ $<

# Assemble softmax_inplace.s
$(SOFTMAX_OBJ): $(SOFTMAX_SRC) | create_dirs
	$(AS) $(ASFLAGS) -Iinclude -o $@ $<

# Assemble gelu_inplace.s
$(GELU_OBJ): $(GELU_SRC) | create_dirs
	$(AS) $(ASFLAGS) -Iinclude -o $@ $<

# Assemble s4d_layer.s
$(S4D_OBJ): $(S4D_SRC) | create_dirs
	$(AS) $(ASFLAGS) -Iinclude -o $@ $<

# Assemble main.s
$(MAIN_OBJ): $(MAIN_SRC) | create_dirs
	$(AS) $(ASFLAGS) -Iinclude -o $@ $<

# Link all object files (normal build)
$(TARGET): $(ALL_OBJ) | create_dirs
	$(LD) $(LDFLAGS) -o $@ $(ALL_OBJ)
	@echo "Linked: $(TARGET)"

# Link ONLY main.o (for minimal testing - no buffers from other modules)
$(BIN_DIR)/s4d_classifier_main_only: $(MAIN_OBJ) | create_dirs
	$(LD) $(LDFLAGS) -o $@ $(MAIN_OBJ)
	@echo "Linked (main-only): $@"

# Alias 'build' to the main target
build: $(TARGET)
	@echo "Build complete: $(TARGET)"

# Build minimal version (main-only, no large buffers)
build-main-only: $(BIN_DIR)/s4d_classifier_main_only
	@echo "Build complete (main-only): $(BIN_DIR)/s4d_classifier_main_only"

# Generate disassembly
disassemble: $(TARGET)
	$(OBJDUMP) -d -S $(TARGET) > $(BUILD_DIR)/disassembly.txt
	@echo "Disassembly written to $(BUILD_DIR)/disassembly.txt"

# Read ELF header
readelf: $(TARGET)
	$(READELF) -h $(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) $(TEST_DIR)/*.o
	@echo "Cleaned build artifacts"

# Help
help:
	@echo "S4D Galaxy Classifier - RISC-V Milestone 3 Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make build        - Build the RISC-V binary (default)"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make disassemble  - Generate disassembly listing"
	@echo "  make readelf      - Show ELF header information"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  CROSS_COMPILE = $(CROSS_COMPILE)"
	@echo "  Target binary = $(TARGET)"
	@echo ""

# ============================================================================
# Test Targets (placeholder - would be implemented with test data)
# ============================================================================

test: build
	@echo "Test target not yet implemented"
	@echo "TODO: Integrate VeeR-iSS simulation and Python validation"
