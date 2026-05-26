# Makefile for M4: Vectorized RISC-V S4D galaxy classifier
# Usage:
#   make          - build galaxy_vec.hex and run
#   make test     - build with --profileinst for instruction counting
#   make clean    - remove build artifacts

TOOLCHAIN = /opt/riscv32imfcv/bin
AS      = $(TOOLCHAIN)/riscv32-unknown-elf-gcc
ASFLAGS = -march=rv32gcv -mabi=ilp32f -nostdlib
LD      = $(TOOLCHAIN)/riscv32-unknown-elf-ld
LDFLAGS = -T veer/link.ld -m elf32lriscv
OBJCOPY = $(TOOLCHAIN)/riscv32-unknown-elf-objcopy
WHISPER = /home/kai/VeeR-ISS/build-Linux/whisper
WFLAGS  = --configfile veer/whisper.json

BUILD   = build
SRCS    = main_vec.s linear_vec.s hilbert_vec.s take_last_vec.s \
          gelu_vec.s softmax_vec.s s4d_vec.s math_vec.s
OBJS    = $(patsubst %.s,$(BUILD)/%.o,$(SRCS))

.PHONY: all clean test run

all: $(BUILD)/galaxy_vec.hex run

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: %.s | $(BUILD)
	$(AS) $(ASFLAGS) -c $< -o $@

$(BUILD)/galaxy_vec.elf: $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^

$(BUILD)/galaxy_vec.hex: $(BUILD)/galaxy_vec.elf
	$(OBJCOPY) -O verilog $< $@

run: $(BUILD)/galaxy_vec.hex
	@echo "Running M4 vectorized inference on sample_09 (true label: Smooth Round)..."
	$(WHISPER) $(WFLAGS) \
	  --tohost 0xd0580000 --consoleio 0xd0580004 -s 0x80000000 \
	  --consoleoutfile $(BUILD)/galaxy_vec_probs.txt \
	  -x $(BUILD)/galaxy_vec.hex
	@echo "Output saved to $(BUILD)/galaxy_vec_probs.txt"

test: $(BUILD)/galaxy_vec.hex
	@echo "Running with --profileinst for instruction counting..."
	$(WHISPER) $(WFLAGS) \
	  --tohost 0xd0580000 --consoleio 0xd0580004 -s 0x80000000 \
	  --profileinst $(BUILD)/galaxy_vec_prof.log \
	  --consoleoutfile $(BUILD)/galaxy_vec_probs.txt \
	  -x $(BUILD)/galaxy_vec.hex
	@echo "Profile saved to $(BUILD)/galaxy_vec_prof.log"
	python3 count_instructions.py $(BUILD)/galaxy_vec_prof.log

clean:
	rm -rf $(BUILD)
