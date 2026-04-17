# Makefile for RISC-V S4D galaxy classifier assembly
# Usage:
#   make          - build and run main
#   make test     - build and run with --profileinst for instruction counting
#   make clean    - remove build artifacts

AS      = riscv32-unknown-elf-gcc
ASFLAGS = -march=rv32imf -mabi=ilp32f -nostdlib
LD      = riscv32-unknown-elf-ld
LDFLAGS = -T veer/link.ld -m elf32lriscv
OBJCOPY = riscv32-unknown-elf-objcopy
WHISPER = whisper
WFLAGS  = --configfile veer/whisper.json

BUILD   = build
SRCS    = main.s layers.s math.s
OBJS    = $(patsubst %.s,$(BUILD)/%.o,$(SRCS))

.PHONY: all clean test run

all: $(BUILD)/galaxy.exe run

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: %.s | $(BUILD)
	$(AS) $(ASFLAGS) -c $< -o $@

$(BUILD)/galaxy.elf: $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^

$(BUILD)/galaxy.hex: $(BUILD)/galaxy.elf
	$(OBJCOPY) -O ihex $< $@

$(BUILD)/galaxy.exe: $(BUILD)/galaxy.hex
	cp $< $@

run: $(BUILD)/galaxy.exe
	@echo "Running inference on sample_09 (true label: Smooth Round)..."
	$(WHISPER) $(WFLAGS) --logfile $(BUILD)/galaxy.log $(BUILD)/galaxy.exe
	@echo "Log saved to $(BUILD)/galaxy.log"

test: $(BUILD)/galaxy.exe
	@echo "Running with --profileinst for instruction counting..."
	$(WHISPER) $(WFLAGS) --profileinst $(BUILD)/galaxy_prof.log $(BUILD)/galaxy.exe
	@echo "Profile saved to $(BUILD)/galaxy_prof.log"

clean:
	rm -rf $(BUILD)
