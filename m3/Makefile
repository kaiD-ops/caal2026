# Makefile for Milestone 3: S4D Galaxy Classifier in scalar RISC-V assembly
# Toolchain: riscv32-unknown-elf-gcc 15.2.0, rv32imf, ilp32f

GCC_PREFIX = riscv32-unknown-elf
ASFLAGS    = -march=rv32imf -mabi=ilp32f -nostdlib
LDFLAGS    = -T veer/link.ld -m elf32lriscv
WHISPER    = /home/kai/VeeR-ISS/build-Linux/whisper
TOHOST     = 0xd0580000
CONSOLEIO  = 0xd0580004
WFLAGS     = --configfile veer/whisper.json --tohost $(TOHOST)
BUILD      = build

export PATH := /opt/riscv32imfcv/bin:$(PATH)

SRCS = math.s hilbert_scan.s linear_layer.s s4d_layer.s \
       gelu.s take_last_timestep.s softmax.s main.s
OBJS = $(patsubst %.s,$(BUILD)/obj/%.o,$(SRCS))

.PHONY: all clean run test

all: $(BUILD)/hex/galaxy.hex

$(BUILD)/obj/%.o: %.s | $(BUILD)/obj $(BUILD)/hex $(BUILD)/logs
	$(GCC_PREFIX)-gcc $(ASFLAGS) -c $< -o $@

$(BUILD)/exe/galaxy.elf: $(OBJS) | $(BUILD)/exe
	$(GCC_PREFIX)-ld $(LDFLAGS) -o $@ $^

$(BUILD)/hex/galaxy.hex: $(BUILD)/exe/galaxy.elf
	$(GCC_PREFIX)-objcopy -O ihex $< $@

run: $(BUILD)/exe/galaxy.elf
	@echo "Running inference on sample_09 (true label: Smooth Round)..."
	$(WHISPER) $(WFLAGS) --consoleio $(CONSOLEIO) \
		--consoleoutfile $(BUILD)/logs/galaxy_probs.txt $<
	@echo "Class probabilities (big-endian hex) -> $(BUILD)/logs/galaxy_probs.txt:"
	@cat $(BUILD)/logs/galaxy_probs.txt

test: $(BUILD)/exe/galaxy.elf
	@echo "Running with --profileinst for instruction counting..."
	$(WHISPER) $(WFLAGS) --profileinst $(BUILD)/logs/galaxy_prof.log $<

$(BUILD)/obj $(BUILD)/hex $(BUILD)/exe $(BUILD)/logs:
	mkdir -p $@

clean:
	rm -rf $(BUILD)
