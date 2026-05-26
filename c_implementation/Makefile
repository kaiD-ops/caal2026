CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -I.
LDFLAGS = -lm

.PHONY: all clean bench asm

all: galaxy_infer galaxy_test

galaxy_infer: main.c nn.c nn.h
	$(CC) $(CFLAGS) -O2 -o $@ main.c nn.c $(LDFLAGS)
	@echo "Built: $@"

galaxy_test: test.c nn.c nn.h
	$(CC) $(CFLAGS) -O2 -o $@ test.c nn.c $(LDFLAGS)
	@echo "Built: $@"

bench: galaxy_infer_O0 galaxy_infer_O1 galaxy_infer_O2 galaxy_infer_O3 galaxy_infer_Ofast

galaxy_infer_O0: main.c nn.c nn.h
	$(CC) $(CFLAGS) -O0 -o $@ main.c nn.c $(LDFLAGS)

galaxy_infer_O1: main.c nn.c nn.h
	$(CC) $(CFLAGS) -O1 -o $@ main.c nn.c $(LDFLAGS)

galaxy_infer_O2: main.c nn.c nn.h
	$(CC) $(CFLAGS) -O2 -o $@ main.c nn.c $(LDFLAGS)

galaxy_infer_O3: main.c nn.c nn.h
	$(CC) $(CFLAGS) -O3 -o $@ main.c nn.c $(LDFLAGS)

galaxy_infer_Ofast: main.c nn.c nn.h
	$(CC) $(CFLAGS) -Ofast -o $@ main.c nn.c $(LDFLAGS)

asm: nn_O0.s nn_O2.s nn_O3.s

nn_O0.s: nn.c nn.h
	$(CC) $(CFLAGS) -O0 -S -o $@ nn.c

nn_O2.s: nn.c nn.h
	$(CC) $(CFLAGS) -O2 -S -o $@ nn.c

nn_O3.s: nn.c nn.h
	$(CC) $(CFLAGS) -O3 -S -o $@ nn.c

clean:
	rm -f galaxy_infer galaxy_test
	rm -f galaxy_infer_O0 galaxy_infer_O1 galaxy_infer_O2 galaxy_infer_O3 galaxy_infer_Ofast
	rm -f *.o *.s
	@echo "Cleaned."
