# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = icc
# OPT = -march=native -O3 -fomit-frame-pointer -mtune=native -fast
OPT = -fast -mtune=native -fomit-frame-pointer
CFLAGS = -Wall -DGETTIMEOFDAY -std=c99 $(OPT) -diag-disable=10441
LDFLAGS = -Wall 
# mkl is needed for blas implementation
LDLIBS = -qmkl=sequential -lpthread -lm -diag-disable=10441

targets = benchmark-test benchmark-naive benchmark-blocked benchmark-blas
objects = benchmark-test.o benchmark.o sgemm-naive.o sgemm-blocked.o sgemm-blas.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-test : benchmark-test.o sgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-naive : benchmark.o sgemm-naive.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o sgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o sgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
