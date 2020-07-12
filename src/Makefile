OUTPUT=mandelbrot

IMAGE=.ppm

CC=gcc
# CC_OPT=-std=c11

CC_OMP=-fopenmp
CC_PTH=-pthread

CCU=nvcc
CMPI=mpicc

.PHONY: all
all: $(OUTPUT)_omp $(OUTPUT)_pth $(OUTPUT)_seq $(OUTPUT)_ompi $(OUTPUT)_cu

$(OUTPUT)_omp: $(OUTPUT)_omp.c
	$(CC) -o $(OUTPUT)_omp $(CC_OMP) $(OUTPUT)_omp.c

$(OUTPUT)_pth: $(OUTPUT)_pth.c
	$(CC) -o $(OUTPUT)_pth $(CC_PTH) $(OUTPUT)_pth.c -lm

$(OUTPUT)_seq: $(OUTPUT)_seq.c
	$(CC) -o $(OUTPUT)_seq $(OUTPUT)_seq.c

$(OUTPUT)_ompi: $(OUTPUT)_ompi.c
	$(CMPI) -o $(OUTPUT)_ompi $(OUTPUT)_ompi.c

$(OUTPUT)_cu: $(OUTPUT)_cu.cu
	$(CCU) -o $(OUTPUT)_cu $(OUTPUT)_cu.cu

.PHONY: clean
clean:
	rm  $(OUTPUT)_omp $(OUTPUT)_pth $(OUTPUT)_seq $(OUTPUT)_ompi $(OUTPUT)_cu *$(IMAGE)