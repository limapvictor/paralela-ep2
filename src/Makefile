OUTPUT=mandelbrot

IMAGE=.ppm

CCU=nvcc
CMPI=mpicc

.PHONY: all
all: $(OUTPUT)_ompi $(OUTPUT)_cu

$(output)_ompi: $(OUTPUT)_ompi.c
	$(CMPI) -o $(OUTPUT)_ompi $(OUTPUT)_ompi.c

$(output)_cu: $(OUTPUT)_cu.cu
	$(CCU) -o $(OUTPUT)_cu $(OUTPUT)_cu.cu

.PHONY: clean
clean:
	rm $(OUTPUT)_ompi $(OUTPUT)_cu *$(IMAGE)