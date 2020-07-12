#! /bin/bash

set -o xtrace

ITERATIONS=5

make
mkdir results

NUMBER_PROCS=2
for ((m=1; m <= 6; m++)); do
    echo "Info $NUMBER_PROCS" >> ompi.log 2>&1
    for ((i=1; i <= ITERATIONS; i++)); do
        mpirun -host localhost:$NUMBER_PROCS mandelbrot_ompi >> ompi.log 2>&1
    done
    NUMBER_PROCS=$(($NUMBER_PROCS * 2))
done

GRID = (2048 1024 512 256 128)
BLOCK = (2 4 8 16 32)
for ((m=0; m < 5; m++)); do
    echo "Info ${GRID[m]} ${GRID[m]} ${BLOCK[m]} ${BLOCK[m]}" >> cuda.log 2>&1
    for ((i=1; i <= ITERATIONS; i++)); do
        ./mandelbrot_cu ${GRID[m]} ${GRID[m]} ${BLOCK[m]} ${BLOCK[m]} >> cuda.log 2>&1
    done
done

NAMES=('mandelbrot_seq' 'mandelbrot_pth' 'mandebrot_omp')
THREADS=('' '16' '32')
FILES=('seq' 'pth' 'omp')
for ((j=0; j < 3; j++)); do
    echo "Info ${THREADS[j]}" >> ${FILES[j]}.log 2>&1
    for ((i=1; i <= ITERATIONS; i++)); do
        ./${NAMES[j]} -0.188 -0.012 0.554 0.754 ${THREADS[j]} >> ${FILES[j]}.log 2>&1
    done
done

done

mv *.log results/
rm output.ppm