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

mv *.log results/
rm output.ppm