#include <stdio.h>
#include <stdlib.h>

#define GRADIENT_SIZE 16

#define C_X_MIN -0.188
#define C_X_MAX -0.012
#define C_Y_MIN 0.554
#define C_Y_MAX 0.754

#define IMAGE_SIZE 4096


void init(int argc, char *argv[])
{

}

__global__ void test(int *t)
{
    *t = IMAGE_SIZE * 2;
}

int main(int argc, char *argv[])
{
    // init(argc, argv);
    int *t;
    cudaMallocHost((void **) &t, sizeof(int));
    test<<<1,1>>>(t);
    cudaDeviceSynchronize();
    printf("%d", *t);
    cudaFree(t)
}