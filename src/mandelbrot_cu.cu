#include <stdio.h>
#include <stdlib.h>


void init(int argc, char *argv[])
{
    __device__ int gradient_size = 16;
    __device__ int colors[17][3] = {
                            {66, 30, 15},
                            {25, 7, 26},
                            {9, 1, 47},
                            {4, 4, 73},
                            {0, 7, 100},
                            {12, 44, 138},
                            {24, 82, 177},
                            {57, 125, 209},
                            {134, 181, 229},
                            {211, 236, 248},
                            {241, 233, 191},
                            {248, 201, 95},
                            {255, 170, 0},
                            {204, 128, 0},
                            {153, 87, 0},
                            {106, 52, 3},
                            {16, 16, 16},
                        };

    __device__ double c_x_min = -0.188;
    __device__ double c_x_max = -0.012;
    __device__ double c_y_min = 0.554;
    __device__ double c_y_max = 0.754;
    __device__ int image_size = 4096;

    __device__ int image_buffer_size = image_size * image_size;
    __device__ double pixel_width  = (c_x_max - c_x_min) / image_size;
    __device__ double pixel_height = (c_y_max - c_y_min) / image_size;
}

__global__ void test()
{
    printf("%d", image_size);
}

int main(int argc, char *argv[])
{
    init(argc, argv);
    test<<<1>>>();
}