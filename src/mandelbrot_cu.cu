#include <stdio.h>
#include <stdlib.h>

// const int GRADIENT_SIZE = 16;
// __const__ int d_gradient_size;
// const int COLORS[17][3] = {
//                         {66, 30, 15},
//                         {25, 7, 26},
//                         {9, 1, 47},
//                         {4, 4, 73},
//                         {0, 7, 100},
//                         {12, 44, 138},
//                         {24, 82, 177},
//                         {57, 125, 209},
//                         {134, 181, 229},
//                         {211, 236, 248},
//                         {241, 233, 191},
//                         {248, 201, 95},
//                         {255, 170, 0},
//                         {204, 128, 0},
//                         {153, 87, 0},
//                         {106, 52, 3},
//                         {16, 16, 16},
//                     };

// const double C_X_MIN = -0.188,
//                 C_X_MAX = -0.012,
//                 C_Y_MIN = 0.554,
//                 C_Y_MAX = 0.754;
// __const__ double d_c_x_min,
//                     d_c_x_max,
//                     d_c_y_min,
//                     d_c_y_max;

const int IMAGE_SIZE = 4096;
__const__ int d_image_size;

void init(int argc, char *argv[])
{
    cudaMemcpyToSymbol(&d_image_size, &IMAGE_SIZE, sizeof(int));
}

__global__ void test()
{
    printf("%d", image_size);
}

int main(int argc, char *argv[])
{
    init(argc, argv);
    test<<<1,1>>>();
}