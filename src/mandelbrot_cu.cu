#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GRADIENT_SIZE 16

#define C_X_MIN -0.188
#define C_X_MAX -0.012
#define C_Y_MIN 0.554
#define C_Y_MAX 0.754

#define IMAGE_SIZE 4096
#define ARRAY_SIZE (3 * IMAGE_SIZE * IMAGE_SIZE * sizeof(unsigned char)) 

#define PIXEL_WIDTH ((C_X_MAX - C_X_MIN) / IMAGE_SIZE)
#define PIXEL_HEIGHT ((C_Y_MAX - C_Y_MIN) / IMAGE_SIZE)
#define ITERATION_MAX 200

int colors[51] = {
                    66, 30, 15,
                    25, 7, 26,
                    9, 1, 47,
                    4, 4, 73,
                    0, 7, 100,
                    12, 44, 138,
                    24, 82, 177,
                    57, 125, 209,
                    134, 181, 229,
                    211, 236, 248,
                    241, 233, 191,
                    248, 201, 95,
                    255, 170, 0,
                    204, 128, 0,
                    153, 87, 0,
                    106, 52, 3,
                    16, 16, 16,
                    };
int *d_colors;

int x_grid;
int y_grid;
int x_block;
int y_block;
dim3 dimGrid;
dim3 dimBlock;

unsigned char *image_buffer;
unsigned char *d_image_buffer;

void init(int argc, char *argv[])
{
    if (argc != 5) {
        printf("usage: ./mandelbrot_cu x_grid y_grid x_blocks y_blocks");
        exit(0);
    } 
    sscanf(argv[1], "%lf", &x_grid);
    sscanf(argv[2], "%lf", &y_grid);
    sscanf(argv[3], "%lf", &x_block);
    sscanf(argv[4], "%lf", &y_block);

    cudaMallocHost((void **) &image_buffer, ARRAY_SIZE);
    cudaMalloc((void **) &d_image_buffer, ARRAY_SIZE);

    cudaMalloc((void **) &d_colors, 51 * sizeof(int));
    cudaMemcpy(d_colors, colors, 51 * sizeof(int), cudaMemcpyHostToDevice);

    dimGrid = dim3(x_grid, y_grid, 1);
    dimBlock = dim3(x_block, y_block, 1);
}

void write_to_file()
{
    FILE * file;
    const char *filename = "output.ppm";
    const char *comment  = "# ";

    int max_color_component_value = 255;

    file = fopen(filename,"wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
            IMAGE_SIZE, IMAGE_SIZE, max_color_component_value);

    for(int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i += 3){
        fwrite(&image_buffer[i], 1 , 3, file);
    };

    fclose(file);
}

__global__ void gpu_compute_mandelbrot(unsigned char *buffer, int *colors_d)
{
    double z_x = 0.0;
    double z_y = 0.0;
    double z_x_squared = 0.0;
    double z_y_squared = 0.0;
    double escape_radius_squared = 4;

    double c_x;
    double c_y;
    
    int i_y = blockIdx.y * blockDim.y + threadIdx.y; 
    int i_x = blockIdx.x * blockDim.x + threadIdx.x;

    int color;
    int iteration;

    c_y = C_Y_MIN + i_y * PIXEL_HEIGHT;
    if (fabs(c_y) < PIXEL_HEIGHT / 2)
        c_y = 0.0;
    c_x = C_X_MIN + i_x * PIXEL_WIDTH;
    for (iteration = 0;
                iteration < ITERATION_MAX && \
                ((z_x_squared + z_y_squared) < escape_radius_squared);
                iteration++) {
                z_y         = 2 * z_x * z_y + c_y;
                z_x         = z_x_squared - z_y_squared + c_x;

                z_x_squared = z_x * z_x;
                z_y_squared = z_y * z_y;
    }
    color = (iteration == ITERATION_MAX) ? GRADIENT_SIZE : iteration % GRADIENT_SIZE;
    for (int i = 0; i < 3; i++)
        buffer[(IMAGE_SIZE * i_y) + i_x + i] = colors_d[color * 3 + i];
}

void compute_mandelbrot()
{
    gpu_compute_mandelbrot<<<dimGrid, dimBlock>>>(d_image_buffer, d_colors);
    cudaMemcpy(image_buffer, d_image_buffer, ARRAY_SIZE, cudaMemcpyDeviceToHost);
}

int main(int argc, char *argv[])
{
    init(argc, argv);

    compute_mandelbrot();

    write_to_file();

    cudaFreeHost(image_buffer); cudaFree(d_image_buffer);
    cudaFree(d_colors);

    printf("Foi");
    return 0;
}