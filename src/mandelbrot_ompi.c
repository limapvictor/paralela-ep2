#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MASTER 0

#define C_X_MIN -0.188
#define C_X_MAX -0.012
#define C_Y_MIN 0.554
#define C_Y_MAX 0.754

#define IMAGE_SIZE 4096
#define ARRAY_SIZE (3 * IMAGE_SIZE * IMAGE_SIZE * sizeof(unsigned char)) 

#define PIXEL_WIDTH ((C_X_MAX - C_X_MIN) / IMAGE_SIZE)
#define PIXEL_HEIGHT ((C_Y_MAX - C_Y_MIN) / IMAGE_SIZE)
#define GRADIENT_SIZE 16
#define ITERATION_MAX 200

int colors[17][3] = {
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

unsigned char *image_buffer;
unsigned char *task_image_buffer;
int task_array_size;

int numtasks;
int taskid;

int task_y_offset;

#define TAG_ID 0
#define TAG_ARRAY 1


void init()
{
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    task_array_size = ARRAY_SIZE / numtasks;
    task_y_offset = IMAGE_SIZE / numtasks;

    if (taskid == MASTER)
        image_buffer = (unsigned char *) malloc(ARRAY_SIZE);
    else
        task_image_buffer = (unsigned char *) malloc(task_array_size);
}

void deallocate_image_buffer()
{
    if (taskid == MASTER)
        free(image_buffer);
    else
        free(task_image_buffer);
}

void write_to_file()
{
    FILE * file;
    char * filename               = "output.ppm";
    char * comment                = "# ";

    int max_color_component_value = 255;

    file = fopen(filename,"wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
            IMAGE_SIZE, IMAGE_SIZE, max_color_component_value);

    for(int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++){
        fwrite(&image_buffer[3 * i], 1 , 3, file);
    };

    fclose(file);
}

void compute_mandelbrot()
{
    double z_x, z_y;
    double z_x_squared, z_y_squared;
    double escape_radius_squared = 4;
    int iteration, color;
    int i_y, i_x;
    double c_y, c_x;
    unsigned char *buffer = (taskid == MASTER) ? image_buffer : task_image_buffer;
    
    for (int i_y = taskid * task_y_offset; i_y < (taskid + 1) * task_y_offset; i_y++) {
        c_y = C_Y_MIN + i_y * PIXEL_HEIGHT;
        if (fabs(c_y) < PIXEL_HEIGHT / 2) 
            c_y = 0.0;
        for (i_x = 0; i_x < IMAGE_SIZE; i_x++) {
            c_x = C_X_MIN + i_x * PIXEL_WIDTH;
            z_x = z_y = 0.0;
            z_x_squared = z_y_squared = 0.0;
            for (iteration = 0;
                iteration < ITERATION_MAX && \
                ((z_x_squared + z_y_squared) < escape_radius_squared);
                iteration++) {
                    z_y = 2 * z_x * z_y + c_y;
                    z_x = z_x_squared - z_y_squared + c_x;
                    z_x_squared = z_x * z_x;
                    z_y_squared = z_y * z_y;
            }
            color = (iteration == ITERATION_MAX) ? GRADIENT_SIZE : (iteration % GRADIENT_SIZE);
            for (int c = 0; c < 3; c++) {
                buffer[3 * ((i_y - taskid * task_y_offset) * IMAGE_SIZE + i_x) + c] = colors[color][c];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int msg_dest, msg_src;
    double program_start, program_finish;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    program_start = MPI_Wtime();
    init();

    compute_mandelbrot();

    if (taskid != MASTER) {
        msg_dest = MASTER;
        MPI_Send(&task_image_buffer[0], task_array_size, MPI_UNSIGNED_CHAR, \
            msg_dest, TAG_ARRAY, MPI_COMM_WORLD);
    } else {
        for (msg_src = 1; msg_src < numtasks; msg_src++) {
            MPI_Recv(&image_buffer[msg_src * task_array_size], task_array_size, MPI_UNSIGNED_CHAR, \
                msg_src, TAG_ARRAY, MPI_COMM_WORLD, &status);
        }
        write_to_file();
    }

    deallocate_image_buffer();

    program_finish = MPI_Wtime();
    MPI_Finalize();

    // printf("%.8f", program_start - program_finish);
}