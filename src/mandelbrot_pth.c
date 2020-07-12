#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

int num_threads;

double c_x_min;
double c_x_max;
double c_y_min;
double c_y_max;

double pixel_width;
double pixel_height;

int iteration_max = 200;

int image_size;
unsigned char **image_buffer;

int i_x_max;
int i_y_max;
int image_buffer_size;

int gradient_size = 16;
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

struct thread_data {
    int thread_initial_y, thread_final_y;
};

struct timer_info {
    clock_t c_start;
    clock_t c_end;
    struct timespec t_start;
    struct timespec t_end;
    struct timeval v_start;
    struct timeval v_end;
};

struct timer_info program_timer;

void allocate_image_buffer()
{
    int rgb_size = 3;
    image_buffer = (unsigned char **) malloc(sizeof(unsigned char *) * image_buffer_size);

    for (int i = 0; i < image_buffer_size; i++) {
        image_buffer[i] = (unsigned char *) malloc(sizeof(unsigned char) * rgb_size);
    }
}

void deallocate_image_buffer()
{
    for (int i = 0; i < image_buffer_size; i++) {
        free(image_buffer[i]);
    }
    free(image_buffer);
}

void init(int argc, char *argv[])
{
    if (argc < 7) {
        printf("usage: ./mandelbrot_pth c_x_min c_x_max c_y_min c_y_max image_size num_threads\n");
        printf("examples with image_size = 11500:\n");
        printf("    Full Picture:         ./mandelbrot_pth -2.5 1.5 -2.0 2.0 11500 64\n");
        printf("    Seahorse Valley:      ./mandelbrot_pth -0.8 -0.7 0.05 0.15 11500 64\n");
        printf("    Elephant Valley:      ./mandelbrot_pth 0.175 0.375 -0.1 0.1 11500 64\n");
        printf("    Triple Spiral Valley: ./mandelbrot_pth -0.188 -0.012 0.554 0.754 11500 64\n");
        exit(0);
    } else {
        sscanf(argv[1], "%lf", &c_x_min);
        sscanf(argv[2], "%lf", &c_x_max);
        sscanf(argv[3], "%lf", &c_y_min);
        sscanf(argv[4], "%lf", &c_y_max);
        sscanf(argv[5], "%d", &image_size);
        sscanf(argv[6], "%d", &num_threads);

        num_threads = (num_threads > image_size) ? image_size : num_threads;

        i_x_max           = image_size;
        i_y_max           = image_size;
        image_buffer_size = image_size * image_size;

        pixel_width       = (c_x_max - c_x_min) / i_x_max;
        pixel_height      = (c_y_max - c_y_min) / i_y_max;
    }
}

void update_rgb_buffer(int iteration, int x, int y)
{
    int color;

    if (iteration == iteration_max) {
        image_buffer[(i_y_max * y) + x][0] = colors[gradient_size][0];
        image_buffer[(i_y_max * y) + x][1] = colors[gradient_size][1];
        image_buffer[(i_y_max * y) + x][2] = colors[gradient_size][2];
    } else {
        color = iteration % gradient_size;

        image_buffer[(i_y_max * y) + x][0] = colors[color][0];
        image_buffer[(i_y_max * y) + x][1] = colors[color][1];
        image_buffer[(i_y_max * y) + x][2] = colors[color][2];
    }
}

void write_to_file()
{
    FILE * file;
    char * filename               = "output.ppm";
    char * comment                = "# ";

    int max_color_component_value = 255;

    file = fopen(filename,"wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
            i_x_max, i_y_max, max_color_component_value);

    for(int i = 0; i < image_buffer_size; i++){
        fwrite(image_buffer[i], 1 , 3, file);
    };

    fclose(file);
}

void *compute_mandelbrot_parallel(void *args)
{
    struct thread_data *thread_args;
    double z_x, z_y;
    double z_x_squared, z_y_squared;
    double escape_radius_squared = 4;
    int iteration;
    int i_y, i_x;
    double c_y, c_x;

    thread_args = (struct thread_data *) args;
    for (i_y = thread_args->thread_initial_y; i_y < thread_args->thread_final_y; i_y++) {
        c_y = c_y_min + i_y * pixel_height;
        if (fabs(c_y) < pixel_height / 2) 
            c_y = 0.0;
        for (i_x = 0; i_x < i_x_max; i_x++) {
            c_x = c_x_min + i_x * pixel_width;
            z_x = z_y = 0.0;
            z_x_squared = z_y_squared = 0.0;
            for (iteration = 0;
                iteration < iteration_max && \
                ((z_x_squared + z_y_squared) < escape_radius_squared);
                iteration++){
                    z_y = 2 * z_x * z_y + c_y;
                    z_x = z_x_squared - z_y_squared + c_x;
                    z_x_squared = z_x * z_x;
                    z_y_squared = z_y * z_y;
            }
            update_rgb_buffer(iteration, i_x, i_y);
        }
    }
    pthread_exit(NULL);
}

void compute_mandelbrot()
{
    pthread_t *threads;
    struct thread_data *thread_data_array;
    pthread_attr_t attr;
    int t, error_code, y_per_thread;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    threads = malloc(num_threads * sizeof(pthread_t));
    thread_data_array = malloc(num_threads * sizeof(struct thread_data));

    y_per_thread = round((double) i_y_max / num_threads);
    for (t = 0; t < num_threads; t++) {
        thread_data_array[t].thread_initial_y = t * y_per_thread;
        if (t == num_threads - 1)
            thread_data_array[t].thread_final_y = i_y_max;
        else
            thread_data_array[t].thread_final_y = (t + 1) * y_per_thread;
        error_code = pthread_create(&threads[t], &attr, 
                        compute_mandelbrot_parallel, (void *) &thread_data_array[t]);
         if (error_code) {
            printf("ERROR; return code from pthread_create() is %d\n", error_code);
            exit(-1);
        }
    }
    
    pthread_attr_destroy(&attr);   
    for (t = 0; t < num_threads; t++) {
        error_code = pthread_join(threads[t], NULL);
        if (error_code){
            printf("ERROR; return code from pthread_join() is %d\n", error_code);
            exit(-1);
        }
    }
}

void start_timer(struct timer_info *timer_pointer) 
{
    timer_pointer->c_start = clock();
    clock_gettime(CLOCK_MONOTONIC, &timer_pointer->t_start);
    gettimeofday(&timer_pointer->v_start, NULL);
}

void finish_timer(struct timer_info *timer_pointer)
{
    timer_pointer->c_end = clock();
    clock_gettime(CLOCK_MONOTONIC, &timer_pointer->t_end);
    gettimeofday(&timer_pointer->v_end, NULL);
}

int main(int argc, char *argv[]){
    init(argc, argv);

    allocate_image_buffer();

    start_timer(&program_timer);
    compute_mandelbrot();
    finish_timer(&program_timer);

    write_to_file();

    deallocate_image_buffer();

    printf("%f\n",
            (double) (program_timer.t_end.tv_sec - program_timer.t_start.tv_sec) +
            (double) (program_timer.t_end.tv_nsec - program_timer.t_start.tv_nsec) / 1000000000.0);
    return 0;
}
