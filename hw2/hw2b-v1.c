#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <pthread.h>

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int is_final_(int rank, int size){
    if (rank == size - 1)
        return(1);
    else
        return(0);
}

int numPart_;
int* image;
int *result;
int numPart;
int p;
int width;
int height;

void* compute(){
    for (int j = p * numPart; j < numPart_; ++j){
        for (int i = 0; i < width; ++i){
            image[j * width + i] = result[j * width + i];
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));

    /* argument parsing */
    assert(argc == 9);
    int num_threads = CPU_COUNT(&cpu_set);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);
    
    // int* result = (int*)malloc(width * height * sizeof(int));
    // assert(result);

    numPart = height / size;

    // if(is_final_(rank, size)){
    //     numPart_ = height;
    // }else{
    //     numPart_ = (rank + 1) * numPart;
    // }

    if(is_final_(rank, size)){
        numPart_ = height;
    }else{
        numPart_ = (rank + 1) * numPart;
    }

    #pragma omp parallel for schedule(dynamic)
    /* mandelbrot set */
    for (int j = rank * numPart; j < numPart_; ++j) {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / width) + left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[j * width + i] = repeats;
        }
    }

    pthread_t threads[num_threads];
    int t;
    int rc;
    int ID[num_threads];

    if (rank == 0){
        result = (int*)malloc(width * height * sizeof(int));

        for (p = 1; p < size; p++){
            MPI_Recv(result, width * height, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if(is_final_(p, size)){
                numPart_ = height;
            }else{
                numPart_ = (p + 1) * numPart;
            }

            for (t = 0; t < num_threads; t++){
                ID[t] = t;

                rc = pthread_create(&threads[t], NULL, compute, NULL);
                if (rc){
                    printf("ERROR; return code from pthread_create() is %d\n", rc);
                    exit(-1);
                }
            }

            for (t = 0; t < num_threads; t++){
                pthread_join(threads[t], NULL);
            }
            compute();
        }
        
        /* draw and cleanup */
        write_png(filename, iters, width, height, image);
        free(image);
    }else{
        int numPart_temp;

        if (is_final_(rank, size)){
            numPart_temp = numPart + height % size;
        }else{
            numPart_temp = numPart;
        }

        MPI_Send(image, width * numPart_, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }



    // int numPart_temp;
    // int *result;

    // if (rank == 0){
    //     for (int p = 1; p < size; p++){
    //         if (is_final_(p, size)){
    //             numPart_temp = numPart + height % size;
    //         }else{
    //             numPart_temp = numPart;
    //         }

    //         result = (int*)malloc(width * numPart_temp * sizeof(int));
    //         // assert(result);

    //         MPI_Recv(result, width * numPart_temp, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         // memcpy(image + p * numPart, result, width * numPart_temp * sizeof(int));
                        
    //         if(is_final_(p, size)){
    //             numPart_ = height;
    //         }else{
    //             numPart_ = (p + 1) * numPart;
    //         }

    //         for (int j = p * numPart; j < numPart_; ++j){
    //             for (int i = 0; i < width; ++i){
    //                 image[j * width + i] = result[j * width + i];
    //             }
    //         }
    //     }
        
    //     /* draw and cleanup */
    //     write_png(filename, iters, width, height, image);
    //     free(image);
    // }else{
    //     if (is_final_(rank, size)){
    //         numPart_temp = numPart + height % size;
    //     }else{
    //         numPart_temp = numPart;
    //     }

    //     MPI_Send(image, width * numPart_temp, MPI_INT, 0, 0, MPI_COMM_WORLD);
    // }

    // int *result;
    
    // if (rank == 0){
    //     result = (int *)malloc(width * height * sizeof(int));
    // }
    // MPI_Gather(send,10,MPI_INT,image,10,MPI_INT,0,MPI_COMM_WORLD);
}
