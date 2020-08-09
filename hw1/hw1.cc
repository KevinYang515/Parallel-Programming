#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

int is_final_(int, int);
// void swap(float *, float *);
// int partition(float[], int, int); 
// void quickSort(float[], int, int);
void is_equal(float[], float[], int, int*);
// void mergeSort(float[], int, int, int);
// void merging(float[], int, int, int, int);
// void mergeArrays(float[], float[], int n1, int n2, float[]);
void mergeArrays0(float[], float[], int , int , float[]);
void mergeArrays1(float[], float[], int , int , float[]);

using namespace std;
int main(int argc, char** argv) {
    int rank, size, bufsize, bufsize_;
    int sorted = 0;
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_File f;

    MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
    long long filesize = atoll(argv[1]);
    bufsize = filesize / size;
    bufsize_ = bufsize + (filesize % size);

    double read_start = 0.0;
    double read_end = 0.0;

    double process_start = 0.0;
    double process_end = 0.0;

    double write_start = 0.0;
    double write_end = 0.0;

    double CPU = 0.0;
    double COMM = 0.0;
    double IO = 0.0;

    double start = 0.0;
    double end = 0.0;

    float *data = (float*)malloc(bufsize_ * sizeof(float));
    float *data_ = (float*)malloc(bufsize_ * sizeof(float));
    float *data0 = (float*)malloc(bufsize * 2 * sizeof(float));
    float *data1 = (float*)malloc((bufsize + bufsize_) * sizeof(float));
    // float *data_temp = (float*)malloc((bufsize + bufsize_) * sizeof(float));

    if(is_final_(rank, size)){
        start = MPI_Wtime();
        MPI_File_read_at(f, sizeof(float) * rank * bufsize, data, bufsize_, MPI_FLOAT, MPI_STATUS_IGNORE);
        end = MPI_Wtime();

        IO += end - start;

        start = MPI_Wtime();
        sort(data, data+bufsize_);
        end = MPI_Wtime();

        CPU += end - start;

		// quickSort(data, 0, bufsize_ - 1);
        // mergeSort(data, bufsize_, 0, bufsize_ - 1);
    }                 
    else{
        start = MPI_Wtime();
        MPI_File_read_at(f, sizeof(float) * rank * bufsize, data, bufsize, MPI_FLOAT, MPI_STATUS_IGNORE);
        end = MPI_Wtime();

        IO += end - start;

        start = MPI_Wtime();
        sort(data, data+bufsize);
        end = MPI_Wtime();

        CPU += end - start;
		// quickSort(data, 0, bufsize - 1);
        // mergeSort(data, bufsize, 0, bufsize - 1);
    }


    

    MPI_File_close(&f);

    int sorted_all;

    int pp = 0;


    while(pp != size - 1){
    // while(sorted_all != size){
        sorted_all = 0;
        sorted = 1;

        for(int i = 0; i < 2; i++){
            for(int j = i; j < size - 1; j += 2){
                if(rank == j){
                    if(is_final_(rank + 1, size)){
                        start = MPI_Wtime();
                        MPI_Sendrecv(data, bufsize, MPI_FLOAT, rank + 1, 0, data_, bufsize_, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        end = MPI_Wtime();

                        COMM += end - start;

                        start = MPI_Wtime();
                        mergeArrays0(data, data_, bufsize, bufsize_, data1);
                        end = MPI_Wtime();

                        CPU += end - start;

                        // MPI_Recv(data_, bufsize_, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        // mergeArrays(data, data_, bufsize, bufsize_, data1);
                        
                        // MPI_Send(data1 + bufsize, bufsize_, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                        if(size>3){
                            if(pp < size - 3){
                                start = MPI_Wtime();
                                is_equal(data, data1, bufsize, &sorted);
                                end = MPI_Wtime();

                                CPU += end - start;
                            }
                        }
                        memcpy(data, data1, bufsize * sizeof(float));
                    }else{
                        start = MPI_Wtime();
                        MPI_Sendrecv(data, bufsize, MPI_FLOAT, rank + 1, 0, data_, bufsize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        end = MPI_Wtime();

                        COMM += end - start;

                        start = MPI_Wtime();
                        mergeArrays0(data, data_, bufsize, bufsize, data0);
                        end = MPI_Wtime();

                        CPU += end - start;

                        // MPI_Recv(data_, bufsize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        // mergeArrays(data, data_, bufsize, bufsize, data0);

                        // MPI_Send(data0 + bufsize, bufsize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                        if(size>3){
                            if(pp < size - 3){
                                start = MPI_Wtime();
                                is_equal(data, data0, bufsize, &sorted);
                                end = MPI_Wtime();

                                CPU += end - start;
                            }
                        }

                        start = MPI_Wtime();
                        memcpy(data, data0, bufsize * sizeof(float));
                        end = MPI_Wtime();

                        CPU += end - start;
                    }
                }

                if(rank == j + 1){
                    if(is_final_(rank, size)){
                        start = MPI_Wtime();
                        MPI_Sendrecv(data, bufsize_, MPI_FLOAT, rank - 1, 0, data_, bufsize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        end = MPI_Wtime();
                        COMM += end - start;

                        start = MPI_Wtime();
                        mergeArrays1(data, data_, bufsize_, bufsize, data1);
                        end = MPI_Wtime();
                        CPU += end - start;

                        // mergeArrays(data, data_, bufsize_, bufsize, data1);


                        // memcpy(data1, data, bufsize_ * sizeof(float));
                        // MPI_Send(data, bufsize_, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
                        // MPI_Recv(data, bufsize_, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        // is_equal(data1, data, bufsize_, &sorted);

                        if(size>3){
                            if(pp < size - 3){
                                start = MPI_Wtime();
                                is_equal(data1, data, bufsize_, &sorted);
                                end = MPI_Wtime();
                                CPU += end - start;
                            }
                        }
                        start = MPI_Wtime();
                        memcpy(data, data1, bufsize_ * sizeof(float));
                        end = MPI_Wtime();
                        CPU += end - start;
                    }
                    else{
                        start = MPI_Wtime();
                        MPI_Sendrecv(data, bufsize, MPI_FLOAT, rank - 1, 0, data_, bufsize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        end = MPI_Wtime();
                        COMM += end - start;

                        start = MPI_Wtime();
                        mergeArrays1(data, data_, bufsize, bufsize, data0);
                        end = MPI_Wtime();
                        CPU += end - start;
                        // mergeArrays(data, data_, bufsize, bufsize, data0);


                        // memcpy(data0, data, bufsize * sizeof(float));
                        // MPI_Send(data, bufsize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
                        // MPI_Recv(data, bufsize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        // is_equal(data0, data, bufsize, &sorted);

                        if(size>3){
                            if(pp < size - 3){
                                start = MPI_Wtime();
                                is_equal(data0, data, bufsize, &sorted);
                                end = MPI_Wtime();
                                CPU += end - start;
                            }
                        }
                        start = MPI_Wtime();
                        memcpy(data, data0, bufsize * sizeof(float));
                        end = MPI_Wtime();
                        CPU += end - start;
                    }
                }
            }            
        }
        

        if(size>4){
            if(pp < size - 4){
                start = MPI_Wtime();
				MPI_Allreduce(&sorted, &sorted_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                end = MPI_Wtime();
                COMM += end - start;

                if(sorted_all == size){
                    break;
                }
            }
        }

        // MPI_Barrier(MPI_COMM_WORLD);

        // MPI_Reduce(&sorted, &sorted_all, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // MPI_Bcast(&sorted_all, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // MPI_Barrier(MPI_COMM_WORLD);


		// if(size > 3){
		// 	if(size0 < size - 3){
		// 		MPI_Barrier(MPI_COMM_WORLD);

		// 		MPI_Reduce(&sorted, &sorted_all, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

		// 		MPI_Bcast(&sorted_all, 1, MPI_INT, 0, MPI_COMM_WORLD);
		// 		MPI_Barrier(MPI_COMM_WORLD);
		// 	}
		// }else{
		// 	MPI_Barrier(MPI_COMM_WORLD);

		// 	MPI_Reduce(&sorted, &sorted_all, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

		// 	MPI_Bcast(&sorted_all, 1, MPI_INT, 0, MPI_COMM_WORLD);
		// 	MPI_Barrier(MPI_COMM_WORLD);
		// }
		// size0 ++;

        // if(rank == 0){
        //     for(int i = 0; i < bufsize; i++){
        //         printf("\nRound: %d, Rank0: %f", pp, data[i]);
        //     }
        // }
        // if(rank == 1){
        //     for(int i = 0; i < bufsize; i++){
        //         printf("\nRound: %d, Rank1: %f", pp, data[i]);
        //     }
        // }
        // if(rank == 2){
        //     for(int i = 0; i < bufsize; i++){
        //         printf("\nRound: %d, Rank2: %f", pp, data[i]);
        //     }
        // }
        // if(rank == 3){
        //     for(int i = 0; i < bufsize_; i++){
        //         printf("\nRound: %d, Rank3: %f", pp, data[i]);
        //     }
        // }
        pp++;
        

    }


    start = MPI_Wtime();
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f);

    if(is_final_(rank, size)){
        MPI_File_write_at(f, sizeof(float) * rank * bufsize, data, bufsize_, MPI_FLOAT, MPI_STATUS_IGNORE);
    }else{
        MPI_File_write_at(f, sizeof(float) * rank * bufsize, data, bufsize, MPI_FLOAT, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&f);
    end = MPI_Wtime();
    IO += end - start;

    // printf("Rank: %d, IO time: %f\n", rank, IO);
    // printf("Rank: %d, CPU time: %f\n", rank, CPU);
    // printf("Rank: %d, COMM time: %f\n", rank, COMM);

    printf("total time: %f\n", IO + CPU + COMM);

    return 0;
}

int is_final_(int rank, int size){
    if(rank == size-1)
       return 1;
    else
       return 0;
}

// void swap(float *x, float *y){
//     float t = *x;
//     *x = *y;
//     *y = t;
// }

// int partition(float number[], int left, int right) { 
//     int i = left - 1; 
//     int j;

//     for(j = left; j < right; j++) { 
//         if(number[j] <= number[right]) { 
//             i++; 
//             swap(&number[i], &number[j]);
//         } 
//     }

//     swap(&number[i+1], &number[right]);

//     return i+1; 
// }

// void quickSort(float number[], int left, int right) {
//     if(left < right) { 
//         int q = partition(number, left, right);
//         quickSort(number, left, q-1);
//         quickSort(number, q+1, right);
//     } 
// }

void is_equal(float x[], float y[], int len, int *sorted){
    for(int i = 0; i < len; i++){
        if(x[i] != y[i])
            *sorted = 0;
    }
}


// //merge sort
// void merging(float a[], int length, int low, int mid, int high) {
//    int l1, l2, i;
//    float *b = (float*)malloc(length * sizeof(float));
   
//    for(l1 = low, l2 = mid + 1, i = low; l1 <= mid && l2 <= high; i++) {
//       if(a[l1] <= a[l2])
//          b[i] = a[l1++];
//       else
//          b[i] = a[l2++];
//    }
   
//    while(l1 <= mid)    
//       b[i++] = a[l1++];

//    while(l2 <= high)   
//       b[i++] = a[l2++];

//    for(i = low; i <= high; i++)
//       a[i] = b[i];
// }

// void mergeSort(float a[], int length, int low, int high) {
//    int mid;
   
//    if(low < high) {
//       mid = (low + high) / 2;
//       mergeSort(a, length, low, mid);
//       mergeSort(a, length, mid+1, high);
//       merging(a, length, low, mid, high);
//    } else { 
//       return;
//    }   
// }

// void mergeArrays(float arr1[], float arr2[], int n1, int n2, float arr3[]) 
// { 
//     int i = 0, j = 0, k = 0; 
   
//     while (i<n1 && j <n2) 
//     { 
//         if (arr1[i] < arr2[j]) 
//             arr3[k++] = arr1[i++]; 
//         else
//             arr3[k++] = arr2[j++]; 
//     } 
  
//     while (i < n1) 
//         arr3[k++] = arr1[i++]; 

//     while (j < n2) 
//         arr3[k++] = arr2[j++]; 
// }

void mergeArrays0(float a[], float b[], int n1, int n2, float temp[]){
    int i=0, j=0, k=0;

    while(i < n1 && j < n2 && k < n1){
        if(a[i] <= b[j]){
            temp[k] = a[i];
            i ++;
        }
        else{
            temp[k] = b[j];
            j++;
        }
        
        k ++;
    }
    
    while(i < n1 && k < n1){
        temp[k] = a[i];
        i ++;
        k ++;
    }
}

void mergeArrays1(float a[], float b[], int n1, int n2, float temp[]){
    int i=n1 -1 , j=n2 -1 , k=n1-1;

    while(i >= 0 && j >= 0 && k >= 0){
        if(a[i] >= b[j]){
            temp[k] = a[i];
            i --;
        }
        else{
            temp[k] = b[j];
            j --;
        }
        
        k --;
    }
    
    while(i >= 0 && k >= 0){
        temp[k] = a[i];
        i --;
        k --;
    }
}
