#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>


int is_final(int, int);
void swap(float *, float *);
int partition(float[], int, int); 
void quickSort(float[], int, int);
int is_equal(float[], float[], int, int*);
void mergeSort(float[], int, int, int);
void merging(float[], int, int, int, int);
void mergeArrays(float[], float[], int n1, int n2, float[]);


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

    float *data = malloc(bufsize_ * sizeof(float));
    float *data_ = malloc(bufsize_ * sizeof(float));
    float *data0 = malloc(bufsize * 2 * sizeof(float));
    float *data1 = malloc((bufsize + bufsize_) * sizeof(float));
    float *data_temp = malloc((bufsize + bufsize_) * sizeof(float));

    if(is_final(rank, size)){
        MPI_File_read_at(f, sizeof(float) * rank * bufsize, data, bufsize_, MPI_FLOAT, MPI_STATUS_IGNORE);
        quickSort(data, 0, bufsize_ - 1);
        // mergeSort(data, bufsize_, 0, bufsize_ - 1);
    }                 
    else{
        MPI_File_read_at(f, sizeof(float) * rank * bufsize, data, bufsize, MPI_FLOAT, MPI_STATUS_IGNORE);
        quickSort(data, 0, bufsize - 1);
        // mergeSort(data, bufsize, 0, bufsize - 1);
    }

    MPI_File_close(&f);

    int sorted_all;

    while(sorted_all != size){
    // for(int _ = 0; _ < 3; _++){
        int _ = 0;
        sorted_all = 0;
        sorted = 1;

        for(int i = 0; i < 2; i++){
            for(int j = i; j < size - 1; j += 2){
                if(rank == j){
                    if(is_final(rank + 1, size)){
                        // memcpy(data1, data, bufsize * sizeof(float));

                        // MPI_Recv(data1 + bufsize, bufsize_, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        
                        // mergeSort(data1, bufsize + bufsize_, 0, bufsize + bufsize_ - 1);
                        // quickSort(data1, 0, bufsize + bufsize_ - 1);


                        MPI_Recv(data_, bufsize_, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        mergeArrays(data, data_, bufsize, bufsize_, data1);
                        

                        MPI_Send(data1 + bufsize, bufsize_, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                        is_equal(data, data1, bufsize, &sorted);
                        memcpy(data, data1, bufsize * sizeof(float));
                    }else{
                        // memcpy(data0, data, bufsize * sizeof(float));

                        // MPI_Recv(data0 + bufsize, bufsize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        
                        // mergeSort(data0, bufsize * 2, 0, bufsize * 2 - 1);
                        // quickSort(data0, 0, bufsize * 2 - 1);


                        MPI_Recv(data_, bufsize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        mergeArrays(data, data_, bufsize, bufsize, data0);


                        MPI_Send(data0 + bufsize, bufsize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                        is_equal(data, data0, bufsize, &sorted);
                        memcpy(data, data0, bufsize * sizeof(float));
                    }
                }

                if(rank == j + 1){
                    if(is_final(rank, size)){
                        memcpy(data1, data, bufsize_ * sizeof(float));
                        MPI_Send(data, bufsize_, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
                        MPI_Recv(data, bufsize_, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        is_equal(data1, data, bufsize_, &sorted);
                    }
                    else{
                        memcpy(data0, data, bufsize * sizeof(float));
                        MPI_Send(data, bufsize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
                        MPI_Recv(data, bufsize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        is_equal(data0, data, bufsize, &sorted);
                    }
                }
            }            
        }
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Reduce(&sorted, &sorted_all, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        // MPI_Gather(sorted, 1, MPI_INT, sorted_all, size, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Bcast(&sorted_all, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // if(rank == 0){
        //     printf("\n Sorted_all: %d", sorted_all);
        //     printf("\nRound: %d, Rank: %d, Sorted: %d", _, rank, sorted_all);
        // }
        
        // if(rank == 1){
        //     printf("\nRound: %d, Rank: %d, Sorted: %d", _, rank, sorted_all);
        // }
        // if(rank == 2){
        //     printf("\nRound: %d, Rank: %d, Sorted: %d", _, rank, sorted_all);
        // }
        
        // if(rank == 0){
        //     for(int m = 0; m < bufsize; m++){
        //         printf("\nRound: %d, Rank: %d, %f, %d", _, rank, data[m], sorted);
        //     }
        // }
        // if(rank == 1){
        //     for(int m = 0; m < bufsize; m++){
        //         printf("\nRound: %d, Rank: %d, %f, %d", _, rank, data[m], sorted);
        //     }
        // }
        // if(rank == 2){
        //     for(int m = 0; m < bufsize; m++){
        //         printf("\nRound: %d, Rank: %d, %f, %d", _, rank, data[m], sorted);
        //     }
        // }
    }

    
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f);

    if(is_final(rank, size)){
        MPI_File_write_at(f, sizeof(float) * rank * bufsize, data, bufsize_, MPI_FLOAT, MPI_STATUS_IGNORE);
    }else{
        MPI_File_write_at(f, sizeof(float) * rank * bufsize, data, bufsize, MPI_FLOAT, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&f);

    return 0;
}

int is_final(int rank, int size){
    if(rank == size-1)
       return 1;
    else
       return 0;
}

void swap(float *x, float *y){
    float t = *x;
    *x = *y;
    *y = t;
}

int partition(float number[], int left, int right) { 
    int i = left - 1; 
    int j;

    for(j = left; j < right; j++) { 
        if(number[j] <= number[right]) { 
            i++; 
            swap(&number[i], &number[j]);
        } 
    }

    swap(&number[i+1], &number[right]);

    return i+1; 
}

void quickSort(float number[], int left, int right) {
    if(left < right) { 
        int q = partition(number, left, right);
        quickSort(number, left, q-1);
        quickSort(number, q+1, right);
    } 
}

int is_equal(float x[], float y[], int len, int *sorted){
    for(int i = 0; i < len; i++){
        if(x[i] != y[i])
            *sorted = 0;
    }
}


//merge sort
void merging(float a[], int length, int low, int mid, int high) {
   int l1, l2, i;
   float *b = malloc(length * sizeof(float));
   
   for(l1 = low, l2 = mid + 1, i = low; l1 <= mid && l2 <= high; i++) {
      if(a[l1] <= a[l2])
         b[i] = a[l1++];
      else
         b[i] = a[l2++];
   }
   
   while(l1 <= mid)    
      b[i++] = a[l1++];

   while(l2 <= high)   
      b[i++] = a[l2++];

   for(i = low; i <= high; i++)
      a[i] = b[i];
}

void mergeSort(float a[], int length, int low, int high) {
   int mid;
   
   if(low < high) {
      mid = (low + high) / 2;
      mergeSort(a, length, low, mid);
      mergeSort(a, length, mid+1, high);
      merging(a, length, low, mid, high);
   } else { 
      return;
   }   
}

void mergeArrays(float arr1[], float arr2[], int n1, int n2, float arr3[]) 
{ 
    int i = 0, j = 0, k = 0; 
   
    while (i<n1 && j <n2) 
    { 
        if (arr1[i] < arr2[j]) 
            arr3[k++] = arr1[i++]; 
        else
            arr3[k++] = arr2[j++]; 
    } 
  
    while (i < n1) 
        arr3[k++] = arr1[i++]; 

    while (j < n2) 
        arr3[k++] = arr2[j++]; 
} 
