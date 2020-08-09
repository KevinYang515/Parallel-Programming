#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void swap(float* p, float* q);
void oddeven(float* arr, int len);

int main(int argc, char** argv) {
	int rank, size, bufsize, bufsize3;
  
  MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_File f;
  // MPI_Offset filesize;

  MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
  // MPI_File_get_size(f, &filesize);

  // filesize = filesize/sizeof(float);
  long long filesize = atoll(argv[1]);

  if(size == 4){
    bufsize = filesize/size;
    bufsize3 = filesize/size + filesize%size;

    float data[bufsize3];

    if(rank != 3){
      MPI_File_read_at(f, sizeof(float) * rank * bufsize, &data, bufsize, MPI_FLOAT, MPI_STATUS_IGNORE);
      
      oddeven(data, bufsize);
    }
    else{
      MPI_File_read_at(f, sizeof(float) * rank * bufsize, &data, bufsize3, MPI_FLOAT, MPI_STATUS_IGNORE);
      
      oddeven(data, bufsize3);
    }

    MPI_File_close(&f);

    float d[bufsize3];
    float new_data0[bufsize + bufsize3]; 

    if(rank == 0){
      memcpy(new_data0, data, bufsize * sizeof(float));
      MPI_Recv(&d, bufsize, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      memcpy(new_data0 + bufsize, d, bufsize * sizeof(float));

      oddeven(new_data0, bufsize * 2);
    }else if(rank == 2){
      memcpy(new_data0, data, bufsize * sizeof(float));
      MPI_Recv(d, bufsize3, MPI_FLOAT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      memcpy(new_data0 + bufsize, d, bufsize3 * sizeof(float));

      oddeven(new_data0, bufsize + bufsize3);
    }else if(rank == 1){
      memcpy(&d, &data, sizeof(float) * bufsize);
      MPI_Send(&d, bufsize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }else if(rank == 3){
      memcpy(&d, &data, sizeof(float) * bufsize3);
      MPI_Send(&d, bufsize3, MPI_FLOAT, 2, 0, MPI_COMM_WORLD);
    }

    float new_data[bufsize*3 + bufsize3];

    if(rank==0){
      memcpy(new_data, new_data0, sizeof new_data0);
      MPI_Recv(&new_data0, bufsize + bufsize3, MPI_FLOAT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      memcpy(new_data + bufsize * 2, new_data0, sizeof new_data0);

      oddeven(new_data, bufsize * 3 + bufsize3);

      MPI_File_open(MPI_COMM_SELF, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f);
      MPI_File_write(f, new_data, bufsize*3 + bufsize3, MPI_FLOAT, MPI_STATUS_IGNORE);
      MPI_File_close(&f);
    }else if(rank ==2){
      MPI_Send(&new_data0, bufsize + bufsize3, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
  }else if(size == 3){

  }

  return 0;
}


void swap(float* p, float* q){
    float temp = *p;
    *p = *q;
    *q = temp;
}

void oddeven(float arr[], int len){
    int sort = 1;
    int temp;

    while(sort){
        sort = 0;
        for(int i = 0; i < 2; i ++){
            for(int j = i; j < len - 1; j += 2){
                if(arr[j] > arr[j+1]){
                    swap(&arr[j], &arr[j+1]);
                    sort = 1;
                }
            }
        }
    }
}