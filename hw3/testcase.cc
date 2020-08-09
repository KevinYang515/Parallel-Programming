#include <cstdio>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    // number of vertex and edge
    int v_n = atoi(argv[2]);
    int e_n = atoi(argv[3]);

    int** e = (int**)malloc(v_n*sizeof(int*));
    for(int i=0; i<v_n; i++)
        e[i] = (int*)malloc(v_n*sizeof(int));

    
    MPI_File f;
	MPI_File_open(MPI_COMM_SELF, argv[1], MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &f);
    MPI_File_write(f, &v_n, sizeof(int), MPI_BYTE, MPI_STATUS_IGNORE);
    MPI_File_write(f, &e_n, sizeof(int), MPI_BYTE, MPI_STATUS_IGNORE);

    int r = 0;

    // random generating edge with weight
    for(int i=0; i<v_n; i++){
        for(int j=0; j<v_n; j++){
            if(i == j)  e[i][j] = -1;
            else{
                if(rand()%101<=70){
                    e[i][j] = rand()%1001;
                    r++;
                }else e[i][j] = -1;
            };
            
            if(r>e_n) break;
            if(e[i][j] != -1){
                MPI_File_write(f, &i, sizeof(int), MPI_BYTE, MPI_STATUS_IGNORE);
                MPI_File_write(f, &j, sizeof(int), MPI_BYTE, MPI_STATUS_IGNORE);
                MPI_File_write(f, &e[i][j], sizeof(int), MPI_BYTE, MPI_STATUS_IGNORE);
            }   
        }
        if(r>e_n) break;
    }

    MPI_File_close(&f);

    // read and print binary file
    int* b = (int*)malloc((e_n*3*2)*sizeof(int));
    MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
    MPI_File_read(f, b, sizeof(int)*(e_n*3*2), MPI_INT, MPI_STATUS_IGNORE);

    int end = e_n*3+2;
    
    printf("v: %d\n", b[0]);
    printf("e: %d\n", b[1]);
    for(int i=2; i<end; i+=3)
        printf("s, d, w: %d %d %d\n", b[i],  b[i+1], b[i+2]);
    
    
    MPI_File_close(&f);
	MPI_Finalize();
}