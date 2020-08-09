#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv) {
    std::ifstream f(argv[1]);

    float start;
    float end;
    
    float io_time = 0.0;
    float cpu_time = 0.0;

    int V;
    int E;

    start = omp_get_wtime();
    f.read((char*)&V, sizeof V);
    f.read((char*)&E, sizeof E);
    end = omp_get_wtime();

    io_time += end - start;

    int** matrix = (int**)malloc(V * sizeof(int*));

    for (int i = 0; i<V; i++){
        matrix[i] = (int*)malloc(V * sizeof(int));
    }

    int* e_info = (int*)malloc(3 * E * sizeof(int));

    start = omp_get_wtime();
    f.read((char*)e_info, 3 * E * sizeof(int));
    
    f.close();
    end = omp_get_wtime();

    io_time += end - start;

    start = omp_get_wtime();

    //Initialize matrix for edge weight
    #pragma omp parallel for
    for (int i=0; i<V; i++){
        for(int j=0; j<V; j++){
            matrix[i][j] = 1073741823;
        }
        matrix[i][i] = 0;
    }

    //Give the edge wieght 
    for (int i=0; i < E * 3; i+=3){
        matrix[e_info[i]][e_info[i+1]] = e_info[i+2];
    }

    //Floyd Warshall
    for (int i=0; i<V; i++){
        #pragma omp parallel for schedule(static, 10)
        for (int j=0; j<V; j++){
            for (int k=0; k<V; k++){
                if(matrix[j][k] > (matrix[j][i] + matrix[i][k]))
                    matrix[j][k] = matrix[j][i] + matrix[i][k];
            }
        }
    }

    end = omp_get_wtime();

    cpu_time += end - start;
    printf("CPU_time: %f\n", cpu_time);

    start = omp_get_wtime();
    std::ofstream fout(argv[2], std::ios::out | std::ios::binary);
    for (int i = 0; i < V; i++) {
        fout.write(reinterpret_cast<const char*>(matrix[i]),V * sizeof(int));
    }
    fout.close();
    end = omp_get_wtime();

    io_time += end - start;
    printf("IO_time: %f\n", io_time);
}