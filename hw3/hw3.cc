#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv) {
    std::ifstream f(argv[1]);

    int V;
    int E;

    f.read((char*)&V, sizeof V);
    f.read((char*)&E, sizeof E);

    int** matrix = (int**)malloc(V * sizeof(int*));

    for (int i = 0; i<V; i++){
        matrix[i] = (int*)malloc(V * sizeof(int));
    }

    int* e_info = (int*)malloc(3 * E * sizeof(int));
    f.read((char*)e_info, 3 * E * sizeof(int));
    
    f.close();

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
        #pragma omp parallel for
        for (int j=0; j<V; j++){
            for (int k=0; k<V; k++){
                if(matrix[j][k] > (matrix[j][i] + matrix[i][k]))
                    matrix[j][k] = matrix[j][i] + matrix[i][k];
            }
        }
    }


    std::ofstream fout(argv[2], std::ios::out | std::ios::binary);
    for (int i = 0; i < V; i++) {
        fout.write(reinterpret_cast<const char*>(matrix[i]),V * sizeof(int));
    }
    fout.close();
}