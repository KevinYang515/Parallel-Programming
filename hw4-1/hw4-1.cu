#include <stdio.h>
#include <stdlib.h>

const int INF = ((1 << 30) - 1);
const int V = 22000;

int n, m;
int *d_Dist, **d_Dist_internal;
int *d_n, *d_m;

int ceil(int x, int y) {return (x + y - 1) / y;}
__global__ void phase1(int round, int n, int V, int* Dist, int B);
__global__ void phase2(int round, int n, int V, int* Dist, int B);
__global__ void phase3(int round, int n, int V, int* Dist, int B);

// Shared Memory
extern __shared__ int SM[];

// Distance Matrix (Global Memory)
int Dist[V][V];

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int main(int argc, char* argv[]){
    input(argv[1]);

    // Initialize for two dimension array (faster than initialize pointer)
    cudaMalloc((void **)&d_Dist, V * V * sizeof(int));

    cudaMalloc((void **)&d_n, sizeof(int));
    cudaMalloc((void **)&d_m, sizeof(int));

    cudaMemcpy(d_Dist, Dist, V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, &m, sizeof(int), cudaMemcpyHostToDevice);

    int thread_in_bank = 32; //1024 = 32 * 32
    int num_round = ceil(n, thread_in_bank);

    dim3 first(1);
    dim3 second(num_round , 2);
    dim3 third(num_round, num_round);
    dim3 bk(thread_in_bank, thread_in_bank);

    // int num_threads = 32;

    for (int rou = 0; rou < num_round; rou ++){
        phase1<<<first, bk, thread_in_bank * thread_in_bank * sizeof(int)>>>(rou, n, V, d_Dist, thread_in_bank);
        phase2<<<second, bk, 2 * thread_in_bank * thread_in_bank *sizeof(int)>>>(rou, n, V, d_Dist, thread_in_bank);
        phase3<<<third, bk, 2 * thread_in_bank * thread_in_bank * sizeof(int)>>>(rou, n, V, d_Dist, thread_in_bank);
    }

    cudaMemcpy(Dist, d_Dist, V * V * sizeof(int), cudaMemcpyDeviceToHost);
    output(argv[2]);

    return 0;
}

__global__ void phase1(int round, int n, int V, int* Dist, int B){
    int shared_i = threadIdx.y;
    int shared_j = threadIdx.x;
    int i = round * B + shared_i;
    int j = round * B + shared_j;

    // Copy From Global Memory To Shared Memory
    if (i < n && j < n){
        SM[shared_i * B + shared_j] = Dist[i * V + j];
    }
    __syncthreads();

    int t_temp = round * B;
    int s_temp = shared_i * B;

    #pragma unroll
    for (int m = 0; m < B && t_temp + m < n; m++){
        if (SM[s_temp + m] + SM[m * B + shared_j] < SM[s_temp + shared_j]){
            SM[s_temp + shared_j] = SM[s_temp + m] + SM[m * B + shared_j];
        }

        __syncthreads();
    }

    // Copy Back To Global Memory
    if (i < n && j < n){
        Dist[i * V + j] = SM[s_temp + shared_j];
    }
}

__global__ void phase2(int round, int n, int V, int* Dist, int B){
    if (blockIdx.x == round) return;

    int* pivot = &SM[0];
    int* S_dist = &SM[B * B];

    int shared_i = threadIdx.y;
    int shared_j = threadIdx.x;

    int i = round * B + shared_i;
    int j = round * B + shared_j;

    int s_temp = shared_i * B;

    if (i < n && j < n){
        pivot[s_temp + shared_j] = Dist[i * V + j];
    }
    __syncthreads();

    if (blockIdx.y == 0){
        j = blockIdx.x * B + shared_j;
    }else{
        i = blockIdx.x * B + shared_i;
    }

    if (i >= n || j >= n) return;

    S_dist[s_temp + shared_j] = Dist[i * V + j];
    __syncthreads();

    int t_temp = round * B;
    
    if (blockIdx.y == 1){
        #pragma unroll
        for (int m = 0; m < B && t_temp + m < n; m++){
            if (S_dist[s_temp + m] + pivot[m * B + shared_j] < S_dist[s_temp + shared_j]){
                S_dist[s_temp + shared_j] = S_dist[s_temp + m] + pivot[m * B + shared_j];
            }
        }
    }else{
        #pragma unroll
        for (int m = 0; m < B && t_temp + m < n; m++){
            if (pivot[s_temp + m] + S_dist[m * B + shared_j] < S_dist[s_temp + shared_j]){
                S_dist[s_temp + shared_j] = pivot[s_temp + m] + S_dist[m * B + shared_j];
            }
        }
    }

    if (i < n && j < n){
        Dist[i * V + j] = S_dist[s_temp + shared_j];
    }
}

__global__ void phase3(int round, int n, int V, int* Dist, int B){
    if (blockIdx.x == round || blockIdx.y == round) return;

    int* pivot_row = &SM[0];
    int* pivot_col = &SM[B * B];

    int shared_i = threadIdx.y;
    int shared_j = threadIdx.x;

    int i = blockIdx.y * B + shared_i;
    int j = blockIdx.x * B + shared_j;

    int block_i = round * B + shared_i;
    int block_j = round * B + shared_j;

    int s_temp = shared_i * B;

    if (i < n && block_j < n){
        pivot_row[s_temp + shared_j] = Dist[i * V + block_j];
    }

    if (j < n && block_i < n){
        pivot_col[s_temp + shared_j] = Dist[block_i * V + j];
    }
    __syncthreads();

    if (i >= n || j >= n) return;

    int distance = Dist[i * V + j];

    int t_temp = round * B;
    #pragma unroll
    for (int m = 0; m < B && t_temp + m < n; m++){
        if (pivot_row[s_temp + m] + pivot_col[m * B + shared_j] < distance){
            distance = pivot_row[s_temp + m] + pivot_col[m * B + shared_j];
        }
    }

    Dist[i * V + j] = distance;
}
