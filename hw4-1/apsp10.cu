#include <stdio.h>
#include <stdlib.h>

const int INF = ((1 << 30) - 1);
const int V = 20000;
void input(char* inFileName);
void output(char* outFileName);

int ceil(int a, int b);
__global__ void phase1(int round, int n, int V, int* Dist, int B);
__global__ void phase2(int round, int n, int V, int* Dist, int B);
__global__ void phase3(int round, int n, int V, int* Dist, int B);
extern __shared__ int S[];

int n, m;
int *d_Dist, **d_Dist_internal;
int *d_n, *d_m;
int Dist[V][V];

int main(int argc, char* argv[]) {
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);   
    
    input(argv[1]);
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) 
    //         printf("%d\n", Dist[i][j]);
    // }
    cudaMalloc((void **)&d_Dist, V * V * sizeof(int));

    cudaMalloc((void **)&d_n, sizeof(int));
    cudaMalloc((void **)&d_m, sizeof(int));

    cudaMemcpy(d_Dist, Dist, V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, &m, sizeof(int), cudaMemcpyHostToDevice);
    int B = 32;
    int round = ceil(n, B);

    dim3 grid1(1, 1);
    dim3 grid2(round, 2);
    dim3 grid3(round, round);
    dim3 blk(B, B);
    int num_threads = 32;
    for (int r = 0; r < round; ++r) {
        phase1<<<grid1, blk, B*B*sizeof(int)>>>(r, n, V, d_Dist, B);
        phase2<<<grid2, blk, 2*B*B*sizeof(int)>>>(r, n, V, d_Dist, B);
        phase3<<<grid3, blk, 2*B*B*sizeof(int)>>>(r, n, V, d_Dist, B);
    }
    
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaThreadSynchronize();

    
    cudaMemcpy(Dist, d_Dist, V * V * sizeof(int), cudaMemcpyDeviceToHost);
    output(argv[2]);
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) 
    //         printf("%d\n", Dist[i][j]);
    // }
    // float time;
    // cudaEventElapsedTime(&time, start, stop);
    // printf("total time: %f\n", time);
    return 0;
}

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

int ceil(int a, int b) { return (a + b - 1) / b; }


__global__ void phase1(int round, int n, int V, int* Dist, int B){
    int s_i = threadIdx.y;
    int s_j = threadIdx.x;
    int i = round * B + s_i;
    int j = round * B + s_j;
    
    if((i < n && j < n))
        S[s_i * B + s_j] = Dist[i * V + j];
    __syncthreads();

    int tt = round * B;
    int ss = s_i * B;
    #pragma unroll
    for (int k = 0; k < B && tt + k < n; ++k) {
        if (S[ss + k] + S[k * B + s_j] < S[ss + s_j])
            S[ss + s_j] = S[ss + k] + S[k * B + s_j];
        
        __syncthreads();
    }
    if (i < n && j < n) Dist[i * V + j] = S[ss + s_j];
    __syncthreads();

}

__global__ void phase2(int round, int n, int V, int* Dist, int B){

    if (blockIdx.x == round) return;

    int* S_pivot = &S[0];
    int* S_dist = &S[B * B];

    int s_i = threadIdx.y;
    int s_j = threadIdx.x;
    int i = round * B + s_i;
    int j = round * B + s_j;
    
    int ss = s_i * B;

    if((i < n && j < n))
        S_pivot[ss + s_j] = Dist[i * V + j];
    __syncthreads();

    if (blockIdx.y == 0)
        j = blockIdx.x * B + s_j;
    else
        i = blockIdx.x * B + s_i;

    if (i >= n || j >= n) return;

    if((i < n && j < n))
        S_dist[ss + s_j] = Dist[i * V + j];
    __syncthreads();

    int tt = round * B;
    if(blockIdx.y == 1){
        #pragma unroll
        for (int k = 0; k < B && tt + k < n; ++k) {
            if (S_dist[ss + k] + S_pivot[k * B + s_j] < S_dist[ss + s_j])
                S_dist[ss + s_j] = S_dist[ss + k] + S_pivot[k * B + s_j];
        }
    }else{
        #pragma unroll
        for (int k = 0; k < B && tt + k < n; ++k) {
            if (S_pivot[ss + k] + S_dist[k * B + s_j] < S_dist[ss + s_j])
                S_dist[ss + s_j] = S_pivot[ss + k] + S_dist[k * B + s_j];
        }
    }
    
    if (i < n && j < n) Dist[i * V + j] = S_dist[ss + s_j];
    __syncthreads();
}

__global__ void phase3(int round, int n, int V, int* Dist, int B){

    if (blockIdx.x == round || blockIdx.y == round) return;

    int* S_pivot_row = &S[0];
    int* S_pivot_col= &S[B * B];

    int s_i = threadIdx.y;
    int s_j = threadIdx.x;
    int i = blockIdx.y * B + s_i;
    int j = blockIdx.x * B + s_j;
    int b_i = round * B + s_i;
    int b_j = round * B + s_j;

    int ss = s_i * B;
    
    if(i < n && b_j < n) S_pivot_row[ss + s_j] = Dist[i * V + b_j];
    if(j < n && b_i < n) S_pivot_col[ss + s_j] = Dist[b_i * V + j];
    __syncthreads();

    if (i >= n || j >= n) return;

    int dst = Dist[i * V + j];

    int tt = round * B;
    #pragma unroll
    for (int k = 0; k < B && tt + k < n; ++k) {
        if (S_pivot_row[ss + k] + S_pivot_col[k * B + s_j] < dst)
            dst = S_pivot_row[ss + k] + S_pivot_col[k * B + s_j];
    }
    
    
    Dist[i * V + j] = dst;

}
