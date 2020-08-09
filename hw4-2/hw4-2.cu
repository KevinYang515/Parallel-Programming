#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int INF = ((1 << 30) - 1);
void input(char* inFileName);
void output(char* outFileName);

int ceil(int a, int b);
__global__ void update_dist(int n, int* Dist, int* udapted_dist, int R, int B);
__global__ void phase1(int round, int n, int* Dist, int B);
__global__ void phase2(int round, int n, int* Dist, int B);
__global__ void phase3(int round, int n, int* Dist, int B, int R, int rank);

// Shared Memory
extern __shared__ int SM[];

int n, m;
int *d_Dist_0, *d_Dist_1, *d_n;

// Distance Matrix (Global Memory)
int *Dist;

int main(int argc, char* argv[]) {

    input(argv[1]);
    int B = 32;
    int updated_dist[B * n];
    int round = ceil(n, B);
    
    #pragma omp parallel num_threads(2)
    {
        int thread_num = omp_get_thread_num();
        cudaSetDevice(thread_num);

        if(thread_num == 0){
            cudaMalloc((void **)&d_Dist_0, n * n * sizeof(int));
            cudaMemcpy(d_Dist_0, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);
        }else if(thread_num == 1){
            cudaMalloc((void **)&d_Dist_1, n * n * sizeof(int));
            cudaMemcpy(d_Dist_1, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);
        }
            
        cudaMalloc((void **)&d_n, sizeof(int));
        cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);

        dim3 grid1(1, 1);
        dim3 grid2(round, 2);
        dim3 grid3((round/2)+1, round);
        dim3 blk(B, B);

        for (int r = 0; r < round; ++r) {
            #pragma omp barrier
            if(n > B && r < (round/2) && thread_num == 1){
                cudaMemcpyPeer((void*) &d_Dist_1[r * B * n], 1, (void*) &d_Dist_0[r * B * n], 0, B * n * sizeof(int));
  
            }else if(n > B && r >= (round/2) && thread_num == 0){
                if(r == (round-1))
                    cudaMemcpyPeer((void*) &d_Dist_0[r * B * n], 0, (void*) &d_Dist_1[r * B * n], 1, (n - r * B) * n * sizeof(int));
                else
                    cudaMemcpyPeer((void*) &d_Dist_0[r * B * n], 0, (void*) &d_Dist_1[r * B * n], 1, B * n * sizeof(int));
            }
            #pragma omp barrier
            
            if(thread_num == 0){
                phase1<<<grid1, blk, B*B*sizeof(int)>>>(r, n, d_Dist_0, B);
                phase2<<<grid2, blk, 2*B*B*sizeof(int)>>>(r, n, d_Dist_0, B);
                phase3<<<grid3, blk, 2*B*B*sizeof(int)>>>(r, n, d_Dist_0, B, round, thread_num);
            }else if(thread_num == 1 && n > B){
                phase1<<<grid1, blk, B*B*sizeof(int)>>>(r, n, d_Dist_1, B);
                phase2<<<grid2, blk, 2*B*B*sizeof(int)>>>(r, n, d_Dist_1, B);
                phase3<<<grid3, blk, 2*B*B*sizeof(int)>>>(r, n, d_Dist_1, B, round, thread_num);
            }
        }
        #pragma omp barrier

        if(thread_num == 0)
            cudaMemcpy(Dist, d_Dist_0, (round/2) * B * n * sizeof(int), cudaMemcpyDeviceToHost);
        else if(n > B && thread_num == 1)
            cudaMemcpy(&Dist[(round/2) * B * n], &d_Dist_1[(round/2) * B * n], (n - (round/2) * B) * n * sizeof(int), cudaMemcpyDeviceToHost);

    }

    output(argv[2]);

    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    Dist = (int*)malloc(n*n*sizeof(int));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i * n + j] = 0;
            } else {
                Dist[i * n + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i * n + j] >= INF) Dist[i * n + j] = INF;
        }
    }
    fwrite(Dist, sizeof(int), n*n, outfile);
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void update_dist(int n, int* Dist, int* udapted_dist, int R, int B){

    int row = blockIdx.x;

    if((R * B + row) >= n) return;

    for (int j = 0; j < n; ++j)
        Dist[R * B * n + row * n + j] = udapted_dist[row * n + j];
        // udapted_dist[0] = Dist[R];
}


__global__ void phase1(int round, int n, int* Dist, int B){
    int shared_i = threadIdx.y;
    int shared_j = threadIdx.x;
    int i = round * B + shared_i;
    int j = round * B + shared_j;

    // Copy From Global Memory To Shared Memory
    if (i < n && j < n){
        SM[shared_i * B + shared_j] = Dist[i * n + j];
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
        Dist[i * n + j] = SM[s_temp + shared_j];
    }
}

__global__ void phase2(int round, int n, int* Dist, int B){

    if (blockIdx.x == round) return;

    int* pivot = &SM[0];
    int* S_dist = &SM[B * B];

    int shared_i = threadIdx.y;
    int shared_j = threadIdx.x;

    int i = round * B + shared_i;
    int j = round * B + shared_j;

    int s_temp = shared_i * B;

    if (i < n && j < n){
        pivot[s_temp + shared_j] = Dist[i * n + j];
    }
    __syncthreads();

    if (blockIdx.y == 0){
        j = blockIdx.x * B + shared_j;
    }else{
        i = blockIdx.x * B + shared_i;
    }

    if (i >= n || j >= n) return;

    S_dist[s_temp + shared_j] = Dist[i * n + j];
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
        Dist[i * n + j] = S_dist[s_temp + shared_j];
    }
}

__global__ void phase3(int round, int n, int* Dist, int B, int R, int thread_num){

    int blk_i = blockIdx.x;
    int blk_j = blockIdx.y;

    if(thread_num == 1) blk_i += (R/2);

    if (blk_i == round || blk_j == round) return;

    int* pivot_row = &SM[0];
    int* pivot_col = &SM[B * B];

    int shared_i = threadIdx.y;
    int shared_j = threadIdx.x;

    int i = blk_i * B + shared_i;
    int j = blk_j * B + shared_j;

    int block_i = round * B + shared_i;
    int block_j = round * B + shared_j;

    int s_temp = shared_i * B;

    if (i < n && block_j < n){
        pivot_row[s_temp + shared_j] = Dist[i * n + block_j];
    }

    if (j < n && block_i < n){
        pivot_col[s_temp + shared_j] = Dist[block_i * n + j];
    }
    __syncthreads();

    if (i >= n || j >= n) return;

    int distance = Dist[i * n + j];

    int t_temp = round * B;
    #pragma unroll
    for (int m = 0; m < B && t_temp + m < n; m++){
        if (pivot_row[s_temp + m] + pivot_col[m * B + shared_j] < distance){
            distance = pivot_row[s_temp + m] + pivot_col[m * B + shared_j];
        }
    }

    Dist[i * n + j] = distance;
}
