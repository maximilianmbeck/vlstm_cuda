#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <random>

#define abs(x) (((x) > 0) ? (x) : (-x))

// The warp-level matrix multiply accumulate.
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

extern "C" __global__ void matmul_bf16(const __nv_bfloat16 *A,
                                       const __nv_bfloat16 *B, float *C,
                                       __nv_bfloat16 *C_b, int M, int N,
                                       int K) {
  // Define fragment objects for A, B, and C matrices.
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16,
                 wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

  // Initialize the output fragment
  wmma::fill_fragment(acc_frag, 0.0f);

  // Compute thread's row and column index in C.
  int row = WMMA_M * (threadIdx.y + blockDim.y * blockIdx.y);
  int warpCol = WMMA_N * ((threadIdx.x + blockDim.x * blockIdx.x) / warpSize);

  // Each thread computes 16x16 result, so check the boundaries.
  if (row < M && warpCol < N) {
    // Loop over the K dimension.
    for (int k = 0; k < K; k += WMMA_K) {
      // Load the A and B fragments from global memory.
      wmma::load_matrix_sync(a_frag, A + row * K + k, K);
      wmma::load_matrix_sync(b_frag, B + warpCol * K + k, K);

      // Perform the matrix multiplication and accumulate the result.
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store the resulting fragment to global memory.
    wmma::store_matrix_sync(C + row * N + warpCol, acc_frag, N,
                            wmma::mem_row_major);

    for (int i = 0; i < 8; i++) {
      C_b[(row + (i + 8 * (threadIdx.x / 16))) * N + warpCol +
          (threadIdx.x % 16)] =
          __float2bfloat16(C[(row + (i + 8 * (threadIdx.x / 16))) * N +
                             warpCol + (threadIdx.x % 16)]);
    }
  }
}

void simple_matmul_bf16(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                        float *C, __nv_bfloat16 *C_b, int M, int N, int K) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      C[m * N + n] = 0.f;
      for (int k = 0; k < K; k++) {
        C[m * N + n] +=
            __bfloat162float(A[m * K + k]) * __bfloat162float(B[n * K + k]);
      }
      C_b[m * N + n] = __float2bfloat16(C[m * N + n]);
    }
  }
}

bool check_equal(const __nv_bfloat16 *A, const __nv_bfloat16 *B, uint size,
                 float rtol, float atol) {
  for (uint i = 0; i < size; i++) {
    float a = __bfloat162float(A[i]);
    float d = a - __bfloat162float(B[i]);
    if (abs(d) > abs(atol + rtol * a)) {
      return false;
    }
  }
  return true;
}

void fill_random(__nv_bfloat16 *A, uint size) {
  for (uint i = 0; i < size; i++) {
    A[i] = __float2bfloat16(1 / (float)(random() % 20 + 1));
  }
}

void print_matrix(__nv_bfloat16 *A, uint M, uint N, uint maxM = 16,
                  uint maxN = 16) {
  printf("[[");
  for (uint i = 0; i < M; i++) {
    if ((M < maxM) || (i < maxM / 2) || (i >= M - maxM / 2)) {
      if (i != 0) {
        printf("]\n [");
      }
      for (uint j = 0; j < N; j++) {
        if ((j < maxN / 2) || (j >= N - maxN / 2)) {
          if (j != 0) {
            printf(", ");
          }
          printf("%4.2e", __bfloat162float(A[i * N + j]));
          if (j + 1 == maxN / 2) {
            printf(", ...");
          }
        }
      }
      if (i + 1 == maxM) {
        printf("...\n");
      }
    }
  }
  printf("]]\n");
}

void print_matrix(float *A, uint M, uint N, uint maxM = 16, uint maxN = 16) {
  printf("[[");
  for (uint i = 0; i < M; i++) {
    if ((M < maxM) || (i < maxM / 2) || (i >= M - maxM / 2)) {
      if (i != 0) {
        printf("]\n [");
      }
      for (uint j = 0; j < N; j++) {
        if ((j < maxN / 2) || (j >= N - maxN / 2)) {
          if (j != 0) {
            printf(", ");
          }
          printf("%4.2e", A[i * N + j]);
          if (j + 1 == maxN / 2) {
            printf(", ...");
          }
        }
      }
      if (i + 1 == maxM) {
        printf("...\n");
      }
    }
  }
  printf("]]\n");
}

int main() {
  // Define matrix dimensions.
  int M = 128, N = 32, K = 32;

  // Allocate and initialize matrices A, B, C.
  __nv_bfloat16 *d_A, *d_B, *d_C_b_1, *d_C_b_2;
  float *d_C;
  cudaMallocManaged(&d_A, M * K * sizeof(__nv_bfloat16));
  cudaMallocManaged(&d_B, K * N * sizeof(__nv_bfloat16));
  cudaMallocManaged(&d_C, M * N * sizeof(float));
  cudaMallocManaged(&d_C_b_1, M * N * sizeof(__nv_bfloat16));
  cudaMallocManaged(&d_C_b_2, M * N * sizeof(__nv_bfloat16));

  // Fill d_A and d_B with some data...
  fill_random(d_A, M * K);
  fill_random(d_B, N * K);

  // Define block and grid dimensions.
  dim3 blockDim(32, 16);
  dim3 gridDim((2 * N + blockDim.x - 1) / blockDim.x,
               (M + blockDim.y - 1) / blockDim.y);

  // Invoke the kernel.
  matmul_bf16<<<gridDim, blockDim>>>(d_A, d_B, d_C, d_C_b_1, M, N, K);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  printf("C\n");
  print_matrix(d_C, M, N, 8, 8);

  simple_matmul_bf16(d_A, d_B, d_C, d_C_b_2, M, N, K);
  printf("A\n");
  print_matrix(d_A, M, N, 8, 8);
  printf("B\n");
  print_matrix(d_B, M, N, 8, 8);
  printf("C_1\n");
  print_matrix(d_C_b_1, M, N, 8, 8);
  printf("C_2\n");
  print_matrix(d_C_b_2, M, N, 8, 8);

  printf("%d\n", check_equal(d_C_b_1, d_C_b_2, M * N, 1e-2, 1e-4));

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_C_b_1);
  cudaFree(d_C_b_2);

  return 0;
}