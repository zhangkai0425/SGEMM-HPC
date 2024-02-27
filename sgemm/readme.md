## 串行优化策略

#### 1.内存访问的优化策略：最简单的连续内存访问+群里的提示
https://siboehm.com/articles/22/Fast-MMM-on-CPU
```cpp```
static void do_block (int lda, int M, int N, int K, float* A, float* B, float* C)
{
  /* For each column j of B */ 
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k)
      /* For each row i of A */
      for (int i = 0; i < M; ++i)
      {
        /* Compute C(i,j) */
        C[i+j*lda] += A[i+k*lda] * B[k+j*lda];
      }
}
```cpp```
#### 2.
void square_sgemm (int lda, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C)