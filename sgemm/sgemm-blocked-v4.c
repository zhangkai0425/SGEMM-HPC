const char* sgemm_desc = "Simple blocked sgemm.";
#include <immintrin.h>
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif
#define SMALL_BLOCK_SIZE 16
#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_opt (int lda, int M, int N, int K, float * A, float * B, float * C)
{
  /* For each column j of B */ 
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k)
    {
      register float b = B[k + j * lda];
      /* For each row i of A */
      int i;
      for (i = 0; i < M - 3; i += 4)
      {
        /* Compute C(i,j) */
        C[i + j * lda] += A[i + k * lda] * b;
        C[i + 1 + j * lda] += A[i + 1 + k * lda] * b;
        C[i + 2 + j * lda] += A[i + 2 + k * lda] * b;
        C[i + 3 + j * lda] += A[i + 3 + k * lda] * b;
      }
      for (; i < M; ++i)
        C[i + j * lda] += A[i + k * lda] * b;
    }
}

static void do_block_avx(int lda, float *A, float *B, float *C)
{
  // 矩阵按列存储 A,B,C都是一维数组,lda是实际的矩阵大小(lda*lda),所以对矩阵元素来说 Aij = A[i + j * lda] 
  // 下面是子块的16*16矩阵乘法,这里把传进来的A,B,C都看成16*16矩阵就行
  for(int j = 0; j < 16; ++j)
  {
    __m512 c = _mm512_loadu_ps(&C[lda * j]);   // 循环16次，AVX-512指令会每次加载第j列中C中的16个float数 C = (c0,c1,...,c15)
    for(int k = 0; k < 16; k++)
    {
      __m512 a = _mm512_loadu_ps(&A[lda * k]);     // 16次,每次加载第k列A中的16个float数
      __m512 b = _mm512_set1_ps(B[k + lda * j]);   // 16次,对应每次加载第j列中第k个B中的数,并复制16次,成为元素相同的16位向量
      c = _mm512_fmadd_ps(a, b, c);                // 执行按位的向量乘法和加法 c = a * b + c 都是按位乘的
    }
    _mm512_storeu_ps(&C[lda * j], c);              // 将c的值存到C的第j列当中
  }
}

static void do_block_large(int lda, int M, int N, int K, float* A, float* B, float* C) 
{
  if((lda < BLOCK_SIZE/2 || lda < BLOCK_SIZE/2 || lda < BLOCK_SIZE/2) && (lda % SMALL_BLOCK_SIZE != 0))
    do_block_opt(lda,M,N,K,A,B,C);
  else
    for (int i = 0; i < M; i += SMALL_BLOCK_SIZE) 
      for (int j = 0; j < N; j += SMALL_BLOCK_SIZE) 
        for (int k = 0; k < K; k += SMALL_BLOCK_SIZE) 
        {
          int M_part = min(SMALL_BLOCK_SIZE, M - i);
          int N_part = min(SMALL_BLOCK_SIZE, N - j);
          int K_part = min(SMALL_BLOCK_SIZE, K - k);
          if (M_part == 16 && N_part == 16 && K_part == 16)
            do_block_avx(lda, A + i + k * lda, B + k + j * lda, C + i + j * lda);
          else 
            do_block_opt(lda, M_part, N_part, K_part, A + i + k * lda, B + k + j * lda, C + i + j * lda);
        }
}

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_sgemm (int lda, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C)
{
  
  /* For each block-column of B */
  for (int j = 0; j < lda; j += BLOCK_SIZE)
  {
    int N = min (BLOCK_SIZE, lda-j);
    /* Accumulate block sgemms into block of C */
    for (int k = 0; k < lda; k += BLOCK_SIZE)
    {
      int K = min (BLOCK_SIZE, lda-k);
      /* For each block-row of A */ 
      for (int i = 0; i < lda; i += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZE, lda-i);
        /* Perform individual block sgemm */
        do_block_large(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
}
