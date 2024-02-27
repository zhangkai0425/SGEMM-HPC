const char* sgemm_desc = "Simple blocked sgemm.";

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
      // 处理剩余的不足 4 个元素的情况
      for (; i < M; ++i)
        C[i + j * lda] += A[i + k * lda] * b;
    }
}

static void do_block_large(int lda, int M, int N, int K, float* A, float* B, float* C) 
{
  for (int i = 0; i < M; i += SMALL_BLOCK_SIZE) 
    for (int j = 0; j < N; j += SMALL_BLOCK_SIZE) 
      for (int k = 0; k < K; k += SMALL_BLOCK_SIZE) 
      {
        int M_part = min(SMALL_BLOCK_SIZE, M - i);
        int N_part = min(SMALL_BLOCK_SIZE, N - j);
        int K_part = min(SMALL_BLOCK_SIZE, K - k);
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
