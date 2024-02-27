const char* sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif
#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, float * A, float * B, float * C)
{
  /* For each column j of B */ 
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k)
    {
      register float b = B[k + j * lda];
      /* For each row i of A */
      for (int i = 0; i < M; ++i)
      {
        /* Compute C(i,j) */
        C[i+j*lda] += A[i+k*lda] * b;
      }
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
        do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
}