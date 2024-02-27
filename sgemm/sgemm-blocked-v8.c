const char* sgemm_desc = "Simple blocked sgemm.";
#include <immintrin.h>
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif
#define SMALL_BLOCK_SIZE 16
#define min(a,b) (((a)<(b))?(a):(b))

// A : 16 * k , B : k  * 16, C : 16 * 16
static void do_block_avx_16k16(int lda, int ldb, int ldc, int K ,float * __restrict__ A, float * __restrict__ B, float * __restrict__ C)
{
  __m512 c1 = _mm512_load_ps(&C[ldc * 0]);    
  __m512 c2 = _mm512_load_ps(&C[ldc * 1]); 
  __m512 c3 = _mm512_load_ps(&C[ldc * 2]); 
  __m512 c4 = _mm512_load_ps(&C[ldc * 3]); 
  __m512 c5 = _mm512_load_ps(&C[ldc * 4]); 
  __m512 c6 = _mm512_load_ps(&C[ldc * 5]); 
  __m512 c7 = _mm512_load_ps(&C[ldc * 6]); 
  __m512 c8 = _mm512_load_ps(&C[ldc * 7]); 
  __m512 c9 = _mm512_load_ps(&C[ldc * 8]); 
  __m512 c10 = _mm512_load_ps(&C[ldc * 9]); 
  __m512 c11 = _mm512_load_ps(&C[ldc * 10]); 
  __m512 c12 = _mm512_load_ps(&C[ldc * 11]); 
  __m512 c13 = _mm512_load_ps(&C[ldc * 12]); 
  __m512 c14 = _mm512_load_ps(&C[ldc * 13]); 
  __m512 c15 = _mm512_load_ps(&C[ldc * 14]); 
  __m512 c16 = _mm512_load_ps(&C[ldc * 15]); 
  int k;
  for(k = 0; k < K-3; k += 4)
  {
    __m512 a1 = _mm512_load_ps(&A[lda * k    ]);  
    __m512 a2 = _mm512_load_ps(&A[lda * (k+1)]);     
    __m512 a3 = _mm512_load_ps(&A[lda * (k+2)]);  
    __m512 a4 = _mm512_load_ps(&A[lda * (k+3)]);


    __m512 b1_1 = _mm512_set1_ps(B[k + 0 + ldb * 0]);
    __m512 b1_2 = _mm512_set1_ps(B[k + 1 + ldb * 0]);
    __m512 b1_3 = _mm512_set1_ps(B[k + 2 + ldb * 0]);     
    __m512 b1_4 = _mm512_set1_ps(B[k + 3 + ldb * 0]);    
    
    __m512 b2_1 = _mm512_set1_ps(B[k + 0 + ldb * 1]);
    __m512 b2_2 = _mm512_set1_ps(B[k + 1 + ldb * 1]);
    __m512 b2_3 = _mm512_set1_ps(B[k + 2 + ldb * 1]);     
    __m512 b2_4 = _mm512_set1_ps(B[k + 3 + ldb * 1]);    

    __m512 b3_1 = _mm512_set1_ps(B[k + 0 + ldb * 2]);
    __m512 b3_2 = _mm512_set1_ps(B[k + 1 + ldb * 2]);
    __m512 b3_3 = _mm512_set1_ps(B[k + 2 + ldb * 2]);     
    __m512 b3_4 = _mm512_set1_ps(B[k + 3 + ldb * 2]);    

    __m512 b4_1 = _mm512_set1_ps(B[k + 0 + ldb * 3]);
    __m512 b4_2 = _mm512_set1_ps(B[k + 1 + ldb * 3]);
    __m512 b4_3 = _mm512_set1_ps(B[k + 2 + ldb * 3]);     
    __m512 b4_4 = _mm512_set1_ps(B[k + 3 + ldb * 3]);    

    __m512 b5_1 = _mm512_set1_ps(B[k + 0 + ldb * 4]);
    __m512 b5_2 = _mm512_set1_ps(B[k + 1 + ldb * 4]);
    __m512 b5_3 = _mm512_set1_ps(B[k + 2 + ldb * 4]);     
    __m512 b5_4 = _mm512_set1_ps(B[k + 3 + ldb * 4]);    

    __m512 b6_1 = _mm512_set1_ps(B[k + 0 + ldb * 5]);
    __m512 b6_2 = _mm512_set1_ps(B[k + 1 + ldb * 5]);
    __m512 b6_3 = _mm512_set1_ps(B[k + 2 + ldb * 5]);     
    __m512 b6_4 = _mm512_set1_ps(B[k + 3 + ldb * 5]);    

    __m512 b7_1 = _mm512_set1_ps(B[k + 0 + ldb * 6]);
    __m512 b7_2 = _mm512_set1_ps(B[k + 1 + ldb * 6]);
    __m512 b7_3 = _mm512_set1_ps(B[k + 2 + ldb * 6]);     
    __m512 b7_4 = _mm512_set1_ps(B[k + 3 + ldb * 6]);    

    __m512 b8_1 = _mm512_set1_ps(B[k + 0 + ldb * 7]);
    __m512 b8_2 = _mm512_set1_ps(B[k + 1 + ldb * 7]);
    __m512 b8_3 = _mm512_set1_ps(B[k + 2 + ldb * 7]);     
    __m512 b8_4 = _mm512_set1_ps(B[k + 3 + ldb * 7]);    

    __m512 b9_1 = _mm512_set1_ps(B[k + 0 + ldb * 8]);
    __m512 b9_2 = _mm512_set1_ps(B[k + 1 + ldb * 8]);
    __m512 b9_3 = _mm512_set1_ps(B[k + 2 + ldb * 8]);     
    __m512 b9_4 = _mm512_set1_ps(B[k + 3 + ldb * 8]);    

    __m512 b10_1 = _mm512_set1_ps(B[k + 0 + ldb * 9]);
    __m512 b10_2 = _mm512_set1_ps(B[k + 1 + ldb * 9]);
    __m512 b10_3 = _mm512_set1_ps(B[k + 2 + ldb * 9]);     
    __m512 b10_4 = _mm512_set1_ps(B[k + 3 + ldb * 9]);    

    __m512 b11_1 = _mm512_set1_ps(B[k + 0 + ldb * 10]);
    __m512 b11_2 = _mm512_set1_ps(B[k + 1 + ldb * 10]);
    __m512 b11_3 = _mm512_set1_ps(B[k + 2 + ldb * 10]);     
    __m512 b11_4 = _mm512_set1_ps(B[k + 3 + ldb * 10]);    

    __m512 b12_1 = _mm512_set1_ps(B[k + 0 + ldb * 11]);
    __m512 b12_2 = _mm512_set1_ps(B[k + 1 + ldb * 11]);
    __m512 b12_3 = _mm512_set1_ps(B[k + 2 + ldb * 11]);     
    __m512 b12_4 = _mm512_set1_ps(B[k + 3 + ldb * 11]);    

    __m512 b13_1 = _mm512_set1_ps(B[k + 0 + ldb * 12]);
    __m512 b13_2 = _mm512_set1_ps(B[k + 1 + ldb * 12]);
    __m512 b13_3 = _mm512_set1_ps(B[k + 2 + ldb * 12]);     
    __m512 b13_4 = _mm512_set1_ps(B[k + 3 + ldb * 12]);    

    __m512 b14_1 = _mm512_set1_ps(B[k + 0 + ldb * 13]);
    __m512 b14_2 = _mm512_set1_ps(B[k + 1 + ldb * 13]);
    __m512 b14_3 = _mm512_set1_ps(B[k + 2 + ldb * 13]);     
    __m512 b14_4 = _mm512_set1_ps(B[k + 3 + ldb * 13]);    

    __m512 b15_1 = _mm512_set1_ps(B[k + 0 + ldb * 14]);
    __m512 b15_2 = _mm512_set1_ps(B[k + 1 + ldb * 14]);
    __m512 b15_3 = _mm512_set1_ps(B[k + 2 + ldb * 14]);     
    __m512 b15_4 = _mm512_set1_ps(B[k + 3 + ldb * 14]);    

    __m512 b16_1 = _mm512_set1_ps(B[k + 0 + ldb * 15]);
    __m512 b16_2 = _mm512_set1_ps(B[k + 1 + ldb * 15]);
    __m512 b16_3 = _mm512_set1_ps(B[k + 2 + ldb * 15]);     
    __m512 b16_4 = _mm512_set1_ps(B[k + 3 + ldb * 15]); 

    c1 = _mm512_fmadd_ps(a1, b1_1, c1);   
    c1 = _mm512_fmadd_ps(a2, b1_2, c1);  
    c1 = _mm512_fmadd_ps(a3, b1_3, c1);   
    c1 = _mm512_fmadd_ps(a4, b1_4, c1); 

    c2 = _mm512_fmadd_ps(a1, b2_1, c2);   
    c2 = _mm512_fmadd_ps(a2, b2_2, c2);  
    c2 = _mm512_fmadd_ps(a3, b2_3, c2);   
    c2 = _mm512_fmadd_ps(a4, b2_4, c2); 

    c3 = _mm512_fmadd_ps(a1, b3_1, c3);   
    c3 = _mm512_fmadd_ps(a2, b3_2, c3);  
    c3 = _mm512_fmadd_ps(a3, b3_3, c3);   
    c3 = _mm512_fmadd_ps(a4, b3_4, c3); 

    c4 = _mm512_fmadd_ps(a1, b4_1, c4);   
    c4 = _mm512_fmadd_ps(a2, b4_2, c4);  
    c4 = _mm512_fmadd_ps(a3, b4_3, c4);   
    c4 = _mm512_fmadd_ps(a4, b4_4, c4); 

    c5 = _mm512_fmadd_ps(a1, b5_1, c5);   
    c5 = _mm512_fmadd_ps(a2, b5_2, c5);  
    c5 = _mm512_fmadd_ps(a3, b5_3, c5);   
    c5 = _mm512_fmadd_ps(a4, b5_4, c5); 

    c6 = _mm512_fmadd_ps(a1, b6_1, c6);   
    c6 = _mm512_fmadd_ps(a2, b6_2, c6);  
    c6 = _mm512_fmadd_ps(a3, b6_3, c6);   
    c6 = _mm512_fmadd_ps(a4, b6_4, c6); 

    c7 = _mm512_fmadd_ps(a1, b7_1, c7);   
    c7 = _mm512_fmadd_ps(a2, b7_2, c7);  
    c7 = _mm512_fmadd_ps(a3, b7_3, c7);   
    c7 = _mm512_fmadd_ps(a4, b7_4, c7); 

    c8 = _mm512_fmadd_ps(a1, b8_1, c8);   
    c8 = _mm512_fmadd_ps(a2, b8_2, c8);  
    c8 = _mm512_fmadd_ps(a3, b8_3, c8);   
    c8 = _mm512_fmadd_ps(a4, b8_4, c8); 

    c9 = _mm512_fmadd_ps(a1, b9_1, c9);   
    c9 = _mm512_fmadd_ps(a2, b9_2, c9);  
    c9 = _mm512_fmadd_ps(a3, b9_3, c9);   
    c9 = _mm512_fmadd_ps(a4, b9_4, c9); 

    c10 = _mm512_fmadd_ps(a1, b10_1, c10);   
    c10 = _mm512_fmadd_ps(a2, b10_2, c10);  
    c10 = _mm512_fmadd_ps(a3, b10_3, c10);   
    c10 = _mm512_fmadd_ps(a4, b10_4, c10); 

    c11 = _mm512_fmadd_ps(a1, b11_1, c11);   
    c11 = _mm512_fmadd_ps(a2, b11_2, c11);  
    c11 = _mm512_fmadd_ps(a3, b11_3, c11);   
    c11 = _mm512_fmadd_ps(a4, b11_4, c11); 

    c12 = _mm512_fmadd_ps(a1, b12_1, c12);   
    c12 = _mm512_fmadd_ps(a2, b12_2, c12);  
    c12 = _mm512_fmadd_ps(a3, b12_3, c12);   
    c12 = _mm512_fmadd_ps(a4, b12_4, c12); 

    c13 = _mm512_fmadd_ps(a1, b13_1, c13);   
    c13 = _mm512_fmadd_ps(a2, b13_2, c13);  
    c13 = _mm512_fmadd_ps(a3, b13_3, c13);   
    c13 = _mm512_fmadd_ps(a4, b13_4, c13); 

    c14 = _mm512_fmadd_ps(a1, b14_1, c14);   
    c14 = _mm512_fmadd_ps(a2, b14_2, c14);  
    c14 = _mm512_fmadd_ps(a3, b14_3, c14);   
    c14 = _mm512_fmadd_ps(a4, b14_4, c14); 

    c15 = _mm512_fmadd_ps(a1, b15_1, c15);   
    c15 = _mm512_fmadd_ps(a2, b15_2, c15);  
    c15 = _mm512_fmadd_ps(a3, b15_3, c15);   
    c15 = _mm512_fmadd_ps(a4, b15_4, c15); 

    c16 = _mm512_fmadd_ps(a1, b16_1, c16);   
    c16 = _mm512_fmadd_ps(a2, b16_2, c16);  
    c16 = _mm512_fmadd_ps(a3, b16_3, c16);   
    c16 = _mm512_fmadd_ps(a4, b16_4, c16);   
  }

  for(; k < K; k++)
  {
    __m512 a = _mm512_load_ps(&A[lda * k]); 
    __m512 b1 = _mm512_set1_ps(B[k + 0 + ldb * 0]);
    __m512 b2 = _mm512_set1_ps(B[k + 0 + ldb * 1]);
    __m512 b3 = _mm512_set1_ps(B[k + 0 + ldb * 2]);
    __m512 b4 = _mm512_set1_ps(B[k + 0 + ldb * 3]);
    __m512 b5 = _mm512_set1_ps(B[k + 0 + ldb * 4]);
    __m512 b6 = _mm512_set1_ps(B[k + 0 + ldb * 5]);
    __m512 b7 = _mm512_set1_ps(B[k + 0 + ldb * 6]);
    __m512 b8 = _mm512_set1_ps(B[k + 0 + ldb * 7]);
    __m512 b9 = _mm512_set1_ps(B[k + 0 + ldb * 8]);
    __m512 b10 = _mm512_set1_ps(B[k + 0 + ldb * 9]);
    __m512 b11 = _mm512_set1_ps(B[k + 0 + ldb * 10]);
    __m512 b12 = _mm512_set1_ps(B[k + 0 + ldb * 11]);
    __m512 b13 = _mm512_set1_ps(B[k + 0 + ldb * 12]);
    __m512 b14 = _mm512_set1_ps(B[k + 0 + ldb * 13]);
    __m512 b15 = _mm512_set1_ps(B[k + 0 + ldb * 14]);
    __m512 b16 = _mm512_set1_ps(B[k + 0 + ldb * 15]);
    c1 = _mm512_fmadd_ps(a, b1, c1); 
    c2 = _mm512_fmadd_ps(a, b2, c2); 
    c3 = _mm512_fmadd_ps(a, b3, c3);  
    c4 = _mm512_fmadd_ps(a, b4, c4); 
    c5 = _mm512_fmadd_ps(a, b5, c5);  
    c6 = _mm512_fmadd_ps(a, b6, c6); 
    c7 = _mm512_fmadd_ps(a, b7, c7); 
    c8 = _mm512_fmadd_ps(a, b8, c8);
    c9 = _mm512_fmadd_ps(a, b9, c9); 
    c10 = _mm512_fmadd_ps(a, b10, c10);  
    c11 = _mm512_fmadd_ps(a, b11, c11); 
    c12 = _mm512_fmadd_ps(a, b12, c12); 
    c13 = _mm512_fmadd_ps(a, b13, c13); 
    c14 = _mm512_fmadd_ps(a, b14, c14); 
    c15 = _mm512_fmadd_ps(a, b15, c15); 
    c16 = _mm512_fmadd_ps(a, b16, c16); 
  }
  _mm512_store_ps(&C[ldc * 0], c1);          
  _mm512_store_ps(&C[ldc * 1], c2);          
  _mm512_store_ps(&C[ldc * 2], c3);          
  _mm512_store_ps(&C[ldc * 3], c4);          
  _mm512_store_ps(&C[ldc * 4], c5);          
  _mm512_store_ps(&C[ldc * 5], c6);          
  _mm512_store_ps(&C[ldc * 6], c7);          
  _mm512_store_ps(&C[ldc * 7], c8);          
  _mm512_store_ps(&C[ldc * 8], c9);          
  _mm512_store_ps(&C[ldc * 9], c10);         
  _mm512_store_ps(&C[ldc * 10], c11);        
  _mm512_store_ps(&C[ldc * 11], c12);        
  _mm512_store_ps(&C[ldc * 12], c13);        
  _mm512_store_ps(&C[ldc * 13], c14);        
  _mm512_store_ps(&C[ldc * 14], c15);        
  _mm512_store_ps(&C[ldc * 15], c16);        
}

// optimize for less packing
static void do_block_large(int lda, int M, int N, int K, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) 
{ 
  // acc for BLOCK % 16 == 0 
  if((M % SMALL_BLOCK_SIZE == 0) && (N % SMALL_BLOCK_SIZE == 0))
  {
    for (int j = 0; j < N; j += SMALL_BLOCK_SIZE) 
      for (int i = 0; i < M; i += 2 * SMALL_BLOCK_SIZE)
      {
        do_block_avx_16k16(lda,lda,lda,K, A + i, B + j * lda, C + i + j * lda);
        do_block_avx_16k16(lda,lda,lda,K, A + i + SMALL_BLOCK_SIZE, B + j * lda, C + i + SMALL_BLOCK_SIZE + j * lda);
      }
    return;
  }
  // align
  float* __restrict__ AA = (float*)_mm_malloc(sizeof(float) * SMALL_BLOCK_SIZE * K, 64);
  float* __restrict__ BB = (float*)_mm_malloc(sizeof(float) * K * SMALL_BLOCK_SIZE, 64);
  float* __restrict__ CC = (float*)_mm_malloc(sizeof(float) * SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE, 64);

  // pack AA and BB for Boundary
  int M_Left = M % SMALL_BLOCK_SIZE;
  int N_Left = N % SMALL_BLOCK_SIZE;
  // pack AA = A[M-M_Left:M-M_Left+SMALL_BLOCK_SIZE][:] shape = (SMALL_BLOCK_SIZE,K) lda = SMALL_BLOCK_SIZE
  for(int jj=0; jj < K; jj++)
  {
    __m512 Avec = _mm512_loadu_ps(&A[M - M_Left + jj * lda]);
    _mm512_storeu_ps(&AA[jj * SMALL_BLOCK_SIZE], Avec);
  }
  // pack BB = B[:][N-N_Left:N-N_Left+SMALL_BLOCK_SIZE] shape = (K,SMALL_BLOCK_SIZE) lda = K
  for(int jj=0; jj < SMALL_BLOCK_SIZE; jj++)
  {
    int ii;
    for(ii = 0; ii < K - 15; ii += 16)
    {
      __m512 Bvec = _mm512_loadu_ps(&B[ii + (N - N_Left + jj) * lda]);
      _mm512_storeu_ps(&BB[ii + jj * K], Bvec);
    }
    for(; ii < K; ii++)
      BB[ii + jj * K] = B[ii + (N - N_Left + jj) * lda]; 
  }
  for (int j = 0; j < N; j += SMALL_BLOCK_SIZE) 
  {
    int N_part = min(SMALL_BLOCK_SIZE, N - j);
    for (int i = 0; i < M; i += SMALL_BLOCK_SIZE)
    {
      int M_part = min(SMALL_BLOCK_SIZE, M - i);
      // Case 1 : No pack for A and B and C => A:lda B:lda C:lda
      if (M_part == SMALL_BLOCK_SIZE && N_part == SMALL_BLOCK_SIZE)
        do_block_avx_16k16(lda,lda,lda,K, A + i, B + j * lda, C + i + j * lda);

      // Case 2 : pack A and C => A:SMALL_BLOCK_SIZE B:lda C:SMALL_BLOCK_SIZE
      else if(N_part == SMALL_BLOCK_SIZE)
      {
        // pack C[M,N] to CC[16][16]
        for(int jj = 0; jj < N_part; jj++)
          for(int ii = 0; ii < M_part; ii++)
          {
            CC[ii + jj * SMALL_BLOCK_SIZE] = C[(ii+i) + (jj+j) * lda];
          }  
        do_block_avx_16k16(SMALL_BLOCK_SIZE, lda, SMALL_BLOCK_SIZE, K, AA, B + j * lda, CC);
        // unpack CC[16][16] to C[M,N]
        for(int jj = 0; jj < N_part; jj++)
          for(int ii = 0; ii < M_part; ii++)
          {
            C[(ii+i) + (jj+j) * lda] = CC[ii + jj * SMALL_BLOCK_SIZE] ;
          }
      }
      // Case 3 : pack B and C => A:lda B:K C:SMALL_BLOCK_SIZE
      else if(M_part == SMALL_BLOCK_SIZE)
      {
        // pack C[M,N] to CC[16][16]
        for(int jj = 0; jj < N_part; jj++)
          for(int ii = 0; ii < M_part; ii++)
          {
            CC[ii + jj * SMALL_BLOCK_SIZE] = C[(ii+i) + (jj+j) * lda];
          }  
        do_block_avx_16k16(lda, K, SMALL_BLOCK_SIZE, K, A + i, BB, CC);
        // unpack CC[16][16] to C[M,N]
        for(int jj = 0; jj < N_part; jj++)
          for(int ii = 0; ii < M_part; ii++)
          {
            C[(ii+i) + (jj+j) * lda] = CC[ii + jj * SMALL_BLOCK_SIZE] ;
          }
      }
      // Case 4 : pack A and B and C => A:SMALL_BLOCK_SIZE B:K C:SMALL_BLOCK_SIZE
      else
      {
        // pack C[M,N] to CC[16][16]
        for(int jj = 0; jj < N_part; jj++)
          for(int ii = 0; ii < M_part; ii++)
          {
            CC[ii + jj * SMALL_BLOCK_SIZE] = C[(ii+i) + (jj+j) * lda];
          }
        do_block_avx_16k16(SMALL_BLOCK_SIZE, K, SMALL_BLOCK_SIZE, K, AA, BB, CC);
        // unpack CC[16][16] to C[M,N]
        for(int jj = 0; jj < N_part; jj++)
          for(int ii = 0; ii < M_part; ii++)
            C[(ii+i) + (jj+j) * lda] = CC[ii + jj * SMALL_BLOCK_SIZE] ;
      }  
    }
  }
  _mm_free(AA);
  _mm_free(BB);
  _mm_free(CC);
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
      /* For each block-row of A */ 
    for (int i = 0; i < lda; i += BLOCK_SIZE)
    {
      /* Correct block dimensions if block "goes off edge of" the matrix */
      int M = min (BLOCK_SIZE, lda-i);
      /* Perform individual block sgemm */
      do_block_large(lda, M, N, lda, A + i, B + j*lda, C + i + j*lda);
    }
  }
}
