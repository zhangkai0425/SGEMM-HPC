const char* sgemm_desc = "Simple blocked sgemm.";
#include <immintrin.h>
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 128
#endif
#define BLOCK_M_SIZE 128
#define BLOCK_N_SIZE 256
#define BLOCK_K_SIZE 256
#define SMALL_BLOCK_SIZE 16
#define SMALL_BLOCK_M_SIZE 32
#define SMALL_BLOCK_N_SIZE 8

#define min(a,b) (((a)<(b))?(a):(b))

void Pack_A(float *src, float *dst, int lda, int M, int K, int ldp){
    //dim_first: M, dim_second: K
    float *tosrc,*todst;
    todst=dst;
    tosrc=src;
    int i,j;
    for(j=0;j<K;j++)
    {
      for(i=0;i<M-15;i+=16)
        _mm512_store_ps(&todst[i + j * ldp], _mm512_load_ps(&tosrc[i + j * lda]));
      for(;i<M;i++)
        todst[i + j * ldp] = tosrc[i + j * lda];
    }
}

void Pack_B(float *src, float *dst, int ldb, int K, int N, int ldp){
    //dim_first: K, dim_second: N
    float *tosrc,*todst;
    todst=dst;
    tosrc=src;
    int i,j;
    for(j=0;j<N;j++)
    {
      for(i=0;i<K-15;i+=16)
        _mm512_store_ps(&todst[i + j * ldp], _mm512_load_ps(&tosrc[i + j * ldb]));
      for(;i<K;i++)
        todst[i + j * ldp] = tosrc[i + j * ldb];
    }
}

__m512 c10,c20,c30,c40,c50,c60,c70,c80,c11,c21,c31,c41,c51,c61,c71,c81;
__m512 a0,a1,b1,b2,b3,b4,b5,b6,b7,b8;
// A : 32 * k , B : k  * 8, C : 32 * 8 => Using (2 + 8 + 16 = 26 reg)
static void do_block_avx_32k8(int lda, int ldb, int ldc, int K ,float * __restrict__ A, float * __restrict__ B, float * __restrict__ C)
{
  c10 = _mm512_setzero_ps(); 
  c20 = _mm512_setzero_ps();  
  c30 = _mm512_setzero_ps();  
  c40 = _mm512_setzero_ps();  
  c50 = _mm512_setzero_ps();  
  c60 = _mm512_setzero_ps();  
  c70 = _mm512_setzero_ps();  
  c80 = _mm512_setzero_ps(); 

  c11 = _mm512_setzero_ps();     
  c21 = _mm512_setzero_ps();  
  c31 = _mm512_setzero_ps();  
  c41 = _mm512_setzero_ps();  
  c51 = _mm512_setzero_ps();  
  c61 = _mm512_setzero_ps();  
  c71 = _mm512_setzero_ps();  
  c81 = _mm512_setzero_ps(); 

  int k;
  for(k = 0; k < K; k++)
  {
    a0 = _mm512_load_ps(&A[lda * k       ]); 
    a1 = _mm512_load_ps(&A[lda * k + 16  ]); 
    b1 = _mm512_set1_ps(B[k + ldb * 0]);
    b2 = _mm512_set1_ps(B[k + ldb * 1]);
    b3 = _mm512_set1_ps(B[k + ldb * 2]);
    b4 = _mm512_set1_ps(B[k + ldb * 3]);
    b5 = _mm512_set1_ps(B[k + ldb * 4]);
    b6 = _mm512_set1_ps(B[k + ldb * 5]);
    b7 = _mm512_set1_ps(B[k + ldb * 6]);
    b8 = _mm512_set1_ps(B[k + ldb * 7]);

    c10 = _mm512_fmadd_ps(a0, b1, c10);
    c20 = _mm512_fmadd_ps(a0, b2, c20);
    c30 = _mm512_fmadd_ps(a0, b3, c30);
    c40 = _mm512_fmadd_ps(a0, b4, c40);
    c50 = _mm512_fmadd_ps(a0, b5, c50);
    c60 = _mm512_fmadd_ps(a0, b6, c60);
    c70 = _mm512_fmadd_ps(a0, b7, c70);
    c80 = _mm512_fmadd_ps(a0, b8, c80);
    c11 = _mm512_fmadd_ps(a1, b1, c11);
    c21 = _mm512_fmadd_ps(a1, b2, c21);
    c31 = _mm512_fmadd_ps(a1, b3, c31);
    c41 = _mm512_fmadd_ps(a1, b4, c41);
    c51 = _mm512_fmadd_ps(a1, b5, c51);
    c61 = _mm512_fmadd_ps(a1, b6, c61);
    c71 = _mm512_fmadd_ps(a1, b7, c71);
    c81 = _mm512_fmadd_ps(a1, b8, c81);
  }
  _mm512_store_ps(&C[ldc * 0], _mm512_add_ps(c10,_mm512_load_ps(&C[ldc * 0])));          
  _mm512_store_ps(&C[ldc * 1], _mm512_add_ps(c20,_mm512_load_ps(&C[ldc * 1])));          
  _mm512_store_ps(&C[ldc * 2], _mm512_add_ps(c30,_mm512_load_ps(&C[ldc * 2])));          
  _mm512_store_ps(&C[ldc * 3], _mm512_add_ps(c40,_mm512_load_ps(&C[ldc * 3])));          
  _mm512_store_ps(&C[ldc * 4], _mm512_add_ps(c50,_mm512_load_ps(&C[ldc * 4])));          
  _mm512_store_ps(&C[ldc * 5], _mm512_add_ps(c60,_mm512_load_ps(&C[ldc * 5])));          
  _mm512_store_ps(&C[ldc * 6], _mm512_add_ps(c70,_mm512_load_ps(&C[ldc * 6])));          
  _mm512_store_ps(&C[ldc * 7], _mm512_add_ps(c80,_mm512_load_ps(&C[ldc * 7])));    

  _mm512_store_ps(&C[ldc * 0 + 16], _mm512_add_ps(c11,_mm512_load_ps(&C[ldc * 0 + 16])));          
  _mm512_store_ps(&C[ldc * 1 + 16], _mm512_add_ps(c21,_mm512_load_ps(&C[ldc * 1 + 16])));          
  _mm512_store_ps(&C[ldc * 2 + 16], _mm512_add_ps(c31,_mm512_load_ps(&C[ldc * 2 + 16])));          
  _mm512_store_ps(&C[ldc * 3 + 16], _mm512_add_ps(c41,_mm512_load_ps(&C[ldc * 3 + 16])));          
  _mm512_store_ps(&C[ldc * 4 + 16], _mm512_add_ps(c51,_mm512_load_ps(&C[ldc * 4 + 16])));          
  _mm512_store_ps(&C[ldc * 5 + 16], _mm512_add_ps(c61,_mm512_load_ps(&C[ldc * 5 + 16])));          
  _mm512_store_ps(&C[ldc * 6 + 16], _mm512_add_ps(c71,_mm512_load_ps(&C[ldc * 6 + 16])));          
  _mm512_store_ps(&C[ldc * 7 + 16], _mm512_add_ps(c81,_mm512_load_ps(&C[ldc * 7 + 16])));    
}

// A : 64 * k , B : k  * 4, C : 64 * 4 => Using (16 + 4 + 4 = 24 reg)
static void do_block_avx_64k4(int lda, int ldb, int ldc, int K ,float * __restrict__ A, float * __restrict__ B, float * __restrict__ C)
{
  __m512 c10 = _mm512_load_ps(&C[ldc * 0]);    
  __m512 c20 = _mm512_load_ps(&C[ldc * 1]); 
  __m512 c30 = _mm512_load_ps(&C[ldc * 2]); 
  __m512 c40 = _mm512_load_ps(&C[ldc * 3]); 

  __m512 c11 = _mm512_load_ps(&C[ldc * 0 + 16]);    
  __m512 c21 = _mm512_load_ps(&C[ldc * 1 + 16]); 
  __m512 c31 = _mm512_load_ps(&C[ldc * 2 + 16]); 
  __m512 c41 = _mm512_load_ps(&C[ldc * 3 + 16]); 

  __m512 c12 = _mm512_load_ps(&C[ldc * 0 + 32]);    
  __m512 c22 = _mm512_load_ps(&C[ldc * 1 + 32]); 
  __m512 c32 = _mm512_load_ps(&C[ldc * 2 + 32]); 
  __m512 c42 = _mm512_load_ps(&C[ldc * 3 + 32]); 

  __m512 c13 = _mm512_load_ps(&C[ldc * 0 + 48]);    
  __m512 c23 = _mm512_load_ps(&C[ldc * 1 + 48]); 
  __m512 c33 = _mm512_load_ps(&C[ldc * 2 + 48]); 
  __m512 c43 = _mm512_load_ps(&C[ldc * 3 + 48]); 

  __m512 a0,a1,a2,a3,b1,b2,b3,b4;
  int k;
  for(k = 0; k < K; k++)
  {
    a0 = _mm512_load_ps(&A[lda * k       ]); 
    a1 = _mm512_load_ps(&A[lda * k + 16  ]); 
    a2 = _mm512_load_ps(&A[lda * k + 32  ]); 
    a3 = _mm512_load_ps(&A[lda * k + 48  ]); 

    b1 = _mm512_set1_ps(B[k + ldb * 0]);
    b2 = _mm512_set1_ps(B[k + ldb * 1]);
    b3 = _mm512_set1_ps(B[k + ldb * 2]);
    b4 = _mm512_set1_ps(B[k + ldb * 3]);

    c10 = _mm512_fmadd_ps(a0, b1, c10);
    c20 = _mm512_fmadd_ps(a0, b2, c20);
    c30 = _mm512_fmadd_ps(a0, b3, c30);
    c40 = _mm512_fmadd_ps(a0, b4, c40);

    c11 = _mm512_fmadd_ps(a1, b1, c11);
    c21 = _mm512_fmadd_ps(a1, b2, c21);
    c31 = _mm512_fmadd_ps(a1, b3, c31);
    c41 = _mm512_fmadd_ps(a1, b4, c41);

    c12 = _mm512_fmadd_ps(a2, b1, c12);
    c22 = _mm512_fmadd_ps(a2, b2, c22);
    c32 = _mm512_fmadd_ps(a2, b3, c32);
    c42 = _mm512_fmadd_ps(a2, b4, c42);

    c13 = _mm512_fmadd_ps(a3, b1, c13);
    c23 = _mm512_fmadd_ps(a3, b2, c23);
    c33 = _mm512_fmadd_ps(a3, b3, c33);
    c43 = _mm512_fmadd_ps(a3, b4, c43);

  }
  _mm512_store_ps(&C[ldc * 0], c10);          
  _mm512_store_ps(&C[ldc * 1], c20);          
  _mm512_store_ps(&C[ldc * 2], c30);          
  _mm512_store_ps(&C[ldc * 3], c40);            

  _mm512_store_ps(&C[ldc * 0 + 16], c11);          
  _mm512_store_ps(&C[ldc * 1 + 16], c21);          
  _mm512_store_ps(&C[ldc * 2 + 16], c31);          
  _mm512_store_ps(&C[ldc * 3 + 16], c41);    

  _mm512_store_ps(&C[ldc * 0 + 32], c12);          
  _mm512_store_ps(&C[ldc * 1 + 32], c22);          
  _mm512_store_ps(&C[ldc * 2 + 32], c32);          
  _mm512_store_ps(&C[ldc * 3 + 32], c42);  

  _mm512_store_ps(&C[ldc * 0 + 48], c13);          
  _mm512_store_ps(&C[ldc * 1 + 48], c23);          
  _mm512_store_ps(&C[ldc * 2 + 48], c33);          
  _mm512_store_ps(&C[ldc * 3 + 48], c43);        
}

// optimize for less packing
static void do_block_large(int lda,int ldb, int ldc, int M, int N, int K, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) 
{ 

  // acc for BLOCK % (64 * 4) == 0 
  if((M % 64 == 0) && (N % 4 == 0))
  {
    for (int j = 0; j < N; j += 4) 
      for (int i = 0; i < M; i += 64)
        do_block_avx_64k4(lda,ldb,ldc,K, A + i, B + j * ldb, C + i + j * ldc);
    return;
  }


  // acc for BLOCK % (32 * 8) == 0 
  if((M % SMALL_BLOCK_M_SIZE == 0) && (N % SMALL_BLOCK_N_SIZE == 0))
  {
    for (int j = 0; j < N; j += SMALL_BLOCK_N_SIZE) 
      for (int i = 0; i < M; i += SMALL_BLOCK_M_SIZE)
        do_block_avx_32k8(lda,ldb,ldc,K, A + i, B + j * ldb, C + i + j * ldc);
    return;
  }

  // align
  float* __restrict__ AA = (float*)_mm_malloc(sizeof(float) * SMALL_BLOCK_M_SIZE * K, 64);
  float* __restrict__ BB = (float*)_mm_malloc(sizeof(float) * K * SMALL_BLOCK_N_SIZE, 64);
  float* __restrict__ CC = (float*)_mm_malloc(sizeof(float) * SMALL_BLOCK_M_SIZE * SMALL_BLOCK_N_SIZE, 64);
  // pack AA and BB for Boundary
  int M_Left = M % SMALL_BLOCK_M_SIZE;
  int N_Left = N % SMALL_BLOCK_N_SIZE;
  // pack AA = A[M-M_Left:M-M_Left+SMALL_BLOCK_M_SIZE][:] shape = (SMALL_BLOCK_M_SIZE,K) lda = SMALL_BLOCK_M_SIZE
  for(int jj=0; jj < K; jj++)
  {
    _mm512_store_ps(&AA[jj * SMALL_BLOCK_M_SIZE], _mm512_load_ps(&A[M - M_Left + jj * lda]));
    _mm512_store_ps(&AA[jj * SMALL_BLOCK_M_SIZE + 16], _mm512_load_ps(&A[M - M_Left + 16 + jj * lda]));
  }
  // pack BB = B[:][N-N_Left:N-N_Left+SMALL_BLOCK_N_SIZE] shape = (K,SMALL_BLOCK_N_SIZE) lda = K
  
  for(int jj=0; jj < SMALL_BLOCK_N_SIZE; jj++)
  {
    int ii;
    for(ii = 0; ii < K - 15; ii += 16)
      _mm512_store_ps(&BB[ii + jj * K], _mm512_load_ps(&B[ii + (N - N_Left + jj) * ldb]));
    for(; ii < K; ii++)
      BB[ii + jj * K] = B[ii + (N - N_Left + jj) * ldb]; 
  }

  for (int j = 0; j < N; j += SMALL_BLOCK_N_SIZE) 
  {
    int N_part = min(SMALL_BLOCK_N_SIZE, N - j);
    for (int i = 0; i < M; i += SMALL_BLOCK_M_SIZE)
    {
      int M_part = min(SMALL_BLOCK_M_SIZE, M - i);
      // Case 1 : No pack for A and B and C => A:lda B:ldb C:ldc
      if (M_part == SMALL_BLOCK_M_SIZE && N_part == SMALL_BLOCK_N_SIZE)
        do_block_avx_32k8(lda,ldb,ldc,K, A + i, B + j * ldb, C + i + j * ldc);

      // Case 2 : pack A and C => A:SMALL_BLOCK_M_SIZE B:ldb C:SMALL_BLOCK_M_SIZE
      else if(N_part == SMALL_BLOCK_N_SIZE)
      {
        // pack C[M,N] to CC[SMALL_BLOCK_M_SIZE][SMALL_BLOCK_N_SIZE]
        for(int jj = 0; jj < N_part; jj++)
          for(int ii = 0; ii < M_part; ii++)
          {
            CC[ii + jj * SMALL_BLOCK_M_SIZE] = C[(ii+i) + (jj+j) * ldc];
          }  
        do_block_avx_32k8(SMALL_BLOCK_M_SIZE, ldb, SMALL_BLOCK_M_SIZE, K, AA, B + j * ldb, CC);
        // unpack CC[SMALL_BLOCK_M_SIZE][SMALL_BLOCK_N_SIZE] to C[M,N]
        for(int jj = 0; jj < N_part; jj++)
          for(int ii = 0; ii < M_part; ii++)
          {
            C[(ii+i) + (jj+j) * ldc] = CC[ii + jj * SMALL_BLOCK_M_SIZE];
          }
      }
      // Case 3 : pack B and C => A:lda B:K C:SMALL_BLOCK_M_SIZE
      else if(M_part == SMALL_BLOCK_M_SIZE)
      {
        // pack C[M,N] to CC[SMALL_BLOCK_M_SIZE][SMALL_BLOCK_N_SIZE]
        for(int jj = 0; jj < N_part; jj++)
          for(int ii = 0; ii < M_part; ii++)
          {
            CC[ii + jj * SMALL_BLOCK_M_SIZE] = C[(ii+i) + (jj+j) * ldc];
          }  
        do_block_avx_32k8(lda, K, SMALL_BLOCK_M_SIZE, K, A + i, BB, CC);
        // unpack CC[SMALL_BLOCK_M_SIZE][SMALL_BLOCK_N_SIZE] to C[M,N]
        for(int jj = 0; jj < N_part; jj++)
          for(int ii = 0; ii < M_part; ii++)
          {
            C[(ii+i) + (jj+j) * ldc] = CC[ii + jj * SMALL_BLOCK_M_SIZE];
          }
      }
      // Case 4 : pack A and B and C => A:SMALL_BLOCK_M_SIZE B:K C:SMALL_BLOCK_M_SIZE
      else
      {
        // pack C[M,N] to CC[16][16]
        for(int jj = 0; jj < N_part; jj++)
          for(int ii = 0; ii < M_part; ii++)
          {
            CC[ii + jj * SMALL_BLOCK_M_SIZE] = C[(ii+i) + (jj+j) * ldc];
          }
        do_block_avx_32k8(SMALL_BLOCK_M_SIZE, K, SMALL_BLOCK_M_SIZE, K, AA, BB, CC);
        // unpack CC[16][16] to C[M,N]
        for(int jj = 0; jj < N_part; jj++)
          for(int ii = 0; ii < M_part; ii++)
            C[(ii+i) + (jj+j) * ldc] = CC[ii + jj * SMALL_BLOCK_M_SIZE] ;
      }  
    }
  }
  _mm_free(AA);
  _mm_free(BB);
  _mm_free(CC);
}

// optimize for less packing
static void do_block_large_opt(int lda,int ldb, int ldc, int M, int N, int K, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) 
{ 

  // acc for BLOCK % (64 * 4) == 0 
  if((M % 64 == 0) && (N % 4 == 0))
  {
    for (int j = 0; j < N; j += 16) 
      for (int i = 0; i < M; i += 64)
      {
        do_block_avx_64k4(lda,ldb,ldc,K, A + i, B + j * ldb, C + i + j * ldc);
        do_block_avx_64k4(lda,ldb,ldc,K, A + i, B + (j + 4) * ldb, C + i + (j + 4) * ldc);
        do_block_avx_64k4(lda,ldb,ldc,K, A + i, B + (j + 8) * ldb, C + i + (j + 8) * ldc);
        do_block_avx_64k4(lda,ldb,ldc,K, A + i, B + (j + 12) * ldb, C + i + (j + 12) * ldc);
      }  
    return;
  }

  // acc for BLOCK % (32 * 8) == 0 
  if((M % SMALL_BLOCK_M_SIZE == 0) && (N % SMALL_BLOCK_N_SIZE == 0))
  {
    for (int j = 0; j < N; j += SMALL_BLOCK_N_SIZE) 
      for (int i = 0; i < M; i += SMALL_BLOCK_M_SIZE)
        do_block_avx_32k8(lda,ldb,ldc,K, A + i, B + j * ldb, C + i + j * ldc);
    return;
  }

  // calculate N_64,M_64,flag
  int M_64 = 0;
  int N_64 = 0;
  int flag_64 = 0;

  int N_L = N;
  int M_L = M;

  // can be divided by 64x4 block
  if((M>=64) && (N>=4))
  {
    flag_64 = 1; 
    M_L = M % 64;
    N_L = N % 4;
    N_64 = N - N_L;
    M_64 = M - M_L;
  }

  // do_block_large(lda,ldb,ldc, M, N, K, A, B, C);
  int i,j;
  if(flag_64==1)
    for (j = 0; j < N_64; j += 4) 
      for (i = 0; i < M_64; i += 64)
        do_block_avx_64k4(lda,ldb,ldc,K, A + i, B + j * ldb, C + i + j * ldc);
  
  // Matrix L1[M_L * N]
  if(M_L > 0)
    do_block_large(lda,ldb,ldc, M_L, N, K, A + M_64, B, C + M_64);
  // Matrix L2[M_64 * N_L]
  if((N_L > 0) && (M_64 > 0))
    do_block_large(lda,ldb,ldc, M_64, N_L, K, A, B + N_64 * ldb, C + N_64 * ldc);
}

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  

void square_sgemm (int lda, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C)
{
  int BLOCK_K_L_SIZE = 0;
  int BLOCK_N_L_SIZE = 0;
  if(lda > 100)
  {
    BLOCK_K_L_SIZE = 64;
    BLOCK_N_L_SIZE = 256;
    if(lda > 250)
    {
      BLOCK_K_L_SIZE = 256;
      BLOCK_N_L_SIZE = 384;
    }
    if(lda > 380)
    {
      BLOCK_K_L_SIZE = 512;
      BLOCK_N_L_SIZE = 512;
    }
    float* A_BUFF = (float*)_mm_malloc(sizeof(float) * BLOCK_M_SIZE * BLOCK_K_L_SIZE, 256);
    float* B_BUFF = (float*)_mm_malloc(sizeof(float) * BLOCK_K_L_SIZE * BLOCK_N_L_SIZE, 256);
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_N_L_SIZE)
    {
      int N = min (BLOCK_N_L_SIZE, lda-j);
      /* Accumulate block sgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_K_L_SIZE)
      {
        int K = min (BLOCK_K_L_SIZE, lda-k);
        // Packing B
        Pack_B(B + k + j*lda, B_BUFF, lda, K, N, BLOCK_K_L_SIZE);
        /* For each block-row of A */ 
        for (int i = 0; i < lda; i += BLOCK_M_SIZE)
        {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int M = min (BLOCK_M_SIZE, lda-i);
          // Packing A
          Pack_A(A + i + k*lda, A_BUFF, lda, M, K, BLOCK_M_SIZE);
          /* Perform individual block sgemm */
          do_block_large_opt(BLOCK_M_SIZE,BLOCK_K_L_SIZE,lda, M, N, K, A_BUFF, B_BUFF, C + i + j*lda);
        }
      }
    }
    _mm_free(A_BUFF);
    _mm_free(B_BUFF);
  }
  else
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_N_SIZE)
    {
      int N = min (BLOCK_N_SIZE, lda-j);
      /* Accumulate block sgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_K_SIZE)
      {
        int K = min (BLOCK_K_SIZE, lda-k);
        /* For each block-row of A */ 
        for (int i = 0; i < lda; i += BLOCK_M_SIZE)
        {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int M = min (BLOCK_M_SIZE, lda-i);
          /* Perform individual block sgemm */
          do_block_large(lda,lda,lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
        }
      }
    }
}