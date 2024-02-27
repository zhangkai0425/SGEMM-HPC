const char* sgemm_desc = "Simple blocked sgemm.";
#include <immintrin.h>
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 128
#endif
#define BLOCK_M_SIZE 128
#define BLOCK_N_SIZE 128
#define BLOCK_K_SIZE 256
#define SMALL_BLOCK_SIZE 16
#define SMALL_BLOCK_M_SIZE 32
#define SMALL_BLOCK_N_SIZE 8
#define SMALL_BLOCK_M_64_SIZE 64
#define SMALL_BLOCK_N_8_SIZE 8
#define SMALL_BLOCK_N_4_SIZE 4
#define min(a,b) (((a)<(b))?(a):(b))

void Pack_A(float *src, float *dst, int lda, int M, int K){
    //dim_first: M, dim_second: K
    float *tosrc,*todst;
    todst=dst;
    tosrc=src;
    int i,j;
    for(j=0;j<K;j++)
    {
      for(i=0;i<M-15;i+=16)
        _mm512_store_ps(&todst[i + j * BLOCK_M_SIZE], _mm512_load_ps(&tosrc[i + j * lda]));
      for(;i<M;i++)
        todst[i + j * BLOCK_M_SIZE] = tosrc[i + j * lda];
    }
}

void Pack_B(float *src, float *dst, int ldb, int K, int N){
    //dim_first: K, dim_second: N
    float *tosrc,*todst;
    todst=dst;
    tosrc=src;
    int i,j;
    for(j=0;j<N;j++)
    {
      for(i=0;i<K-15;i+=16)
        _mm512_store_ps(&todst[i + j * BLOCK_K_SIZE], _mm512_load_ps(&tosrc[i + j * ldb]));
      for(;i<K;i++)
        todst[i + j * BLOCK_K_SIZE] = tosrc[i + j * ldb];
    }
}

void Pack_C(float *src, float *dst, int ldc, int M, int N){
    //dim_first: K, dim_second: N
    float *tosrc,*todst;
    todst=dst;
    tosrc=src;
    int i,j;
    for(j=0;j<N;j++)
    {
      for(i=0;i<M-15;i+=16)
        _mm512_store_ps(&todst[i + j * BLOCK_M_SIZE], _mm512_load_ps(&tosrc[i + j * ldc]));
      for(;i<M;i++)
        todst[i + j * BLOCK_M_SIZE] = tosrc[i + j * ldc];
    }
}

void Un_Pack_C(float *src, float *dst, int ldc, int M, int N){
    //dim_first: K, dim_second: N
    float *tosrc,*todst;
    todst=dst;
    tosrc=src;
    int i,j;
    for(j=0;j<N;j++)
    {
      for(i=0;i<M-15;i+=16)
        _mm512_store_ps(&tosrc[i + j * ldc], _mm512_load_ps(&todst[i + j * BLOCK_M_SIZE]));
      for(;i<M;i++)
        tosrc[i + j * ldc] = todst[i + j * BLOCK_M_SIZE];
    }
}

// __m512 a[2];
// __m512 b[8];
// __m512 c[16];
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

// A : 48 * k , B : k  * 8, C : 48 * 8 => Using (3 + 8 + 24 = 35 reg)
static void do_block_avx_48k8(int lda, int ldb, int ldc, int K ,float * __restrict__ A, float * __restrict__ B, float * __restrict__ C)
{
  __m512 c10 = _mm512_load_ps(&C[ldc * 0]);    
  __m512 c20 = _mm512_load_ps(&C[ldc * 1]); 
  __m512 c30 = _mm512_load_ps(&C[ldc * 2]); 
  __m512 c40 = _mm512_load_ps(&C[ldc * 3]); 
  __m512 c50 = _mm512_load_ps(&C[ldc * 4]); 
  __m512 c60 = _mm512_load_ps(&C[ldc * 5]); 
  __m512 c70 = _mm512_load_ps(&C[ldc * 6]); 
  __m512 c80 = _mm512_load_ps(&C[ldc * 7]);

  __m512 c11 = _mm512_load_ps(&C[ldc * 0 + 16]);    
  __m512 c21 = _mm512_load_ps(&C[ldc * 1 + 16]); 
  __m512 c31 = _mm512_load_ps(&C[ldc * 2 + 16]); 
  __m512 c41 = _mm512_load_ps(&C[ldc * 3 + 16]); 
  __m512 c51 = _mm512_load_ps(&C[ldc * 4 + 16]); 
  __m512 c61 = _mm512_load_ps(&C[ldc * 5 + 16]); 
  __m512 c71 = _mm512_load_ps(&C[ldc * 6 + 16]); 
  __m512 c81 = _mm512_load_ps(&C[ldc * 7 + 16]);

  __m512 c12 = _mm512_load_ps(&C[ldc * 0 + 32]);    
  __m512 c22 = _mm512_load_ps(&C[ldc * 1 + 32]); 
  __m512 c32 = _mm512_load_ps(&C[ldc * 2 + 32]); 
  __m512 c42 = _mm512_load_ps(&C[ldc * 3 + 32]); 
  __m512 c52 = _mm512_load_ps(&C[ldc * 4 + 32]); 
  __m512 c62 = _mm512_load_ps(&C[ldc * 5 + 32]); 
  __m512 c72 = _mm512_load_ps(&C[ldc * 6 + 32]); 
  __m512 c82 = _mm512_load_ps(&C[ldc * 7 + 32]);

  __m512 a0,a1,a2,b1,b2,b3,b4,b5,b6,b7,b8;
  int k;
  for(k = 0; k < K; k++)
  {
    a0 = _mm512_load_ps(&A[lda * k       ]); 
    a1 = _mm512_load_ps(&A[lda * k + 16  ]); 
    a2 = _mm512_load_ps(&A[lda * k + 32  ]); 

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

    c12 = _mm512_fmadd_ps(a2, b1, c12);
    c22 = _mm512_fmadd_ps(a2, b2, c22);
    c32 = _mm512_fmadd_ps(a2, b3, c32);
    c42 = _mm512_fmadd_ps(a2, b4, c42);
    c52 = _mm512_fmadd_ps(a2, b5, c52);
    c62 = _mm512_fmadd_ps(a2, b6, c62);
    c72 = _mm512_fmadd_ps(a2, b7, c72);
    c82 = _mm512_fmadd_ps(a2, b8, c82);

  }
  _mm512_store_ps(&C[ldc * 0], c10);          
  _mm512_store_ps(&C[ldc * 1], c20);          
  _mm512_store_ps(&C[ldc * 2], c30);          
  _mm512_store_ps(&C[ldc * 3], c40);          
  _mm512_store_ps(&C[ldc * 4], c50);          
  _mm512_store_ps(&C[ldc * 5], c60);          
  _mm512_store_ps(&C[ldc * 6], c70);          
  _mm512_store_ps(&C[ldc * 7], c80);    

  _mm512_store_ps(&C[ldc * 0 + 16], c11);          
  _mm512_store_ps(&C[ldc * 1 + 16], c21);          
  _mm512_store_ps(&C[ldc * 2 + 16], c31);          
  _mm512_store_ps(&C[ldc * 3 + 16], c41);          
  _mm512_store_ps(&C[ldc * 4 + 16], c51);          
  _mm512_store_ps(&C[ldc * 5 + 16], c61);          
  _mm512_store_ps(&C[ldc * 6 + 16], c71);          
  _mm512_store_ps(&C[ldc * 7 + 16], c81); 

  _mm512_store_ps(&C[ldc * 0 + 32], c12);          
  _mm512_store_ps(&C[ldc * 1 + 32], c22);          
  _mm512_store_ps(&C[ldc * 2 + 32], c32);          
  _mm512_store_ps(&C[ldc * 3 + 32], c42);          
  _mm512_store_ps(&C[ldc * 4 + 32], c52);          
  _mm512_store_ps(&C[ldc * 5 + 32], c62);          
  _mm512_store_ps(&C[ldc * 6 + 32], c72);          
  _mm512_store_ps(&C[ldc * 7 + 32], c82); 
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

// A : 128 * k , B : k  * 2, C : 128 * 2 => Using (8 + 2 + 16 = 26 reg)
static void do_block_avx_128k2(int lda, int ldb, int ldc, int K ,float * __restrict__ A, float * __restrict__ B, float * __restrict__ C)
{
  __m512 c10 = _mm512_load_ps(&C[ldc * 0]);    
  __m512 c20 = _mm512_load_ps(&C[ldc * 1]); 

  __m512 c11 = _mm512_load_ps(&C[ldc * 0 + 16]);    
  __m512 c21 = _mm512_load_ps(&C[ldc * 1 + 16]); 

  __m512 c12 = _mm512_load_ps(&C[ldc * 0 + 32]);    
  __m512 c22 = _mm512_load_ps(&C[ldc * 1 + 32]); 

  __m512 c13 = _mm512_load_ps(&C[ldc * 0 + 48]);    
  __m512 c23 = _mm512_load_ps(&C[ldc * 1 + 48]); 

  __m512 c14 = _mm512_load_ps(&C[ldc * 0 + 64]);    
  __m512 c24 = _mm512_load_ps(&C[ldc * 1 + 64]); 

  __m512 c15 = _mm512_load_ps(&C[ldc * 0 + 80]);    
  __m512 c25 = _mm512_load_ps(&C[ldc * 1 + 80]); 

  __m512 c16 = _mm512_load_ps(&C[ldc * 0 + 96]);    
  __m512 c26 = _mm512_load_ps(&C[ldc * 1 + 96]); 

  __m512 c17 = _mm512_load_ps(&C[ldc * 0 + 112]);    
  __m512 c27 = _mm512_load_ps(&C[ldc * 1 + 112]); 


  __m512 a0,a1,a2,a3,a4,a5,a6,a7,b1,b2,b3,b4;
  int k;
  for(k = 0; k < K; k++)
  {
    a0 = _mm512_load_ps(&A[lda * k       ]); 
    a1 = _mm512_load_ps(&A[lda * k + 16  ]); 
    a2 = _mm512_load_ps(&A[lda * k + 32  ]); 
    a3 = _mm512_load_ps(&A[lda * k + 48  ]); 

    a4 = _mm512_load_ps(&A[lda * k + 64  ]); 
    a5 = _mm512_load_ps(&A[lda * k + 80  ]); 
    a6 = _mm512_load_ps(&A[lda * k + 96  ]); 
    a7 = _mm512_load_ps(&A[lda * k + 112 ]); 


    b1 = _mm512_set1_ps(B[k + ldb * 0]);
    b2 = _mm512_set1_ps(B[k + ldb * 1]);

    c10 = _mm512_fmadd_ps(a0, b1, c10);
    c20 = _mm512_fmadd_ps(a0, b2, c20);

    c11 = _mm512_fmadd_ps(a1, b1, c11);
    c21 = _mm512_fmadd_ps(a1, b2, c21);

    c12 = _mm512_fmadd_ps(a2, b1, c12);
    c22 = _mm512_fmadd_ps(a2, b2, c22);

    c13 = _mm512_fmadd_ps(a3, b1, c13);
    c23 = _mm512_fmadd_ps(a3, b2, c23);

    c14 = _mm512_fmadd_ps(a4, b1, c14);
    c24 = _mm512_fmadd_ps(a4, b2, c24);

    c15 = _mm512_fmadd_ps(a5, b1, c15);
    c25 = _mm512_fmadd_ps(a5, b2, c25);

    c16 = _mm512_fmadd_ps(a6, b1, c16);
    c26 = _mm512_fmadd_ps(a6, b2, c26);

    c17 = _mm512_fmadd_ps(a7, b1, c17);
    c27 = _mm512_fmadd_ps(a7, b2, c27);

  }
  _mm512_store_ps(&C[ldc * 0], c10);          
  _mm512_store_ps(&C[ldc * 1], c20);                  

  _mm512_store_ps(&C[ldc * 0 + 16], c11);          
  _mm512_store_ps(&C[ldc * 1 + 16], c21);          

  _mm512_store_ps(&C[ldc * 0 + 32], c12);          
  _mm512_store_ps(&C[ldc * 1 + 32], c22);          

  _mm512_store_ps(&C[ldc * 0 + 48], c13);          
  _mm512_store_ps(&C[ldc * 1 + 48], c23);    

  _mm512_store_ps(&C[ldc * 0 + 64], c14);          
  _mm512_store_ps(&C[ldc * 1 + 64], c24);  

  _mm512_store_ps(&C[ldc * 0 + 80], c15);          
  _mm512_store_ps(&C[ldc * 1 + 80], c25);  

  _mm512_store_ps(&C[ldc * 0 + 96], c16);          
  _mm512_store_ps(&C[ldc * 1 + 96], c26);  

  _mm512_store_ps(&C[ldc * 0 + 112], c17);          
  _mm512_store_ps(&C[ldc * 1 + 112], c27);           
}

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
static void do_block_large(int lda,int ldb, int ldc, int M, int N, int K, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) 
{ 

  // acc for BLOCK % (128 * 2) == 0 
  // if((M % 128 == 0) && (N % 2 == 0))
  // {
  //   for (int j = 0; j < N; j += 2) 
  //     for (int i = 0; i < M; i += 128)
  //       do_block_avx_128k2(lda,ldb,ldc,K, A + i, B + j * ldb, C + i + j * ldc);
  //   return;
  // }

  // acc for BLOCK % (64 * 4) == 0 
  if((M % 64 == 0) && (N % 4 == 0))
  {
    for (int j = 0; j < N; j += 4) 
      for (int i = 0; i < M; i += 64)
        do_block_avx_64k4(lda,ldb,ldc,K, A + i, B + j * ldb, C + i + j * ldc);
    return;
  }

  // acc for BLOCK % (48 * 8) == 0 
  // if((M % 48 == 0) && (N % 8 == 0))
  // {
  //   for (int j = 0; j < N; j += 8) 
  //     for (int i = 0; i < M; i += 48)
  //       do_block_avx_48k8(lda,ldb,ldc,K, A + i, B + j * ldb, C + i + j * ldc);
  //   return;
  // }

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
  // pack AA = A[M-M_Left:M-M_Left+SMALL_BLOCK_SIZE][:] shape = (SMALL_BLOCK_SIZE,K) lda = SMALL_BLOCK_SIZE
  for(int jj=0; jj < K; jj++)
  {
    _mm512_store_ps(&AA[jj * SMALL_BLOCK_M_SIZE], _mm512_load_ps(&A[M - M_Left + jj * lda]));
    _mm512_store_ps(&AA[jj * SMALL_BLOCK_M_SIZE + 16], _mm512_load_ps(&A[M - M_Left + 16 + jj * lda]));
  }
  // pack BB = B[:][N-N_Left:N-N_Left+SMALL_BLOCK_SIZE] shape = (K,SMALL_BLOCK_SIZE) lda = K
  
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

      // Case 2 : pack A and C => A:SMALL_BLOCK_SIZE B:ldb C:SMALL_BLOCK_SIZE
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
      // Case 3 : pack B and C => A:lda B:K C:SMALL_BLOCK_SIZE
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
      // Case 4 : pack A and B and C => A:SMALL_BLOCK_SIZE B:K C:SMALL_BLOCK_SIZE
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

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  

void square_sgemm (int lda, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C)
{
  if(lda > 120)
  {
    float* A_BUFF = (float*)_mm_malloc(sizeof(float) * BLOCK_M_SIZE * BLOCK_K_SIZE, 256);
    float* B_BUFF = (float*)_mm_malloc(sizeof(float) * BLOCK_K_SIZE * BLOCK_N_SIZE, 256);
    // float* C_BUFF = (float*)_mm_malloc(sizeof(float) * BLOCK_M_SIZE * BLOCK_N_SIZE, 256);
    // float* A_TEST = (float*)_mm_malloc(sizeof(float) * lda * lda, 256);
    /* For each block-column of B */
    
    for (int j = 0; j < lda; j += BLOCK_N_SIZE)
    {
      int N = min (BLOCK_N_SIZE, lda-j);
      /* Accumulate block sgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_K_SIZE)
      {
        int K = min (BLOCK_K_SIZE, lda-k);
        // Packing B
        Pack_B(B + k + j*lda, B_BUFF, lda, K, N);
        /* For each block-row of A */ 
        for (int i = 0; i < lda; i += BLOCK_M_SIZE)
        {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int M = min (BLOCK_M_SIZE, lda-i);
          // Packing A
          Pack_A(A + i + k*lda, A_BUFF, lda, M, K);
          /* Perform individual block sgemm */
          do_block_large(BLOCK_M_SIZE,BLOCK_K_SIZE,lda, M, N, K, A_BUFF, B_BUFF, C + i + j*lda);
        }
      }
    }
    _mm_free(A_BUFF);
    _mm_free(B_BUFF);
    // _mm_free(C_BUFF);
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