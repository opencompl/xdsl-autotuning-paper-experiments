typedef DTYPE mKxM __attribute__((matrix_type(K,M)));
typedef DTYPE mNxK __attribute__((matrix_type(N,K)));
typedef DTYPE mNxM __attribute__((matrix_type(N,M)));

void matmul(DTYPE* restrict A, DTYPE* restrict B, DTYPE* restrict C) {
  
  mKxM a = __builtin_matrix_column_major_load(A, K, M, K);
  
  mNxK b = __builtin_matrix_column_major_load(B, N, K, N);
  
  mNxM c = __builtin_matrix_column_major_load(C, N, M, N);

  mNxM r = b * a + c;

  __builtin_matrix_column_major_store(r, C, N);
}
