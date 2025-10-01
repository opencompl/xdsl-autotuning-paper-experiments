typedef DTYPE mMxK __attribute__((matrix_type(M, K)));
typedef DTYPE mKxN __attribute__((matrix_type(K, N)));
typedef DTYPE mMxN __attribute__((matrix_type(M, N)));

typedef DTYPE mKxM __attribute__((matrix_type(K, M)));
typedef DTYPE mNxK __attribute__((matrix_type(N, K)));
typedef DTYPE mNxM __attribute__((matrix_type(N, M)));

void matmul(DTYPE* restrict A, DTYPE* restrict B, DTYPE* restrict C) {
  mMxK aT = __builtin_matrix_column_major_load(A, M, K, M);
  mMxK a  = __builtin_matrix_transpose(aT);
  
  mKxN bT = __builtin_matrix_column_major_load(B, K, N, K);
  mKxN b  = __builtin_matrix_transpose(bT);
  
  mMxN cT = __builtin_matrix_column_major_load(C, M, N, M);
  mMxN c  = __builtin_matrix_transpose(cT);

  mMxN r = a * b + c;

  mNxM rT = __builtin_matrix_transpose(r);
  __builtin_matrix_column_major_store(rT, C, M);
}
