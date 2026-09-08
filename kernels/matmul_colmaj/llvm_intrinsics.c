// Clang's matrix types are column-major, so the kernel's matrices load and
// store with no transposition: A is M*K, B is K*N and C is M*N, with leading
// dimensions M, K and M.
typedef DTYPE mMxK __attribute__((matrix_type(M,K)));
typedef DTYPE mKxN __attribute__((matrix_type(K,N)));
typedef DTYPE mMxN __attribute__((matrix_type(M,N)));

void matmul(DTYPE* restrict A, DTYPE* restrict B, DTYPE* restrict C) {

  mMxK a = __builtin_matrix_column_major_load(A, M, K, M);

  mKxN b = __builtin_matrix_column_major_load(B, K, N, K);

  mMxN c = __builtin_matrix_column_major_load(C, M, N, M);

  mMxN r = a * b + c;

  __builtin_matrix_column_major_store(r, C, M);
}
