#include <stdio.h>
void print_f32_polydl(
	long long int rank, long long int offset,
	long long int size1, long long int size2,
	long long int stride1, long long int stride2,
	void *base) {
	int i, j;
	printf("rank = %ld, offset = %ld, size1 = %ld, size2 = %ld, stride1 = %ld, stride2 = %ld",
		rank, offset, size1, size2, stride1, stride2);
	float *ptr = (float*)base;
	for (i = 0; i < size1; i++) {
		for (j = 0; j < size2; j++) {
			printf("%f ", ptr[i*stride1 + j * stride2 + offset]);
		}
	}

}

void polydl_lib_matmul_f32(
	long long int M, long long int N, long long int K,
	long long int A_stride, long long int B_stride, long long int C_stride,
	float *A, float *B, float *C) {

	printf("In polydl_lib_matmul_f32 function\n");
	printf("M = %ld, N = %ld, K = %ld, A_stride = %ld, B_stride = %ld, C_stride = %ld\n",
		M, N, K, A_stride, B_stride, C_stride);

	int i, j, k;
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < K; k++) {
				C[i*C_stride + j] +=
					A[i*A_stride + k] * B[k*B_stride + j];
			}
		}
	}
}
