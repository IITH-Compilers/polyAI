void polydl_lib_matmul_f32_i_8_j_16_k_1_fma( long long int M, long long int N, long long int K,long long int A_stride, long long int B_stride, long long int C_stride, float *A, float *B, float *C) {
__m512 vec_C;
__m512 vec_B;
__m512 vec_A;
__m512 vec_C1, vec_C2, vec_C3, vec_C4, vec_C5, vec_C6, vec_C7, vec_C8, vec_C9, vec_C10, vec_C11, vec_C12, vec_C13, vec_C14, vec_C15;
__m512 vec_A1, vec_A2, vec_A3, vec_A4, vec_A5, vec_A6, vec_A7, vec_A8, vec_A9, vec_A10, vec_A11, vec_A12, vec_A13, vec_A14, vec_A15;
__m512 vec_B1;
__m512 vec_A16, vec_A17, vec_A18, vec_A19, vec_A20, vec_A21, vec_A22, vec_A23, vec_A24, vec_A25, vec_A26, vec_A27, vec_A28, vec_A29, vec_A30, vec_A31;
int i, j, k;
long long M_full, N_full, K_full;
M_full = (M / 16) * 16 ;
N_full = (N / 16) * 16 ;
for (i = 0; i < M_full; i += 16) {
for (j = 0; j < N_full; j += 16) {
vec_C = _mm512_load_ps((__m512*)&C[i*C_stride + j]);
vec_C1 = _mm512_load_ps((__m512*)&C[(i + 1)*C_stride + j]);
vec_C2 = _mm512_load_ps((__m512*)&C[(i + 2)*C_stride + j]);
vec_C3 = _mm512_load_ps((__m512*)&C[(i + 3)*C_stride + j]);
vec_C4 = _mm512_load_ps((__m512*)&C[(i + 4)*C_stride + j]);
vec_C5 = _mm512_load_ps((__m512*)&C[(i + 5)*C_stride + j]);
vec_C6 = _mm512_load_ps((__m512*)&C[(i + 6)*C_stride + j]);
vec_C7 = _mm512_load_ps((__m512*)&C[(i + 7)*C_stride + j]);
vec_C8 = _mm512_load_ps((__m512*)&C[(i + 8)*C_stride + j]);
vec_C9 = _mm512_load_ps((__m512*)&C[(i + 9)*C_stride + j]);
vec_C10 = _mm512_load_ps((__m512*)&C[(i + 10)*C_stride + j]);
vec_C11 = _mm512_load_ps((__m512*)&C[(i + 11)*C_stride + j]);
vec_C12 = _mm512_load_ps((__m512*)&C[(i + 12)*C_stride + j]);
vec_C13 = _mm512_load_ps((__m512*)&C[(i + 13)*C_stride + j]);
vec_C14 = _mm512_load_ps((__m512*)&C[(i + 14)*C_stride + j]);
vec_C15 = _mm512_load_ps((__m512*)&C[(i + 15)*C_stride + j]);
for (k = 0; k < K; k += 2) {
vec_A = _mm512_set1_ps(A[i*A_stride + k]);
vec_B = _mm512_load_ps((__m512*)&B[k*B_stride + j]);
vec_B1 = _mm512_load_ps((__m512*)&B[(k+1)*B_stride + (j+0)]);
vec_A1 = _mm512_set1_ps(A[(i+0)*A_stride + (k+1)]);
vec_A2 = _mm512_set1_ps(A[(i+1)*A_stride + (k+0)]);
vec_A3 = _mm512_set1_ps(A[(i+1)*A_stride + (k+1)]);
vec_A4 = _mm512_set1_ps(A[(i+2)*A_stride + (k+0)]);
vec_A5 = _mm512_set1_ps(A[(i+2)*A_stride + (k+1)]);
vec_A6 = _mm512_set1_ps(A[(i+3)*A_stride + (k+0)]);
vec_A7 = _mm512_set1_ps(A[(i+3)*A_stride + (k+1)]);
vec_A8 = _mm512_set1_ps(A[(i+4)*A_stride + (k+0)]);
vec_A9 = _mm512_set1_ps(A[(i+4)*A_stride + (k+1)]);
vec_A10 = _mm512_set1_ps(A[(i+5)*A_stride + (k+0)]);
vec_A11 = _mm512_set1_ps(A[(i+5)*A_stride + (k+1)]);
vec_A12 = _mm512_set1_ps(A[(i+6)*A_stride + (k+0)]);
vec_A13 = _mm512_set1_ps(A[(i+6)*A_stride + (k+1)]);
vec_A14 = _mm512_set1_ps(A[(i+7)*A_stride + (k+0)]);
vec_A15 = _mm512_set1_ps(A[(i+7)*A_stride + (k+1)]);
vec_A16 = _mm512_set1_ps(A[(i+8)*A_stride + (k+0)]);
vec_A17 = _mm512_set1_ps(A[(i+8)*A_stride + (k+1)]);
vec_A18 = _mm512_set1_ps(A[(i+9)*A_stride + (k+0)]);
vec_A19 = _mm512_set1_ps(A[(i+9)*A_stride + (k+1)]);
vec_A20 = _mm512_set1_ps(A[(i+10)*A_stride + (k+0)]);
vec_A21 = _mm512_set1_ps(A[(i+10)*A_stride + (k+1)]);
vec_A22 = _mm512_set1_ps(A[(i+11)*A_stride + (k+0)]);
vec_A23 = _mm512_set1_ps(A[(i+11)*A_stride + (k+1)]);
vec_A24 = _mm512_set1_ps(A[(i+12)*A_stride + (k+0)]);
vec_A25 = _mm512_set1_ps(A[(i+12)*A_stride + (k+1)]);
vec_A26 = _mm512_set1_ps(A[(i+13)*A_stride + (k+0)]);
vec_A27 = _mm512_set1_ps(A[(i+13)*A_stride + (k+1)]);
vec_A28 = _mm512_set1_ps(A[(i+14)*A_stride + (k+0)]);
vec_A29 = _mm512_set1_ps(A[(i+14)*A_stride + (k+1)]);
vec_A30 = _mm512_set1_ps(A[(i+15)*A_stride + (k+0)]);
vec_A31 = _mm512_set1_ps(A[(i+15)*A_stride + (k+1)]);
vec_C = _mm512_fmadd_ps(vec_A, vec_B, vec_C);
vec_C1 = _mm512_fmadd_ps(vec_A1, vec_B, vec_C1);
vec_C2 = _mm512_fmadd_ps(vec_A2, vec_B, vec_C2);
vec_C3 = _mm512_fmadd_ps(vec_A3, vec_B, vec_C3);
vec_C4 = _mm512_fmadd_ps(vec_A4, vec_B, vec_C4);
vec_C5 = _mm512_fmadd_ps(vec_A5, vec_B, vec_C5);
vec_C6 = _mm512_fmadd_ps(vec_A6, vec_B, vec_C6);
vec_C7 = _mm512_fmadd_ps(vec_A7, vec_B, vec_C7);
vec_C8 = _mm512_fmadd_ps(vec_A8, vec_B, vec_C8);
vec_C9 = _mm512_fmadd_ps(vec_A9, vec_B, vec_C9);
vec_C10 = _mm512_fmadd_ps(vec_A10, vec_B, vec_C10);
vec_C11 = _mm512_fmadd_ps(vec_A11, vec_B, vec_C11);
vec_C12 = _mm512_fmadd_ps(vec_A12, vec_B, vec_C12);
vec_C13 = _mm512_fmadd_ps(vec_A13, vec_B, vec_C13);
vec_C14 = _mm512_fmadd_ps(vec_A14, vec_B, vec_C14);
vec_C15 = _mm512_fmadd_ps(vec_A15, vec_B, vec_C15);
vec_C = _mm512_fmadd_ps(vec_A16, vec_B1, vec_C);
vec_C1 = _mm512_fmadd_ps(vec_A17, vec_B1, vec_C1);
vec_C2 = _mm512_fmadd_ps(vec_A18, vec_B1, vec_C2);
vec_C3 = _mm512_fmadd_ps(vec_A19, vec_B1, vec_C3);
vec_C4 = _mm512_fmadd_ps(vec_A20, vec_B1, vec_C4);
vec_C5 = _mm512_fmadd_ps(vec_A21, vec_B1, vec_C5);
vec_C6 = _mm512_fmadd_ps(vec_A22, vec_B1, vec_C6);
vec_C7 = _mm512_fmadd_ps(vec_A23, vec_B1, vec_C7);
vec_C8 = _mm512_fmadd_ps(vec_A24, vec_B1, vec_C8);
vec_C9 = _mm512_fmadd_ps(vec_A25, vec_B1, vec_C9);
vec_C10 = _mm512_fmadd_ps(vec_A26, vec_B1, vec_C10);
vec_C11 = _mm512_fmadd_ps(vec_A27, vec_B1, vec_C11);
vec_C12 = _mm512_fmadd_ps(vec_A28, vec_B1, vec_C12);
vec_C13 = _mm512_fmadd_ps(vec_A29, vec_B1, vec_C13);
vec_C14 = _mm512_fmadd_ps(vec_A30, vec_B1, vec_C14);
vec_C15 = _mm512_fmadd_ps(vec_A31, vec_B1, vec_C15);
}
_mm512_store_ps((__m512*)&C[i*C_stride + j], vec_C);
_mm512_store_ps((__m512*)&C[(i + 1)*C_stride + j], vec_C1);
_mm512_store_ps((__m512*)&C[(i + 2)*C_stride + j], vec_C2);
_mm512_store_ps((__m512*)&C[(i + 3)*C_stride + j], vec_C3);
_mm512_store_ps((__m512*)&C[(i + 4)*C_stride + j], vec_C4);
_mm512_store_ps((__m512*)&C[(i + 5)*C_stride + j], vec_C5);
_mm512_store_ps((__m512*)&C[(i + 6)*C_stride + j], vec_C6);
_mm512_store_ps((__m512*)&C[(i + 7)*C_stride + j], vec_C7);
_mm512_store_ps((__m512*)&C[(i + 8)*C_stride + j], vec_C8);
_mm512_store_ps((__m512*)&C[(i + 9)*C_stride + j], vec_C9);
_mm512_store_ps((__m512*)&C[(i + 10)*C_stride + j], vec_C10);
_mm512_store_ps((__m512*)&C[(i + 11)*C_stride + j], vec_C11);
_mm512_store_ps((__m512*)&C[(i + 12)*C_stride + j], vec_C12);
_mm512_store_ps((__m512*)&C[(i + 13)*C_stride + j], vec_C13);
_mm512_store_ps((__m512*)&C[(i + 14)*C_stride + j], vec_C14);
_mm512_store_ps((__m512*)&C[(i + 15)*C_stride + j], vec_C15);
}
}
} 