#include<iostream>
#include<stdlib.h>
#include<sys/time.h>
#include<immintrin.h>
#include<omp.h>
#include<cmath>        // std::abs
#include<malloc.h>
#include<time.h>
#ifdef VTUNE_ANALYSIS
#include <ittnotify.h>
#endif 


using namespace std;

#define M 1024				// 1008
#define N 1024		        // Number of rows and columns
#define K 1024		        // Number of rows and columns

#define TILE_SIZE_A 1024      	//1024   //36
#define TILE_SIZE_B 1024		//1024
#define TILE_SIZE_C 128			//128

#define ITR 100

void naiveMultiply(float *A, float *B, float *C);
void finalMultiply(float *A, float *B, float *C);
void Multithreadedfinal(float *A, float *B, float *C);


void checkMatrix(float *In, float *Ref);

int main(){

#ifdef VTUNE_ANALYSIS
    __itt_pause();
#endif

	//float A[N][N], B[N][N], C[N][N], debugC[N][N];
	
	float *A, *B, *C, *debugC;
	A = (float*) memalign(64, M * K * sizeof(float));
	B = (float*) memalign(64, K * N * sizeof(float));
	C = (float*) memalign(64, M * N * sizeof(float));
	debugC = (float*) memalign(64, M * N * sizeof(float));
	

	float LO = -1;						// Low range of random numbers
	float HI =  1;						// High range of random numbers
	srand(time(0));							// constant seed for consistency
	
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < K; j++)
		{
			A[K*i + j] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
		}
	}

	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < N; j++)
		{
			B[N*i + j] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
		}
	}

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			/*Generate random numbers in a range for input matrices and zeros for the output matrix*/
			// A[N*i + j] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
			// B[N*i + j] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
			C[N*i + j] = 0.0;
			debugC[N*i + j] = 0.0;
		}
	}
	
	naiveMultiply(A,B,debugC); 

    double cpu_time_used, total_time;
	struct timeval start, end;
	// struct timespec start, end;

	total_time = 0;
	for(int n=0; n < ITR; n++){

		for (int i = 0; i < M; i++){
			for (int j = 0; j < N; j++){
				/* zeros for the output matrix*/
				C[N*i + j] = 0.0;
			}
		}
		gettimeofday(&start, NULL);     
		// clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
			finalMultiply(A,B,C);                      //kernel
		gettimeofday(&end, NULL);
		// clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
		cpu_time_used = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
		// cout << "Iteration: " << n << "\t"<< cpu_time_used << endl;
		// cout << end.tv_sec << "\t" << start.tv_sec << "\t" << end.tv_usec << "\t" << start.tv_usec << endl;
		total_time += cpu_time_used;
	}

	cout<<"                          final: "<<(total_time/ITR)*1000 << endl;

	// checkMatrix(C, debugC);

	total_time = 0;
	for(int n=0; n < ITR; n++){

		for (int i = 0; i < M; i++){
			for (int j = 0; j < N; j++){
				/* zeros for the output matrix*/
				C[N*i + j] = 0.0;
			}
		}

		gettimeofday(&start, NULL);      
#ifdef VTUNE_ANALYSIS
    __itt_resume();
#endif                   
		Multithreadedfinal(A,B,C);                      //kernel
#ifdef VTUNE_ANALYSIS
    __itt_pause();
#endif
		gettimeofday(&end, NULL);
		cpu_time_used = ((end.tv_sec  - start.tv_sec) * 1000000u + 
			end.tv_usec - start.tv_usec) / 1.e6;
		total_time += cpu_time_used;
		// cout << "Iteration: " << n << "\t"<< cpu_time_used << endl;
	}
	// cout<<"            Multithreaded final: "<<total_time/ITR << endl;

	// checkMatrix(C, debugC);

	return 0;
}

void naiveMultiply(float *A, float *B, float *C){

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			for (int k = 0; k < K; k++)
			{
				C[N*i + j] += A[K*i + k] * B[N*k + j];
			}
		}
	}
}

void finalMultiply(float *A, float *B, float *C){
	
	#pragma omp parallel for
	for (int a = 0; a < M; a += TILE_SIZE_A){
		for (int b = 0; b < N; b += TILE_SIZE_B){
			for (int c = 0; c < K; c += TILE_SIZE_C){
				for (int i = a; i < a + TILE_SIZE_A; i+=4){
					for (int j = b; j < b + TILE_SIZE_B; j += 64){
						__m512 vec_C1_1 = _mm512_load_ps((__m512*)&C[N*i + j]);
						__m512 vec_C1_2 = _mm512_load_ps((__m512*)&C[N*i + j+16]);
						__m512 vec_C1_3 = _mm512_load_ps((__m512*)&C[N*i + j+32]);
						__m512 vec_C1_4 = _mm512_load_ps((__m512*)&C[N*i + j+48]);

						__m512 vec_C2_1 = _mm512_load_ps((__m512*)&C[N*(i+1) + j]);
						__m512 vec_C2_2 = _mm512_load_ps((__m512*)&C[N*(i+1) + j+16]);
						__m512 vec_C2_3 = _mm512_load_ps((__m512*)&C[N*(i+1) + j+32]);
						__m512 vec_C2_4 = _mm512_load_ps((__m512*)&C[N*(i+1) + j+48]);

						__m512 vec_C3_1 = _mm512_load_ps((__m512*)&C[N*(i+2) + j]);
						__m512 vec_C3_2 = _mm512_load_ps((__m512*)&C[N*(i+2) + j+16]);
						__m512 vec_C3_3 = _mm512_load_ps((__m512*)&C[N*(i+2) + j+32]);
						__m512 vec_C3_4 = _mm512_load_ps((__m512*)&C[N*(i+2) + j+48]);

						__m512 vec_C4_1 = _mm512_load_ps((__m512*)&C[N*(i+3) + j]);
						__m512 vec_C4_2 = _mm512_load_ps((__m512*)&C[N*(i+3) + j+16]);
						__m512 vec_C4_3 = _mm512_load_ps((__m512*)&C[N*(i+3) + j+32]);
						__m512 vec_C4_4 = _mm512_load_ps((__m512*)&C[N*(i+3) + j+48]);

						for (int k = c; k < c + TILE_SIZE_C; k++){
							// C[i][j] += A[i][k] * B[k][j];

							__m512 vec_A1_1 = _mm512_set1_ps(A[K*i + k]);
							__m512 vec_A2_1 = _mm512_set1_ps(A[K*(i+1) + k]);
							__m512 vec_A3_1 = _mm512_set1_ps(A[K*(i+2) + k]);
							__m512 vec_A4_1 = _mm512_set1_ps(A[K*(i+3) + k]);

							vec_C1_1 = _mm512_fmadd_ps(vec_A1_1, _mm512_load_ps((__m512*)&B[N*k + j]), vec_C1_1);
							vec_C1_2 = _mm512_fmadd_ps(vec_A1_1, _mm512_load_ps((__m512*)&B[N*k + j+16]), vec_C1_2);
							vec_C1_3 = _mm512_fmadd_ps(vec_A1_1, _mm512_load_ps((__m512*)&B[N*k + j+32]), vec_C1_3);
							vec_C1_4 = _mm512_fmadd_ps(vec_A1_1, _mm512_load_ps((__m512*)&B[N*k + j+48]), vec_C1_4);
							
							vec_C2_1 = _mm512_fmadd_ps(vec_A2_1, _mm512_load_ps((__m512*)&B[N*k + j]), vec_C2_1);
							vec_C2_2 = _mm512_fmadd_ps(vec_A2_1, _mm512_load_ps((__m512*)&B[N*k + j+16]), vec_C2_2);
							vec_C2_3 = _mm512_fmadd_ps(vec_A2_1, _mm512_load_ps((__m512*)&B[N*k + j+32]), vec_C2_3);
							vec_C2_4 = _mm512_fmadd_ps(vec_A2_1, _mm512_load_ps((__m512*)&B[N*k + j+48]), vec_C2_4);
							
							vec_C3_1 = _mm512_fmadd_ps(vec_A3_1, _mm512_load_ps((__m512*)&B[N*k + j]), vec_C3_1);
							vec_C3_2 = _mm512_fmadd_ps(vec_A3_1, _mm512_load_ps((__m512*)&B[N*k + j+16]), vec_C3_2);
							vec_C3_3 = _mm512_fmadd_ps(vec_A3_1, _mm512_load_ps((__m512*)&B[N*k + j+32]), vec_C3_3);
							vec_C3_4 = _mm512_fmadd_ps(vec_A3_1, _mm512_load_ps((__m512*)&B[N*k + j+48]), vec_C3_4);

							vec_C4_1 = _mm512_fmadd_ps(vec_A4_1, _mm512_load_ps((__m512*)&B[N*k + j]), vec_C4_1);
							vec_C4_2 = _mm512_fmadd_ps(vec_A4_1, _mm512_load_ps((__m512*)&B[N*k + j+16]), vec_C4_2);
							vec_C4_3 = _mm512_fmadd_ps(vec_A4_1, _mm512_load_ps((__m512*)&B[N*k + j+32]), vec_C4_3);
							vec_C4_4 = _mm512_fmadd_ps(vec_A4_1, _mm512_load_ps((__m512*)&B[N*k + j+48]), vec_C4_4);

						}
						_mm512_store_ps((__m512*)&C[N*i + j], vec_C1_1);
						_mm512_store_ps((__m512*)&C[N*i + j+16], vec_C1_2);
						_mm512_store_ps((__m512*)&C[N*i + j+32], vec_C1_3);
						_mm512_store_ps((__m512*)&C[N*i + j+48], vec_C1_4);

						_mm512_store_ps((__m512*)&C[N*(i+1) + j], vec_C2_1);
						_mm512_store_ps((__m512*)&C[N*(i+1) + j+16], vec_C2_2);
						_mm512_store_ps((__m512*)&C[N*(i+1) + j+32], vec_C2_3);
						_mm512_store_ps((__m512*)&C[N*(i+1) + j+48], vec_C2_4);

						_mm512_store_ps((__m512*)&C[N*(i+2) + j], vec_C3_1);
						_mm512_store_ps((__m512*)&C[N*(i+2) + j+16], vec_C3_2);
						_mm512_store_ps((__m512*)&C[N*(i+2) + j+32], vec_C3_3);
						_mm512_store_ps((__m512*)&C[N*(i+2) + j+48], vec_C3_4);

						_mm512_store_ps((__m512*)&C[N*(i+3) + j], vec_C4_1);
						_mm512_store_ps((__m512*)&C[N*(i+3) + j+16], vec_C4_2);
						_mm512_store_ps((__m512*)&C[N*(i+3) + j+32], vec_C4_3);
						_mm512_store_ps((__m512*)&C[N*(i+3) + j+48], vec_C4_4);
					}
				}
			}
		}
	}
}


void Multithreadedfinal(float *A, float *B, float *C){


	#pragma omp parallel
	{
		int32_t num_threads = omp_get_num_threads();
    	int32_t tid = omp_get_thread_num();
		const int32_t work = M;
		// const int32_t chunksize = (work % num_threads == 0) ? (work / num_threads) : ((work / num_threads) + 1);
		const int32_t chunksize = 36;
		int32_t thr_begin = (tid * chunksize < work) ? (tid * chunksize) : work; 
    	int32_t thr_end = ((tid + 1) * chunksize < work) ? ((tid+1) * chunksize) : work; 
		if (tid == num_threads-1)
			thr_end = work;
		for (int b = 0; b < N; b += TILE_SIZE_B){
			for (int c = 0; c < K; c += TILE_SIZE_C){
				for (int i = thr_begin; i < thr_end; i+=4){
					for (int j = b; j < b + TILE_SIZE_B; j += 64){
						__m512 vec_C1_1 = _mm512_load_ps((__m512*)&C[N*i + j]);
						__m512 vec_C1_2 = _mm512_load_ps((__m512*)&C[N*i + j+16]);
						__m512 vec_C1_3 = _mm512_load_ps((__m512*)&C[N*i + j+32]);
						__m512 vec_C1_4 = _mm512_load_ps((__m512*)&C[N*i + j+48]);

						__m512 vec_C2_1 = _mm512_load_ps((__m512*)&C[N*(i+1) + j]);
						__m512 vec_C2_2 = _mm512_load_ps((__m512*)&C[N*(i+1) + j+16]);
						__m512 vec_C2_3 = _mm512_load_ps((__m512*)&C[N*(i+1) + j+32]);
						__m512 vec_C2_4 = _mm512_load_ps((__m512*)&C[N*(i+1) + j+48]);

						__m512 vec_C3_1 = _mm512_load_ps((__m512*)&C[N*(i+2) + j]);
						__m512 vec_C3_2 = _mm512_load_ps((__m512*)&C[N*(i+2) + j+16]);
						__m512 vec_C3_3 = _mm512_load_ps((__m512*)&C[N*(i+2) + j+32]);
						__m512 vec_C3_4 = _mm512_load_ps((__m512*)&C[N*(i+2) + j+48]);

						__m512 vec_C4_1 = _mm512_load_ps((__m512*)&C[N*(i+3) + j]);
						__m512 vec_C4_2 = _mm512_load_ps((__m512*)&C[N*(i+3) + j+16]);
						__m512 vec_C4_3 = _mm512_load_ps((__m512*)&C[N*(i+3) + j+32]);
						__m512 vec_C4_4 = _mm512_load_ps((__m512*)&C[N*(i+3) + j+48]);


						for (int k = c; k < c + TILE_SIZE_C; k++)
						{
							// C[i][j] += A[i][k] * B[k][j];

							__m512 vec_A1_1 = _mm512_set1_ps(A[K*i + k]);
							__m512 vec_A2_1 = _mm512_set1_ps(A[K*(i+1) + k]);
							__m512 vec_A3_1 = _mm512_set1_ps(A[K*(i+2) + k]);
							__m512 vec_A4_1 = _mm512_set1_ps(A[K*(i+3) + k]);

							// vec_C1_1 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*i + k]), _mm512_load_ps((__m512*)&B[N*k + j]), vec_C1_1);
							// vec_C1_2 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*i + k]), _mm512_load_ps((__m512*)&B[N*k + j+16]), vec_C1_2);
							// vec_C1_3 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*i + k]), _mm512_load_ps((__m512*)&B[N*k + j+32]), vec_C1_3);
							// vec_C1_4 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*i + k]), _mm512_load_ps((__m512*)&B[N*k + j+48]), vec_C1_4);
							
							// vec_C2_1 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*(i+1) + k]), _mm512_load_ps((__m512*)&B[N*k + j]), vec_C2_1);
							// vec_C2_2 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*(i+1) + k]), _mm512_load_ps((__m512*)&B[N*k + j+16]), vec_C2_2);
							// vec_C2_3 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*(i+1) + k]), _mm512_load_ps((__m512*)&B[N*k + j+32]), vec_C2_3);
							// vec_C2_4 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*(i+1) + k]), _mm512_load_ps((__m512*)&B[N*k + j+48]), vec_C2_4);
							
							// vec_C3_1 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*(i+2) + k]), _mm512_load_ps((__m512*)&B[N*k + j]), vec_C3_1);
							// vec_C3_2 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*(i+2) + k]), _mm512_load_ps((__m512*)&B[N*k + j+16]), vec_C3_2);
							// vec_C3_3 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*(i+2) + k]), _mm512_load_ps((__m512*)&B[N*k + j+32]), vec_C3_3);
							// vec_C3_4 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*(i+2) + k]), _mm512_load_ps((__m512*)&B[N*k + j+48]), vec_C3_4);

							// vec_C4_1 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*(i+3) + k]), _mm512_load_ps((__m512*)&B[N*k + j]), vec_C4_1);
							// vec_C4_2 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*(i+3) + k]), _mm512_load_ps((__m512*)&B[N*k + j+16]), vec_C4_2);
							// vec_C4_3 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*(i+3) + k]), _mm512_load_ps((__m512*)&B[N*k + j+32]), vec_C4_3);
							// vec_C4_4 = _mm512_fmadd_ps(_mm512_set1_ps(A[K*(i+3) + k]), _mm512_load_ps((__m512*)&B[N*k + j+48]), vec_C4_4);


							vec_C1_1 = _mm512_fmadd_ps(vec_A1_1, _mm512_load_ps((__m512*)&B[N*k + j]), vec_C1_1);
							vec_C1_2 = _mm512_fmadd_ps(vec_A1_1, _mm512_load_ps((__m512*)&B[N*k + j+16]), vec_C1_2);
							vec_C1_3 = _mm512_fmadd_ps(vec_A1_1, _mm512_load_ps((__m512*)&B[N*k + j+32]), vec_C1_3);
							vec_C1_4 = _mm512_fmadd_ps(vec_A1_1, _mm512_load_ps((__m512*)&B[N*k + j+48]), vec_C1_4);
						
							vec_C2_1 = _mm512_fmadd_ps(vec_A2_1, _mm512_load_ps((__m512*)&B[N*k + j]), vec_C2_1);
							vec_C2_2 = _mm512_fmadd_ps(vec_A2_1, _mm512_load_ps((__m512*)&B[N*k + j+16]), vec_C2_2);
							vec_C2_3 = _mm512_fmadd_ps(vec_A2_1, _mm512_load_ps((__m512*)&B[N*k + j+32]), vec_C2_3);
							vec_C2_4 = _mm512_fmadd_ps(vec_A2_1, _mm512_load_ps((__m512*)&B[N*k + j+48]), vec_C2_4);
							
							vec_C3_1 = _mm512_fmadd_ps(vec_A3_1, _mm512_load_ps((__m512*)&B[N*k + j]), vec_C3_1);
							vec_C3_2 = _mm512_fmadd_ps(vec_A3_1, _mm512_load_ps((__m512*)&B[N*k + j+16]), vec_C3_2);
							vec_C3_3 = _mm512_fmadd_ps(vec_A3_1, _mm512_load_ps((__m512*)&B[N*k + j+32]), vec_C3_3);
							vec_C3_4 = _mm512_fmadd_ps(vec_A3_1, _mm512_load_ps((__m512*)&B[N*k + j+48]), vec_C3_4);

							vec_C4_1 = _mm512_fmadd_ps(vec_A4_1, _mm512_load_ps((__m512*)&B[N*k + j]), vec_C4_1);
							vec_C4_2 = _mm512_fmadd_ps(vec_A4_1, _mm512_load_ps((__m512*)&B[N*k + j+16]), vec_C4_2);
							vec_C4_3 = _mm512_fmadd_ps(vec_A4_1, _mm512_load_ps((__m512*)&B[N*k + j+32]), vec_C4_3);
							vec_C4_4 = _mm512_fmadd_ps(vec_A4_1, _mm512_load_ps((__m512*)&B[N*k + j+48]), vec_C4_4);

						}
						_mm512_store_ps((__m512*)&C[N*i + j], vec_C1_1);
						_mm512_store_ps((__m512*)&C[N*i + j+16], vec_C1_2);
						_mm512_store_ps((__m512*)&C[N*i + j+32], vec_C1_3);
						_mm512_store_ps((__m512*)&C[N*i + j+48], vec_C1_4);

						_mm512_store_ps((__m512*)&C[N*(i+1) + j], vec_C2_1);
						_mm512_store_ps((__m512*)&C[N*(i+1) + j+16], vec_C2_2);
						_mm512_store_ps((__m512*)&C[N*(i+1) + j+32], vec_C2_3);
						_mm512_store_ps((__m512*)&C[N*(i+1) + j+48], vec_C2_4);

						_mm512_store_ps((__m512*)&C[N*(i+2) + j], vec_C3_1);
						_mm512_store_ps((__m512*)&C[N*(i+2) + j+16], vec_C3_2);
						_mm512_store_ps((__m512*)&C[N*(i+2) + j+32], vec_C3_3);
						_mm512_store_ps((__m512*)&C[N*(i+2) + j+48], vec_C3_4);

						_mm512_store_ps((__m512*)&C[N*(i+3) + j], vec_C4_1);
						_mm512_store_ps((__m512*)&C[N*(i+3) + j+16], vec_C4_2);
						_mm512_store_ps((__m512*)&C[N*(i+3) + j+32], vec_C4_3);
						_mm512_store_ps((__m512*)&C[N*(i+3) + j+48], vec_C4_4);

					}
				}
			}
		}
	}
}


void checkMatrix(float *In, float *Ref){
    int diff=0;
    for(int i = 0;  i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
			if (abs(In[N*i + j] - Ref[N*i + j]) > 0.00001)
            	diff++ ;
        }
    }

	cout << "Matrix value differences: " << diff <<endl;
}
