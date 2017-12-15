/*
* This program uses the host CURAND API to generate 100
* pseudorandom floats .
*/

//C or C++ headers
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include<iostream>

#define DENG_NOT_USING_CUBLAS_V2

//CUDA headers
#include <cuda.h>
#include <curand.h>

#ifdef DENG_NOT_USING_CUBLAS_V2
#include <cublas.h>
#else
#include <cublas_v2.h>
#endif

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
////CUDA thrust
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/copy.h>
//#include <thrust/for_each.h>

//my headers
#include "rand64.hpp"

#define CUDA_CALL (x) do{ if (( x) != cudaSuccess ) { printf (" Error at % s :% d\ n" , __FILE__ , __LINE__ ) ;return EXIT_FAILURE ;}}while(0)

#define CURAND_CALL (x) do{ if (( x) != CURAND_STATUS_SUCCESS ) { printf (" Error at % s :% d\ n" , __FILE__ , __LINE__ ) ;return EXIT_FAILURE ;}}while(0)


__global__ void natural_numbers(float* x_device, int length)
{
	int thrd_id = threadIdx.x + blockIdx.x * blockDim.x;

	while (thrd_id < length)
	{
		x_device[thrd_id] = thrd_id;
		thrd_id += blockDim.x* gridDim.x;
	}
}

__global__ void zeros(float* x_device, int length)
{
	////slightly faster, but not safe
	//int per_thrd = (length ) / (blockDim.x* gridDim.x);
	//int thrd_id = threadIdx.x + blockIdx.x * blockDim.x;

	//for (int i = thrd_id; i < thrd_id + per_thrd; ++i)
	//{
	//	x_device[i] = 0.0f;
	//}

	int thrd_id = threadIdx.x + blockIdx.x * blockDim.x;

	while (thrd_id < length)
	{
		x_device[thrd_id] = 0.0f;
		thrd_id += blockDim.x* gridDim.x;
	}
}

__global__ void ones(float* x_device, int length)
{
	int thrd_id = threadIdx.x + blockIdx.x * blockDim.x;

	while (thrd_id < length)
	{
		x_device[thrd_id] = 1.0f;
		thrd_id += blockDim.x* gridDim.x;
	}
}

__global__ void my_add_vector(size_t length, float* a, float*b)
{
	int thrd_id = threadIdx.x + blockIdx.x * blockDim.x;

	while (thrd_id < length)
	{
		b[thrd_id] += 1.0* a[thrd_id];
		thrd_id += blockDim.x* gridDim.x;
	}
}

__host__ __device__ float f_integral(float t)
{
	return sin(6.28318530718*t);
}


int main(int argc, char * argv[])
{
	size_t n = 1024*8*8;
	size_t threads_per_block = 512;

	size_t i;
	curandGenerator_t gen;

	
#ifdef DENG_NOT_USING_CUBLAS_V2
#else
	cublasHandle_t cublas_hd;
	cublasCreate(&cublas_hd);
#endif
	

	float *x_host, *v_host;
	float *x_device, *v_device, *rand_device;
	float *temp_device, *temp_device2, *swap_device;

	cudaEvent_t start_cuda, stop_cuda;
	cudaEventCreate(&start_cuda);
	cudaEventCreate(&stop_cuda);
	float ellapese_time;

	
	//Allocate n floats on host
	x_host = (float *)calloc(n, sizeof(float));
	v_host = (float *)calloc(n, sizeof(float));
		
	// Allocate n floats on device
	cudaMalloc((void **)& x_device, n * sizeof(float));
	cudaMalloc((void **)& v_device, n * sizeof(float));
	cudaMalloc((void **)& rand_device, n * sizeof(float));
	cudaMalloc((void **)& temp_device, n * sizeof(float));
	cudaMalloc((void **)& temp_device2, n * sizeof(float));

	//Assign x on device
	//natural_numbers<<< 2, 128 >>>(x_device, n);
	//cudaEventRecord(start_cuda, 0);
	zeros <<<2, threads_per_block >>> (x_device, n);
	zeros <<<2, threads_per_block >>> (v_device, n);
	//cudaEventRecord(stop_cuda, 0);
	//cudaEventSynchronize(stop_cuda);
	//cudaEventElapsedTime(&ellapese_time, start_cuda, stop_cuda);
	//std::cout << "initialize the vector once costs: " << ellapese_time << " ms\n";
	
	//Create pseudo - random number generator
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	//Set seed
	time_t seed;
	time(&seed);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
	printf("Start random walk\n");
	cudaEventRecord(start_cuda, 0);

	size_t N_t = 1024 * 16;
	float tau = 16.0;
	float dt = tau / N_t;
	float dt_sqrt = sqrt(dt);

	float omega_sq = 1.0f;
	float C = 1.0f;
	float gamma = 8.0;// 0.015625f;//  1/64
	float gamma_dt = -gamma*dt;


	//simplest Euler Maruyama method
	for (size_t i = 0; i < N_t; ++i)
	{
		//Generate n floats on device 
		//curandGenerateUniform(gen, rand_device, n);
		curandGenerateNormal(gen, rand_device, n, 0, dt_sqrt);

#ifdef DENG_NOT_USING_CUBLAS_V2
		//copy V
		cublasScopy(n, v_device, 1, temp_device, 1);
		//cublasScopy(n, v_device, 1, temp_device2, 1);
		//for Damped Harmonic Oscillator
		//vector addition for V		
		cublasSaxpy(n, C           , rand_device , 1, temp_device, 1);
		cublasSaxpy(n, -gamma*dt   , v_device    , 1, temp_device, 1);
		//cublasSaxpy(n, -omega_sq*dt, x_device    , 1, temp_device, 1);
		//swap
		//cublasScopy(n, temp_device, 1, v_device, 1);
		swap_device = temp_device;
		temp_device = v_device;
		v_device = swap_device;
		//vector addition for X
		cublasSaxpy(n,           dt, temp_device, 1, x_device   , 1);


		//my_add_vector<<< 3, threads_per_block>>>(n, rand_device, x_device);
		//zeros<<<2, threads_per_block >>>(rand_device, n);
#else
		//copy V
		cublasScopy(cublas_hd, n, v_device, 1, temp_device, 1);
		//cublasScopy(n, v_device, 1, temp_device2, 1);
		//for Damped Harmonic Oscillator
		//vector addition for V		
		cublasSaxpy(cublas_hd, n, &C, rand_device, 1, temp_device, 1);
		cublasSaxpy(cublas_hd, n, &gamma_dt, v_device, 1, temp_device, 1);
		//cublasSaxpy(n, -omega_sq*dt, x_device    , 1, temp_device, 1);
		//swap
		//cublasScopy(n, temp_device, 1, v_device, 1);
		swap_device = temp_device;
		temp_device = v_device;
		v_device = swap_device;
		//vector addition for X
		cublasSaxpy(cublas_hd, n, &dt, temp_device, 1, x_device, 1);
#endif


	}
	cudaEventRecord(stop_cuda, 0);
	cudaEventSynchronize(stop_cuda);
	cudaEventElapsedTime(&ellapese_time, start_cuda, stop_cuda);
	std::cout << "simulation costs: " << ellapese_time << " ms\n";

	ones<<<2, 128>>>(rand_device, n);

	float average;
	float variance;

#ifdef DENG_NOT_USING_CUBLAS_V2
	average = cublasSdot(n, rand_device, 1, x_device, 1);
	variance = cublasSdot(n, x_device, 1, x_device, 1);
#else
	cublasSdot(cublas_hd, n, rand_device, 1, x_device, 1, &average);
	cublasSdot(cublas_hd, n, x_device, 1, x_device, 1, &variance);
#endif
	average = average / n;
	variance = variance / n;
		
	std::cout << "average " << average << std::endl;
	std::cout << "variance " << variance << std::endl;

	//printf("average %.6f\n", cublasSdot(n, rand_device, 1, x_device, 1) / n);
	//printf("deviation %.6f\n", cublasSdot(n, x_device, 1, x_device, 1) / n);
	
	Deng::RandNumGen::LCG64 rand64(0);
	std::clock_t start;
	double duration;

	float temp = 0.0f;
	start = std::clock();
	for (size_t i = 0; i < N_t; ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			//temp = rand64();
			x_host[j] += temp;
		}
	}
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "CPU costs " << 1000* duration << " ms\n";
	
	/* Copy device memory to host */
	cudaMemcpy(x_host, x_device, n * sizeof(float), cudaMemcpyDeviceToHost);
	//// Show result
	//for (i = 0; i < n; i++) {
	//	if (i % 8 != 0)
	//		printf(", ");
	//	else
	//		printf("\n");

	//	printf("%1.7f", x_host[i]);
	//}
	//printf("\n");
	// Cleanup
	curandDestroyGenerator(gen);
	cudaFree(x_device);
	cudaFree(rand_device);
	free(x_host);

	return EXIT_SUCCESS;
}