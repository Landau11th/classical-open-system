/*
* This program uses the host CURAND API to generate 100
* pseudorandom floats .
*/

//C or C++ headers
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include<iostream>

//CUDA headers
#include <cuda.h>
#include <curand.h>
#include <cublas.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

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

float f_integral(const float t)
{
	return sin(6.28318530718*t);
}

int main(int argc, char * argv[])
{
	size_t n = 1024*8*8;
	size_t N_t = 1024 * 16;
	float tau = 1.0;
	float dt = tau / N_t;
	float dt_sq = sqrt(dt);

	size_t i;
	curandGenerator_t gen;
	float *x_host, *y_host;
	float *x_device, *y_device, *rand_device;

	cudaEvent_t start_cuda, stop_cuda;
	cudaEventCreate(&start_cuda);
	cudaEventCreate(&stop_cuda);
	float ellapese_time;

	
	//Allocate n floats on host
	x_host = (float *)calloc(n, sizeof(float));
		
	// Allocate n floats on device
	cudaMalloc((void **)& x_device, n * sizeof(float));
	cudaMalloc((void **)& rand_device, n * sizeof(float));
	


	//Assign x on device
	//natural_numbers<<< 2, 128 >>>(x_device, n);
	zeros<<<2, 256>>>(x_device, n);
	
	//Create pseudo - random number generator
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	//Set seed
	time_t seed;
	time(&seed);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
	printf("Start random walk\n");
	cudaEventRecord(start_cuda, 0);
	for (size_t i = 0; i < N_t; ++i)
	{
		//Generate n floats on device 
		//curandGenerateUniform(gen, rand_device, n);
		curandGenerateNormal(gen, rand_device, n, 0, f_integral((i*tau) / N_t)*dt_sq);
		//vector addition
		cublasSaxpy(n, 1.0, rand_device, 1, x_device, 1);
	}
	cudaEventRecord(stop_cuda, 0);
	cudaEventSynchronize(stop_cuda);
	cudaEventElapsedTime(&ellapese_time, start_cuda, stop_cuda);
	std::cout << ellapese_time << " ms\n";

	ones<<<2, 128>>>(rand_device, n);

	float average;
	std::cout << cublasSdot(n, rand_device, 1, x_device, 1)/n << std::endl;
	std::cout << cublasSdot(n, x_device, 1, x_device, 1)/n << std::endl;
	
	Deng::RandNumGen::LCG64 rand64(0);
	std::clock_t start;
	double duration;

	float temp;
	start = std::clock();
	for (size_t i = 0; i < N_t; ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			temp = rand64();
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