/*
* This program uses the host CURAND API to generate 100
* pseudorandom floats .
*/

//C or C++ headers
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include<iostream>
#include<cmath>

//#define DENG_NOT_USING_CUBLAS_V2

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

__global__ void squares(float* x_device, int length)
{
	int thrd_id = threadIdx.x + blockIdx.x * blockDim.x;

	while (thrd_id < length)
	{
		x_device[thrd_id] = x_device[thrd_id]* x_device[thrd_id];
		thrd_id += blockDim.x* gridDim.x;
	}
}

__global__ void exps(float* x_device, int length)
{
	int thrd_id = threadIdx.x + blockIdx.x * blockDim.x;

	while (thrd_id < length)
	{
		x_device[thrd_id] = exp(x_device[thrd_id]);
		thrd_id += blockDim.x* gridDim.x;
	}
}

__global__ void array_operation(float* x_device, int length, float (*op)(float))
{
	int thrd_id = threadIdx.x + blockIdx.x * blockDim.x;

	while (thrd_id < length)
	{
		x_device[thrd_id] = op(x_device[thrd_id]);
		thrd_id += blockDim.x* gridDim.x;
	}
}

__host__ __device__ float f_integral(float t)
{
	return sin(6.28318530718*t);
}

__host__ __device__ float Deng_square(float t)
{
	return t*t;
}
__host__ __device__ float Deng_exp(float t)
{
	return exp(t);
}


int main(int argc, char * argv[])
{
	size_t n = 1024*8*8;
	size_t threads_per_block = 512;

	size_t i;

	//Create pseudo-random number generator
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	time_t seed;
	time(&seed);
	curandSetPseudoRandomGeneratorSeed(gen, seed);

	
#ifdef DENG_NOT_USING_CUBLAS_V2
#else
	cublasHandle_t cublas_hd;
	cublasStatus_t cublas_stat = cublasCreate(&cublas_hd);
#endif
	

	size_t N_t = 1024 * 8;
	float tau = 16.0;
	float dt = tau / N_t;
	float dt_sqrt = sqrt(dt);

	float beta = 1.0;
	float minus_beta = -beta;

	float omega_0 = 1.0;
	float omega_tau = 3.0;

	
	float gamma = 1.0;// 0.015625f;//  1/64
	float gamma_dt = -gamma*dt;
	float C = sqrt(2 * gamma/beta);



	float *omega, *omega_dot, *omega_omega_dot, *omega_sq;
	float *x_host, *v_host, *W_host;
	float *x_device, *v_device, *rand_device, *W_device;
	float *temp_device, *temp_device2, *swap_device;

	cudaEvent_t start_cuda, stop_cuda;
	cudaEventCreate(&start_cuda);
	cudaEventCreate(&stop_cuda);
	float ellapese_time;

	
	//Allocate n floats on host
	x_host = (float *)calloc(n, sizeof(float));
	v_host = (float *)calloc(n, sizeof(float));
	//parameter
	omega = (float *)calloc(n, sizeof(float));
	omega_dot = (float *)calloc(n, sizeof(float));
	omega_omega_dot = (float *)calloc(n, sizeof(float));
	omega_sq = (float *)calloc(n, sizeof(float));
	for (size_t i = 0; i < N_t; ++i)
	{
		omega[i] = omega_0 + (i*(omega_tau - omega_0)) / (float)n;
		omega_dot[i] = (omega_tau - omega_0) / tau;
		omega_omega_dot[i] = omega[i] * omega_dot[i];
		omega_sq[i] = omega[i] * omega[i];
	}

		
	// Allocate n floats on device
	cudaMalloc((void **)& x_device, n * sizeof(float));
	cudaMalloc((void **)& v_device, n * sizeof(float));
	cudaMalloc((void **)& rand_device, n * sizeof(float));
	cudaMalloc((void **)& temp_device, n * sizeof(float));
	cudaMalloc((void **)& temp_device2, n * sizeof(float));
	cudaMalloc((void **)& W_device, n * sizeof(float));

	//Assign x on device
	//natural_numbers<<< 2, 128 >>>(x_device, n);
	//cudaEventRecord(start_cuda, 0);
	//zeros <<<2, threads_per_block >>> (x_device, n);
	//zeros <<<2, threads_per_block >>> (v_device, n);
	curandGenerateNormal(gen, x_device, n, 0, 1 / (sqrt(beta)*omega_0));
	curandGenerateNormal(gen, v_device, n, 0, 1 / sqrt(beta));

	zeros << <2, threads_per_block >> > (W_device, n);
	//cudaEventRecord(stop_cuda, 0);
	//cudaEventSynchronize(stop_cuda);
	//cudaEventElapsedTime(&ellapese_time, start_cuda, stop_cuda);
	//std::cout << "initialize the vector once costs: " << ellapese_time << " ms\n";
	
	//start recording time
	printf("Start random walk\n");
	cudaEventRecord(start_cuda, 0);

	//simplest Euler Maruyama method
	for (size_t i = 0; i < N_t; ++i)
	{
		//Generate n floats on device 
		//curandGenerateUniform(gen, rand_device, n);
		
		//dt or squareroot of dt???
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
		float temp_host;
		//copy X
		cublasScopy(cublas_hd, n, x_device, 1, temp_device, 1);
		//calculate dW
		//array_operation <<<2, threads_per_block >>> (temp_device, n, Deng_square);
		squares << <2, threads_per_block >> > (temp_device, n);

		//if (i == 10)
		//{
		//	cudaMemcpy(x_host, temp_device, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
		//	
		//	for (int k = 0; k < 1024; ++k)
		//	{
		//		printf("%f, ", x_host[k]);
		//		if (k % 8 == 0)
		//		{
		//			printf("\n");
		//		}
		//	}
		//}
			

		temp_host = omega_omega_dot[i] * dt;
		//printf("%f\n", temp_host);
		cublasSaxpy(cublas_hd, n, &temp_host, temp_device, 1, W_device, 1);

		//copy V
		cublasScopy(cublas_hd, n, v_device, 1, temp_device, 1);
		//for Damped Harmonic Oscillator
		//calculate dV
		temp_host = -omega_sq[i] * dt;
		cublasSaxpy(cublas_hd, n, &C, rand_device, 1, temp_device, 1);
		cublasSaxpy(cublas_hd, n, &gamma_dt, v_device, 1, temp_device, 1);
		cublasSaxpy(cublas_hd, n, &temp_host, x_device, 1, temp_device, 1);
		//cublasSaxpy(n, -omega_sq*dt, x_device    , 1, temp_device, 1);
		//swap
		//cublasScopy(n, temp_device, 1, v_device, 1);
		swap_device = temp_device;
		temp_device = v_device;
		v_device = swap_device;
		//calculate dX
		cublasSaxpy(cublas_hd, n, &dt, temp_device, 1, x_device, 1);
#endif
	}
	cudaEventRecord(stop_cuda, 0);
	cudaEventSynchronize(stop_cuda);
	cudaEventElapsedTime(&ellapese_time, start_cuda, stop_cuda);
	std::cout << "simulation costs: " << ellapese_time << " ms\n";

	zeros <<<2, threads_per_block >> >(x_device, n);
	cublasSaxpy(cublas_hd, n, &minus_beta, W_device, 1, x_device, 1);
	//array_operation <<<2, 128 >> >(x_device, n,exp);
	exps << <2, threads_per_block >> >(x_device, n);

	ones<<<2, threads_per_block >>>(rand_device, n);

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
	
	//Deng::RandNumGen::LCG64 rand64(0);
	//std::clock_t start;
	//double duration;

	//float temp = 0.0f;
	//start = std::clock();
	//for (size_t i = 0; i < N_t; ++i)
	//{
	//	for (size_t j = 0; j < n; ++j)
	//	{
	//		//temp = rand64();
	//		x_host[j] += temp;
	//	}
	//}
	//duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	//std::cout << "CPU costs " << 1000* duration << " ms\n";
	
	/* Copy device memory to host */
	cudaMemcpy(x_host, x_device, n * sizeof(float), cudaMemcpyDeviceToHost);
	curandDestroyGenerator(gen);
	cudaFree(x_device);
	cudaFree(rand_device);
	free(x_host);

	return EXIT_SUCCESS;
}