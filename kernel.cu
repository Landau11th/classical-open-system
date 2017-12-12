/*
* This program uses the host CURAND API to generate 100
* pseudorandom floats .
*/

//C or C++ headers
#include <stdio.h>
#include <stdlib.h>
#include <ctime>


//CUDA headers
#include <cuda.h>
#include <curand.h>
#include <cublas.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

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

int main(int argc, char * argv[])
{
	size_t n = 1024;
	size_t i;
	curandGenerator_t gen;
	float *x_host, *x_device, *rand_device;
	
	//Allocate n floats on host
	x_host = (float *)calloc(n, sizeof(float));
		
	// Allocate n floats on device
	cudaMalloc((void **)& x_device, n * sizeof(float));
	cudaMalloc((void **)& rand_device, n * sizeof(float));
	
	//Assign x on device
	natural_numbers<<< 2, 32 >>>(x_device, n);
	
	//Create pseudo - random number generator
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	//Set seed
	time_t seed;
	time(&seed);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
	//Generate n floats on device 
	curandGenerateUniform(gen, rand_device, n);

	//vector addition
	cublasSaxpy(n, 1.0,
		rand_device, 1,
		x_device, 1);
	
	
	/* Copy device memory to host */
	cudaMemcpy(x_host, x_device, n * sizeof(float), cudaMemcpyDeviceToHost);
	/* Show result */
	for (i = 0; i < n; i++) {
		if (i % 8 != 0)
			printf(", ");
		else
			printf("\n");

		printf("%1.7f", x_host[i]);
	}
	printf("\n");
	// Cleanup
	curandDestroyGenerator(gen);
	cudaFree(x_device);
	cudaFree(rand_device);
	free(x_host);

	return EXIT_SUCCESS;
}