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

#define CUDA_CALL (x) do{ if (( x) != cudaSuccess ) { printf (" Error at % s :% d\ n" , __FILE__ , __LINE__ ) ;return EXIT_FAILURE ;}}while(0)
#define CURAND_CALL (x) do{ if (( x) != CURAND_STATUS_SUCCESS ) { printf (" Error at % s :% d\ n" , __FILE__ , __LINE__ ) ;return EXIT_FAILURE ;}}while(0)



int main(int argc, char * argv[])
{
	size_t n = 100;
	size_t i;
	curandGenerator_t gen;
	float * devData, *hostData;
	/* Allocate n floats on host */
	hostData = (float *)calloc(n, sizeof(float));
	/* Allocate n floats on device */
	cudaMalloc((void **)& devData, n * sizeof(float));
	/* Create pseudo - random number generator */
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	/* Set seed */
	time_t seed;
	time(&seed);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
	/* Generate n floats on device */
	curandGenerateUniform(gen, devData, n);
	/* Copy device memory to host */
	cudaMemcpy(hostData, devData, n * sizeof(float), cudaMemcpyDeviceToHost);
	/* Show result */
	for (i = 0; i < n; i++) {
		if (i % 8 != 0)
			printf(", ");
		else
			printf("\n");

		printf("%1.7f", hostData[i]);
	}
	printf("\n");
	/* Cleanup */
	curandDestroyGenerator(gen);
	cudaFree(devData);
	free(hostData);	return EXIT_SUCCESS;
}