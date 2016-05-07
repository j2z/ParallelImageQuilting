#include <cstdio>
#include <ctime>
#include "util.hpp"
#include "constants.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#define NUM_ITERATIONS 20
#define BLOCK_SIZE 32

__global__ void initRandom(unsigned int seed, curandState* states)
{
  int idX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  curand_init(seed, idX,0,&states[idX]);
}

__global__ void kernelRandomOutput(curandState* states, int* output, int output_width, int source_size)
{
  int idX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  output[idX] = curand(&states[idX]) % source_size;

}

void imagequilt_cuda(int texture_width, int texture_height, unsigned char* source, int* output)
{
  //initialize CUDA global memory
  unsigned char* source_cuda;
  int* output_cuda;
  //actually, I think it might be possible to store the previous 2 values in shared memory
  unsigned char* back_pointers;

  int output_height = HEIGHT_TILES*TILE_HEIGHT;
  int output_width = WIDTH_TILES*TILE_WIDTH;
  int tile_size = TILE_HEIGHT*TILE_WIDTH;
  int num_tiles = HEIGHT_TILES*WIDTH_TILES;

  size_t source_size = sizeof(unsigned char)*texture_width*texture_height*3;
  size_t output_size = sizeof(int)*output_width*output_height;

  //size_t errors_size = sizeof(float)*output_width*output_height;
  //size_t float_polar_size = sizeof(float)*tile_size*num_tiles;
  size_t char_polar_size = sizeof(unsigned char)*tile_size*num_tiles;

  cudaMalloc((void**)&source_cuda, source_size);
  cudaMalloc((void**)&output_cuda, output_size);
  cudaMalloc((void**)&back_pointers, char_polar_size);

  cudaMemcpy(source_cuda, source, source_size, cudaMemcpyHostToDevice);

  //first copy random pixels from source to output
  int seed = 15418;
  curandState *randStates;
  cudaMalloc((void**)&randStates, sizeof(curandState)*output_width*output_height);
  int numBlocks = output_width*output_height/BLOCK_SIZE;

  initRandom<<<numBlocks, BLOCK_SIZE>>>(seed, randStates);
  kernelRandomOutput<<<numBlocks, BLOCK_SIZE>>>(randStates, 
                                                output_cuda,
                                                output_width,
                                                texture_width*texture_height);

  dim3 blockDim(32, 1);
  dim3 gridDim(num_tiles);

  for (int iter = 0; iter < NUM_ITERATIONS; iter++)
  {
    //choose random grid alignment
    const int randX = std::rand() % (TILE_WIDTH);
    const int randY = std::rand() % (TILE_HEIGHT);

    //copy random pixels over
    //choose random patch from source
    //convert energies in polar and perform DP
    //copy the pixels over
    //check whether there was an improvement???

    
    //dim3 blockDim(32, 1);
    //dim3 gridDim(num_tiles/8);

    
  }

  cudaMemcpy(output, output_cuda, output_size, cudaMemcpyDeviceToHost);

  cudaFree(randStates);
  cudaFree(source_cuda);
  cudaFree(output_cuda);
}


