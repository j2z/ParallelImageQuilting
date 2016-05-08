#include <cstdio>
#include <ctime>
#include "util.hpp"
#include "constants.hpp"
#include "point.hpp"
#include "color_cu.hpp"
#include "util_cu.hpp"

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

__global__ void kernelFindBoundaries(curandState* states, int sourceWidth, int sourceHeight, int* output, int xOffset, int yOffset, unsigned char* minPaths)
{
  int tileIdx = blockIdx.y * WIDTH_TILES + blockIdx.x;
  int idX = tileIdx * POLAR_WIDTH + threadIdx.x;

  int tileX = blockIdx.x;
  int tileY = blockIdx.y;
  int colIdx = threadIdx.x;
  
  __shared__ float currentErrors[POLAR_WIDTH];
  __shared__ float nextErrors[POLAR_WIDTH];
  __shared__ char back_pointers[POLAR_HEIGHT][POLAR_WIDTH];
  __shared__ int randX;
  __shared__ int randY;

  if (colIdx == 0)
  {
    randX = curand(&states[idX]) % (sourceWidth - 2*MAX_RADIUS) + MAX_RADIUS;
    randY = curand(&states[idX]) % (sourceWidth - 2*MAX_RADIUS) + MAX_RADIUS;
  }
  __syncthreads();


  int mapX = tileX * TILE_WIDTH + TILE_WIDTH / 2 + xOffset;
  int mapY = tileY * TILE_HEIGHT + TILE_HEIGHT / 2 + yOffset;

  

  for (int i = 0; i < POLAR_HEIGHT; i++)
  {
  }
  
  

}

void imagequilt_cuda(int texture_width, int texture_height, unsigned char* source, int* output)
{
  //initialize CUDA global memory
  unsigned char* source_cuda;
  int* output_cuda;
  //actually, I think it might be possible to store the previous 2 values in shared memory
  unsigned char* min_paths;

  int output_height = (HEIGHT_TILES + 1)*TILE_HEIGHT;
  int output_width = (WIDTH_TILES + 1)*TILE_WIDTH;
  int tile_size = TILE_HEIGHT*TILE_WIDTH;
  int num_tiles = HEIGHT_TILES*WIDTH_TILES;

  size_t source_size = sizeof(unsigned char)*texture_width*texture_height*3;
  size_t output_size = sizeof(int)*output_width*output_height;
  size_t paths_size = sizeof(unsigned char)*POLAR_HEIGHT*num_tiles;

  cudaMalloc((void**)&source_cuda, source_size);
  cudaMalloc((void**)&output_cuda, output_size);
  cudaMalloc((void**)&min_paths, paths_size);

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

  dim3 blockDim(POLAR_WIDTH, 1);
  dim3 gridDim(WIDTH_TILES, HEIGHT_TILES, 1);

  for (int iter = 0; iter < ITERATIONS; iter++)
  {
    //choose random grid alignment
    const int offsetX = std::rand() % TILE_WIDTH;
    const int offsetY = std::rand() % TILE_HEIGHT;

    kernelFindBoundaries<<<gridDim, blockDim>>>(randStates, texture_width, texture_height, output_cuda, offsetX, offsetY, min_paths);
    
  }

  cudaMemcpy(output, output_cuda, output_size, cudaMemcpyDeviceToHost);

  cudaFree(randStates);
  cudaFree(source_cuda);
  cudaFree(output_cuda);
}


