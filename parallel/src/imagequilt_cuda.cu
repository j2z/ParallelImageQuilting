#include <cstdio>
#include <ctime>
#include "util.hpp"
#include "constants.hpp"
#include "point.hpp"

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

inline __device__ Point offsetToPolar(int x, int y)
{
  Point out;
  out.x = sqrt((float) (x*x + y*y)) * RADIUS_FACTOR;
  out.y = atan2(y,x);
  if (out.y < 0.f)
  {
    out.y += 2 * M_PI;
  }
  out.y *= ANGLE_FACTOR;
  return out;
}

inline __device__ Point polarToOffset(int r, int theta)
{
  Point out;
  float rEff = r / RADIUS_FACTOR;
  float thetaEff = theta / ANGLE_FACTOR;
  out.x = rEff * cos(thetaEff);
  out.y = rEff * sin(thetaEff);
  return out;
}

inline __device__ Point absoluteToPolar(int cx, int cy, int x, int y)
{
  return offsetToPolar(x-cx, y-cy);
}

inline __device__ Point polarToAbsolute(int cx, int cy, int r, int theta)
{
  Point out = polarToOffset(r, theta);
  out.x += cx;
  out.y += cy;
  return out;
}


__global__ void kernelFindBoundaries(int* output, int xOffset, int yOffset, unsigned char* minPaths)
{
  //int idX = blockIdx.x * POLAR_WIDTH + threadIdx.x;

  int tileIdx = blockIdx.x;
  int colIdx = threadIdx.x;
  
  __shared__ float currentErrors[POLAR_WIDTH];
  __shared__ float nextErrors[POLAR_WIDTH];
  __shared__ float randomStuff[POLAR_WIDTH];
  __shared__ char back_pointers[POLAR_WIDTH*POLAR_HEIGHT];

  Point p;
  p.x = colIdx;

  for (int i = 0; i < POLAR_HEIGHT; i++)
  {
    //random filler garbage
    currentErrors[colIdx] = colIdx;
    nextErrors[colIdx] = currentErrors[colIdx];
    randomStuff[colIdx] = nextErrors[colIdx];
    if (randomStuff[colIdx] != 123.123)
    {
      back_pointers[i*POLAR_WIDTH + colIdx] = colIdx;
    }
    if (back_pointers[i*POLAR_WIDTH + colIdx] == colIdx)
    {
      minPaths[tileIdx*POLAR_HEIGHT + i] = p.x;
    }
  }
  
  

}

void imagequilt_cuda(int texture_width, int texture_height, unsigned char* source, int* output)
{
  //initialize CUDA global memory
  unsigned char* source_cuda;
  int* output_cuda;
  //actually, I think it might be possible to store the previous 2 values in shared memory
  unsigned char* back_pointers;
  unsigned char* min_paths;

  int output_height = (HEIGHT_TILES + 1)*TILE_HEIGHT;
  int output_width = (WIDTH_TILES + 1)*TILE_WIDTH;
  int tile_size = TILE_HEIGHT*TILE_WIDTH;
  int num_tiles = HEIGHT_TILES*WIDTH_TILES;

  size_t source_size = sizeof(unsigned char)*texture_width*texture_height*3;
  size_t output_size = sizeof(int)*output_width*output_height;

  //size_t errors_size = sizeof(float)*output_width*output_height;
  //size_t float_polar_size = sizeof(float)*tile_size*num_tiles;
  size_t char_polar_size = sizeof(unsigned char)*tile_size*num_tiles;
  size_t paths_size = sizeof(unsigned char)*POLAR_HEIGHT*num_tiles;

  cudaMalloc((void**)&source_cuda, source_size);
  cudaMalloc((void**)&output_cuda, output_size);
  cudaMalloc((void**)&back_pointers, char_polar_size);
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
  dim3 gridDim(num_tiles);

  for (int iter = 0; iter < ITERATIONS; iter++)
  {
    //choose random grid alignment
    const int randX = std::rand() % (TILE_WIDTH/2);
    const int randY = std::rand() % (TILE_HEIGHT/2);

    kernelFindBoundaries<<<gridDim, blockDim>>>(output_cuda, randX, randY, min_paths);
    
  }

  cudaMemcpy(output, output_cuda, output_size, cudaMemcpyDeviceToHost);

  cudaFree(randStates);
  cudaFree(source_cuda);
  cudaFree(output_cuda);
}


