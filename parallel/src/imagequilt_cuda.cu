#include <cstdio>
#include <ctime>
#include "cu_helpers.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#define NUM_ITERATIONS 20
#define BLOCK_SIZE 64

#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)
#define SCAN_BLOCK_DIM POLAR_WIDTH

//Almost the same as naive scan1Inclusive, but doesn't need __syncthreads()
//assuming size <= WARP_SIZE
inline __device__ uint
warpScanInclusive(int threadIndex, float idata, volatile float *s_Data, uint size){
  uint pos = 2 * threadIndex - (threadIndex & (size - 1));
  s_Data[pos] = 0;
  pos += size;
  s_Data[pos] = idata;

  for(uint offset = 1; offset < size; offset <<= 1)
    s_Data[pos] += s_Data[pos - offset];

  return s_Data[pos];
}

inline __device__ float warpScanExclusive(int threadIndex, float idata, volatile float *sScratch, uint size){
  return warpScanInclusive(threadIndex, idata, sScratch, size) - idata;
}

__inline__ __device__ void
sharedMemExclusiveScan(int threadIndex, float* sInput, float* sOutput, volatile float* sScratch, uint size)
{
  if (size > WARP_SIZE) {

    float idata = sInput[threadIndex];

    //Bottom-level inclusive warp scan
    float warpResult = warpScanInclusive(threadIndex, idata, sScratch, WARP_SIZE);

    // Save top elements of each warp for exclusive warp scan sync
    // to wait for warp scans to complete (because s_Data is being
    // overwritten)
    __syncthreads();

    if ( (threadIndex & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
      sScratch[threadIndex >> LOG2_WARP_SIZE] = warpResult;

    // wait for warp scans to complete
    __syncthreads();

    if ( threadIndex < (SCAN_BLOCK_DIM / WARP_SIZE)) {
      // grab top warp elements
      float val = sScratch[threadIndex];
      // calculate exclusive scan and write back to shared memory
      sScratch[threadIndex] = warpScanExclusive(threadIndex, val, sScratch, size >> LOG2_WARP_SIZE);
    }

    //return updated warp scans with exclusive scan results
    __syncthreads();

    sOutput[threadIndex] = warpResult + sScratch[threadIndex >> LOG2_WARP_SIZE] - idata;

    } else if (threadIndex < WARP_SIZE) {
      float idata = sInput[threadIndex];
      sOutput[threadIndex] = warpScanExclusive(threadIndex, idata, sScratch, size);
    }
}

__global__ void initRandom(unsigned int seed, curandState* states)
{
  int idX = (blockIdx.y*BLOCK_SIZE + blockIdx.x) * BLOCK_SIZE + threadIdx.x;
  curand_init(seed, idX,0,&states[idX]);
}

__global__ void kernelRandomOutput(curandState* states, int* output, int output_width, int source_size)
{
  int idX = (blockIdx.y*BLOCK_SIZE + blockIdx.x) * BLOCK_SIZE + threadIdx.x;
  output[idX] = curand(&states[idX]) % source_size;
}

__global__ void kernelFindBoundaries(curandState* states, unsigned char* source, int sourceWidth, int sourceHeight, int* output, int xOffset, int yOffset, unsigned char* minPaths, short* samplesX, short* samplesY)
{
  int tileIdx = blockIdx.y * WIDTH_TILES + blockIdx.x;
  int idX = tileIdx * POLAR_WIDTH + threadIdx.x;

  int tileX = blockIdx.x;
  int tileY = blockIdx.y;
  int colIdx = threadIdx.x;
  
  __shared__ float array1[POLAR_WIDTH];
  __shared__ float array2[POLAR_WIDTH];
  __shared__ char back_pointers[POLAR_HEIGHT][POLAR_WIDTH];

  __shared__ float scratch[POLAR_WIDTH];
  __shared__ float scanOut[POLAR_WIDTH];

  if (colIdx == 0)
  {
    samplesX[tileIdx] = curand(&states[idX]) % (sourceWidth - 2*MAX_RADIUS) + MAX_RADIUS;
    samplesY[tileIdx] = curand(&states[idX]) % (sourceHeight - 2*MAX_RADIUS) + MAX_RADIUS;
  }
  __syncthreads();

  int srcX = samplesX[tileIdx];
  int srcY = samplesY[tileIdx];

  int mapX = tileX * TILE_WIDTH + TILE_WIDTH / 2 + xOffset;
  int mapY = tileY * TILE_HEIGHT + TILE_HEIGHT / 2 + yOffset;

  float* current_array = array1;
  current_array[colIdx] = colIdx;
  __syncthreads();
  sharedMemExclusiveScan(colIdx, current_array, scanOut, scratch, POLAR_WIDTH);
  

  //ErrorFunctionCu errFunc(source, sourceWidth, srcX, srcY, output, OUTPUT_WIDTH, mapX, mapY, PolarTransformation(MAX_RADIUS, RADIUS_FACTOR, ANGLE_FACTOR));

  for (int i = 0; i < POLAR_HEIGHT; i++)
  {
  }
  
  ColorCu color(3.f,4.f,5.f);
  array1[threadIdx.x] = color.red;
  array2[threadIdx.x] = colorSqDiff(color, color);

}

void imagequilt_cuda(int texture_width, int texture_height, unsigned char* source, int* output)
{
  //initialize CUDA global memory
  unsigned char* source_cuda;
  int* output_cuda;
  //actually, I think it might be possible to store the previous 2 values in shared memory
  unsigned char* min_paths;
  short* samplesX;
  short* samplesY;

  int output_height = (HEIGHT_TILES + 1)*TILE_HEIGHT;
  int output_width = (WIDTH_TILES + 1)*TILE_WIDTH;
  int tile_size = TILE_HEIGHT*TILE_WIDTH;
  int num_tiles = HEIGHT_TILES*WIDTH_TILES;

  size_t source_size = sizeof(unsigned char)*texture_width*texture_height*3;
  size_t output_size = sizeof(int)*output_width*output_height;
  size_t paths_size = sizeof(unsigned char)*POLAR_HEIGHT*num_tiles;
  size_t samples_size = sizeof(short)*num_tiles;

  cudaMalloc((void**)&source_cuda, source_size);
  cudaMalloc((void**)&output_cuda, output_size);
  cudaMalloc((void**)&min_paths, paths_size);
  cudaMalloc((void**)&samplesX, samples_size);
  cudaMalloc((void**)&samplesY, samples_size);

  cudaMemcpy(source_cuda, source, source_size, cudaMemcpyHostToDevice);

  //first copy random pixels from source to output
  int seed = 15418;
  curandState *randStates;
  cudaMalloc((void**)&randStates, sizeof(curandState)*output_width*output_height);
  dim3 tempblock(32,32);
  int widthGrid = output_width*output_height/BLOCK_SIZE/BLOCK_SIZE;
  dim3 randGrid(BLOCK_SIZE, widthGrid);
  printf("Lolz %d\n",widthGrid);
  
  initRandom<<<randGrid, BLOCK_SIZE>>>(seed, randStates);
  cudaDeviceSynchronize();
  kernelRandomOutput<<<randGrid, BLOCK_SIZE>>>(randStates, 
                                                output_cuda,
                                                output_width,
                                                texture_width*texture_height);
  cudaDeviceSynchronize();

  dim3 blockDim(POLAR_WIDTH, 1);
  dim3 gridDim(WIDTH_TILES, HEIGHT_TILES, 1);

  /*for (int iter = 0; iter < ITERATIONS; iter++)
  {
    //choose random grid alignment
    const int offsetX = std::rand() % TILE_WIDTH;
    const int offsetY = std::rand() % TILE_HEIGHT;

    kernelFindBoundaries<<<gridDim, blockDim>>>(randStates, source_cuda, texture_width, texture_height, output_cuda, offsetX, offsetY, min_paths, samplesX, samplesY);
    
  }*/

  cudaMemcpy(output, output_cuda, output_size, cudaMemcpyDeviceToHost);

  cudaFree(randStates);
  cudaFree(source_cuda);
  cudaFree(output_cuda);
}


