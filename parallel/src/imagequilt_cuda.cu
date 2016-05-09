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



__global__ void kernelFindBoundaries(curandState* states, unsigned char* source, int sourceWidth, int sourceHeight, int* output, int xOffset, int yOffset, char* minPaths, short* samplesX, short* samplesY)
{
  int tileIdx = blockIdx.y * WIDTH_TILES + blockIdx.x;
  int idX = tileIdx * POLAR_WIDTH + threadIdx.x;

  int tileX = blockIdx.x;
  int tileY = blockIdx.y;
  int colIdx = threadIdx.x;
  
  __shared__ float array1[POLAR_WIDTH];
  __shared__ float array2[POLAR_WIDTH];
  __shared__ char backPointers[POLAR_HEIGHT][POLAR_WIDTH];

  __shared__ float scratch[POLAR_WIDTH];
  __shared__ float existingErrors[POLAR_WIDTH];

  __shared__ MappingData mapping;

  if (colIdx == 0)
  {
    samplesX[tileIdx] = curand(&states[idX]) % (sourceWidth - 2*MAX_RADIUS) + MAX_RADIUS;
    samplesY[tileIdx] = curand(&states[idX]) % (sourceHeight - 2*MAX_RADIUS) + MAX_RADIUS;

    mapping.src = source;
    mapping.srcWidth = sourceWidth;
    mapping.srcX = samplesX[tileIdx];
    mapping.srcY = samplesY[tileIdx];
    mapping.map = output;
    mapping.mapWidth = OUTPUT_WIDTH;
    mapping.mapX = tileX * TILE_WIDTH + TILE_WIDTH / 2 + xOffset;
    mapping.mapY = tileY * TILE_HEIGHT + TILE_HEIGHT / 2 + yOffset;
  }
  __syncthreads();
  
  existingErrors[colIdx] = -existing_error(mapping, colIdx, 0);
  
  __syncthreads();

  float* currentRow = array1;
  // populates currentRow with the negative sum of existing errors
  // (not including current)
  sharedMemExclusiveScan(colIdx, existingErrors, currentRow, scratch, POLAR_WIDTH);

  currentRow[colIdx] += horiz_error(mapping, colIdx, 0) + existingErrors[colIdx];
  
  float* previousRow = currentRow;
  currentRow = array2;


  for (int theta = 1; theta < POLAR_HEIGHT; theta++)
  {
    existingErrors[colIdx] = -existing_error(mapping, colIdx, theta);

    __syncthreads();
    
    // populates currentRow with the negative sum of existing errors
    // (not including current)
    sharedMemExclusiveScan(colIdx, existingErrors, currentRow, scratch, POLAR_WIDTH);

    char minTry = -1;
    float minVal = 0.f;
    for (char arg = colIdx - 1; arg <= colIdx + 1; arg++)
    {
      if (arg >= 0 && arg < POLAR_WIDTH)
      {
        if (minTry == -1 || previousRow[arg] < minVal)
        {
          minTry = arg;
          minVal = previousRow[arg];
        }
      }
    }
    currentRow[colIdx] += minVal + horiz_error(mapping, colIdx, theta) + existingErrors[colIdx];
    backPointers[theta][colIdx] = minTry;
    
    float* temp = previousRow;
    previousRow = currentRow;
    currentRow = temp;

  }

  // at this point, previousRow stores the seam costs

  int* scratch2 = (int*)scratch;

  int index = backPointers[POLAR_HEIGHT - 1][colIdx];
  for (int step = POLAR_HEIGHT - 2; step > 0; step--)
  {
    index = backPointers[step][index];
  }
  if (index == colIdx)
  {
    scratch2[colIdx] = colIdx;
  }
  else
  {
    scratch2[colIdx] = -1;
  }

  __syncthreads();

  // do a reduction
  for (int s = 1; s < POLAR_WIDTH; s*=2)
  {
    if (colIdx % (2 * s) == 0)
    {
      if (scratch2[colIdx] == -1)
      {
        if (scratch2[colIdx + s] != -1)
        {
          scratch2[colIdx] = scratch2[colIdx + s];
          previousRow[colIdx] = previousRow[colIdx + s];
        }
      }
      else
      {
        if (scratch2[colIdx+s] != -1 &&
              previousRow[colIdx+s] < previousRow[colIdx])
        {
          scratch2[colIdx] = scratch2[colIdx + s];
          previousRow[colIdx] = previousRow[colIdx + s];
        }
      }
    }
    __syncthreads();
  }

  if (scratch2[0] == colIdx && previousRow[0] < 1.0)
  {
    char index = colIdx;
    for (int step = POLAR_HEIGHT - 1; step >= 0; step--)
    {
      minPaths[tileIdx*POLAR_HEIGHT + step] = index;
      index = backPointers[step][index];
    }
  }

}

__global__ void kernelUpdateMap(int srcWidth, int* map, int xOffset, int yOffset, char* minPaths, short* samplesX, short* samplesY);


void imagequilt_cuda(int texture_width, int texture_height, unsigned char* source, int* output)
{
  //initialize CUDA global memory
  unsigned char* source_cuda;
  int* output_cuda;
  //actually, I think it might be possible to store the previous 2 values in shared memory
  char* min_paths;
  short* samplesX;
  short* samplesY;

  int output_height = (HEIGHT_TILES + 1)*TILE_HEIGHT;
  int output_width = (WIDTH_TILES + 1)*TILE_WIDTH;
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
  cudaMemcpy(output_cuda, output, output_size, cudaMemcpyHostToDevice);

  // seam carving: each tile gets 1 block of 32 threads
  dim3 seamCarveBlockDim(POLAR_WIDTH, 1);
  dim3 seamCarveGridDim(WIDTH_TILES, HEIGHT_TILES, 1);

  dim3 updateBlockDim(TILE_WIDTH / 2, TILE_HEIGHT / 2);
  dim3 updateGridDim(WIDTH_TILES, HEIGHT_TILES, 4);

  //first copy random pixels from source to output
  int seed = 15418;
  curandState *randStates;
  cudaMalloc((void**)&randStates, sizeof(curandState)*num_tiles);
  
  initRandom<<<seamCarveGridDim, seamCarveBlockDim>>>(seed, randStates);
  
  for (int iter = 0; iter < ITERATIONS; iter++)
  {
    cudaDeviceSynchronize();
    
    //choose random grid alignment
    const int offsetX = std::rand() % TILE_WIDTH;
    const int offsetY = std::rand() % TILE_HEIGHT;
    
    kernelFindBoundaries<<<seamCarveGridDim, seamCarveBlockDim>>>(randStates, source_cuda, texture_width, texture_height, output_cuda, offsetX, offsetY, min_paths, samplesX, samplesY);
   
    cudaDeviceSynchronize();
    // activate this when ready
    
    kernelUpdateMap<<<updateGridDim, updateBlockDim>>>
      (texture_width, output_cuda, offsetX, offsetY, min_paths, samplesX, samplesY);
  }

  cudaDeviceSynchronize();

  cudaMemcpy(output, output_cuda, output_size, cudaMemcpyDeviceToHost);

  cudaFree(randStates);
  cudaFree(source_cuda);
  cudaFree(output_cuda);
  cudaFree(min_paths);
  cudaFree(samplesX);
  cudaFree(samplesY);
}


