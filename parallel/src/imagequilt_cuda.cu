#include <cstdio>
#include <ctime>
#include "util.hpp"
#include "constants.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define NUM_ITERATIONS 20

void imagequilt_cuda(int texture_width, int texture_height, unsigned char* source, unsigned char* output)
{
  //initialize CUDA global memory
  unsigned char* source_cuda;
  unsigned char* output_cuda;
  float* errors_cuda; //cost of the old seams
  float* energy_cuda; //differences between the source patch and output (maybe not necessary and just use polar)
  float* energy_polar;
  float* min_seam; //smallest acculumation of energy from that pixel
  //actually, I think it might be possible to store the previous 2 values in shared memory
  unsigned char* back_pointers;

  int output_height = HEIGHT_TILES*TILE_HEIGHT;
  int output_width = WIDTH_TILES*TILE_WIDTH;
  int tile_size = TILE_HEIGHT*TILE_WIDTH;
  int num_tiles = HEIGHT_TILES*WIDTH_TILES;

  size_t source_size = sizeof(unsigned char)*texture_width*texture_height*3;
  size_t output_size = sizeof(unsigned char)*output_width*output_height*3;
  size_t errors_size = sizeof(float)*output_width*output_height;
  size_t float_polar_size = sizeof(float)*tile_size*num_tiles;
  size_t char_polar_size = sizeof(unsigned char)*tile_size*num_tiles;

  cudaMalloc((void**)&source_cuda, source_size);
  cudaMalloc((void**)&output_cuda, output_size);
  cudaMalloc((void**)&errors_cuda, errors_size);
  cudaMalloc((void**)&energy_cuda, errors_size);
  cudaMalloc((void**)&energy_polar, float_polar_size);
  cudaMalloc((void**)&min_seam, float_polar_size);
  cudaMalloc((void**)&back_pointers, char_polar_size);

  cudaMemcpy(source_cuda, source, source_size, cudaMemcpyHostToDevice);
  cudaMemcpy(output_cuda, output, output_size, cudaMemcpyHostToDevice);

  for (int iter = 0; iter < NUM_ITERATIONS; iter++)
  {
    //choose random grid alignment
    const int randX = std::rand() % (TILE_WIDTH);
    const int randY = std::rand() % (TILE_HEIGHT);

    //choose random patch from source and calculates the energy of each patch

    //convert energies in polar and perform DP

    //check whether there was an improvement???

  }



  cudaFree(source_cuda);
  cudaFree(output_cuda);
  cudaFree(errors_cuda);
}
