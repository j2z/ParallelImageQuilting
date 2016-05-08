
#ifndef UTIL_CU_H
#define UTIL_CU_H

#include "color_cu.hpp"

// 2D array index operations
inline __device__ int getRefIndx(int width, int row, int col)
{
  return row*width + col;
}

inline __device__ int refIndexRow(int width, int index)
{
  return index / width;
}

inline __device__ int refIndexCol(int width, int index)
{
  return index % width;
}

// getters/setters for the map
inline __device__ int imgGetRef(int* img, int width, int row, int col)
{
  return *(img + row*width + col);
}

inline __device__ void imgSetRef(int* img, int width, int row, int col, int val)
{
  *(img + row*width + col) = val;
}

// very granular getters/setters for the actual image
inline __device__ unsigned char imgGet(unsigned char* img, int width, int row, int col, int channel)
{
  return *(img + (row*width + col)*3 + channel);
}

inline __device__ void imgSet(unsigned char* img, int width, int row, int col, int channel, unsigned char val)
{
  *(img + (row*width + col)*3 + channel) = val;
}

// access by row,col
inline __device__ ColorCu imgGetColor(unsigned char* img, int width, int row, int col)
{
  unsigned char* addr = img + (row*width + col) * 3;
  return ColorCu(*addr, *(addr + 1), *(addr + 2));
}

// access by raw offset
inline __device__ ColorCu imgGetColor(unsigned char* img, int offset)
{
  unsigned char* addr = img + offset * 3;
  return ColorCu(*addr, *(addr + 1), *(addr + 2));
}


#endif
