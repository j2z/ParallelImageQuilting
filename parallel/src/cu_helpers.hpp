/** Helper classes and functions for the serial implementation
 */

#ifndef CU_HELPERS_H
#define CU_HELPERS_H

#include "point.hpp"
#include "constants.hpp"
#include "polar_transform_cu.hpp"


class ErrorFunctionCu
{
private:
  unsigned char* src;
  int srcWidth;
  int srcX;
  int srcY;

  int* map;
  int mapWidth;
  int mapX;
  int mapY;

  PolarTransformationCu& transform;

public:
  __device__ ErrorFunctionCu(unsigned char*, int, int, int, int*, int, int, int, PolarTransformationCu&);

  __device__ float horiz_error(int rho, int theta);

  __device__ float existing_error(int rho, int theta);
};

#endif
