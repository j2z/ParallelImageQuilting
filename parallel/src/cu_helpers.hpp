/** Helper classes and functions for the serial implementation
 */

#ifndef CU_HELPERS_H
#define CU_HELPERS_H

#include "point.hpp"
#include "constants.hpp"

// Transforms offsets to whereever the center is defined into polar
inline __device__ Point offsetToPolar(int x, int y)
{
  Point out;
  // rho
  out.x = sqrt(x*x + y*y) * RADIUS_FACTOR;
  // theta
  out.y = atan2(y,x);
  if (out.y < 0.f)
  {
    out.y += 2 * M_PI;
  }
  out.y *= ANGLE_FACTOR;
  return out;
}

// transforms polar to coordinates that are offsets to whereever the
// center is defined
inline __device__ Point polarToOffset(int rho, int theta)
{
  Point out;
  float rEff = rho / RADIUS_FACTOR;
  float thetaEff = theta / ANGLE_FACTOR;
  out.x = rEff * cos(thetaEff);
  out.y = rEff * sin(thetaEff);
  return out;
}


inline __device__ Point absoluteToPolar(int cx, int cy, int x, int y)
{
  return offsetToPolar(x-cx, y-cy);
}


inline __device__ Point polarToAbsolute(int cx, int cy, int rho, int theta)
{
  Point out = polarToOffset(rho, theta);
  out.x += cx;
  out.y += cy;
  return out;
}



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
