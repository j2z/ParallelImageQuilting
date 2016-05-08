/** For converting to polar representation as described in the paper
 *  
 *  Outputs are always floats (so that interpolation can be done)
 *  Inputs are always ints (pixel aligned coordinates)
 *
 *  Polar coordinate field assumes you are using the radiusFactor and angleFactor
 *  for increased resolution
 */

#include "point.hpp"

class PolarTransformationCu
{

  int radius;
  
  // increase resolution when converting to rectangle
  float radiusFactor;
  float angleFactor;


public:
  
  PolarTransformationCu(int,float,float);
  
  __device__ Point offsetToPolar(int, int);

  __device__ Point polarToOffset(int, int);

  __device__ Point absoluteToPolar(int, int, int, int);

  __device__ Point polarToAbsolute(int, int, int, int);

  __device__ int getRadius();

};

