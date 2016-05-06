/** For converting to polar representation as described in the paper
 *  
 *  Outputs are always floats (so that interpolation can be done)
 *  Inputs are always ints (pixel aligned coordinates)
 *
 *  Polar coordinate field assumes you are using the radiusFactor and angleFactor
 *  for increased resolution
 */

#include "point.hpp"


class PolarTransformation
{

  int radius;
  
  // increase resolution when converting to rectangle
  float radiusFactor;
  float angleFactor;


public:
  
  PolarTransformation(int,float,float);
  
  Point offsetToPolar(int, int);

  Point polarToOffset(int, int);

  Point absoluteToPolar(int, int, int, int);

  Point polarToAbsolute(int, int, int, int);

  int getRadius();

};

