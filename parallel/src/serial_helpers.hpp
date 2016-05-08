/** Helper classes and functions for the serial implementation
 */

#ifndef SERIAL_HELPERS_H
#define SERIAL_HELPERS_H

#include "point.hpp"
#include "constants.hpp"

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


class ErrorFunction
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

  PolarTransformation& transform;

public:
  ErrorFunction(unsigned char*, int, int, int, int*, int, int, int, PolarTransformation&);

  float horiz_error(int rho, int theta);

  float existing_error(int rho, int theta);
};


bool seam_carve(ErrorFunction&, int[POLAR_HEIGHT]);

void update_map(unsigned char*, int, int, int, int*, int, int, int, PolarTransformation&, int[POLAR_HEIGHT]);

// This function basically does everything
// (except initializing random colors at the beginning)
void imagequilt_serial(unsigned char*, int, int, int*, int, int);


#endif
