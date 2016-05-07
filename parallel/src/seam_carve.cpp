
#include "serial_helpers.hpp"
#include "color.hpp"
#include "util.hpp"
#include <cmath>

#define ETA 2.f
#define MAX_JUMP 1

inline float normFactor(int r)
{
  return 2 * M_PI * r / POLAR_HEIGHT / RADIUS_FACTOR / RADIUS_FACTOR;
}

ErrorFunction::ErrorFunction(unsigned char* s, int sW, int cX, int cY, int* m, int mW, int mX, int mY, PolarTransformation t):
    src(s), srcWidth(sW), srcX(cX), srcY(cY), map(m), mapWidth(mW), mapX(mX), mapY(mY), transform(t)
{
}

float ErrorFunction::horiz_error(int rho, int theta)
{
  Point src1 = transform.polarToAbsolute(srcX, srcY, rho, theta);
  Point src2 = transform.polarToAbsolute(srcX, srcY, rho+1, theta);
  Point map1 = transform.polarToAbsolute(mapX, mapY, rho, theta);
  Point map2 = transform.polarToAbsolute(mapX, mapY, rho+1, theta);

  Color srcColor1 = imgGetColor(src, srcWidth, (int)round(src1.y), (int)round(src1.x));
  Color srcColor2 = imgGetColor(src, srcWidth, (int)round(src2.y), (int)round(src2.x));


  int dst1Offset = imgGetRef(map, mapWidth, (int)round(map1.y), (int)round(map1.x));
  int dst2Offset = imgGetRef(map, mapWidth, (int)round(map2.y), (int)round(map2.x));
  Color dstColor1 = imgGetColor(src, dst1Offset);
  Color dstColor2 = imgGetColor(src, dst2Offset);

  float error1 = (Color::sqDiff(srcColor1, dstColor1) +
                    Color::sqDiff(srcColor2, dstColor2))
                  / (Color::sqDiff(srcColor1, srcColor2) +
                    Color::sqDiff(dstColor1, dstColor2) + ETA);
  
  return normFactor(rho) * error1 * error1;

}

float ErrorFunction::existing_error(int rho, int theta)
{
  Point map1 = transform.polarToAbsolute(mapX, mapY, rho, theta);
  Point map2 = transform.polarToAbsolute(mapX, mapY, rho+1, theta);

  int map1Y = (int)round(map1.y);
  int map1X = (int)round(map1.x);

  int map2Y = (int)round(map2.y);
  int map2X = (int)round(map2.x);

  int xShift = map2X - map1X;
  int yShift = map2Y - map1Y;


  int dst1Offset = imgGetRef(map, mapWidth, map1Y, map1X);
  int dst2Offset = imgGetRef(map, mapWidth, map2Y, map2X);

  int src1X = refIndexCol(dst1Offset, srcWidth);
  int src1Y = refIndexRow(dst1Offset, srcWidth);

  Color dstColor1 = imgGetColor(src, dst1Offset);
  Color dstColor2 = imgGetColor(src, dst2Offset);


  Color srcColor1 = dstColor1;
  Color srcColor2 = imgGetColor(src, srcWidth, src1X + xShift, src1Y + yShift);

  // note that the square diff between srcColor1 and dstColor1 is 0 so we omit it
  float error1 = Color::sqDiff(srcColor2, dstColor2)
                  / (Color::sqDiff(srcColor1, srcColor2) +
                    Color::sqDiff(dstColor1, dstColor2) + ETA);
  
  return normFactor(rho) * error1 * error1;

}

// pass in an array for the polar coordinate error map
// with constants defined in constants.hpp

void seam_carve(ErrorFunction errFunc, int seam[POLAR_HEIGHT])
{
  // create 2 buffers for currentRow and previousRow
  float array1[POLAR_WIDTH];
  float array2[POLAR_WIDTH];

  float* currentRow = array1;

  // First row:
  // accumulate existing error on the left of the seam
  float existingError = 0.0;
  for (int rho = 0; rho < POLAR_WIDTH; rho++)
  {
    // total error = h + v - epsilon
    // we don't use v currently
    currentRow[rho] = errFunc.horiz_error(rho, 0) - existingError;

    existingError += errFunc.existing_error(rho, 0);
  }

  // backpointers array
  int backPointers[POLAR_HEIGHT][POLAR_WIDTH];


  float* previousRow = array1;
  currentRow = array2;

  // rest of the rows
  for (int theta = 1; theta < POLAR_HEIGHT; theta++)
  {
    existingError = 0.0;

    for (int rho = 0; rho < POLAR_WIDTH; rho++)
    {
      // find the min from the previous row
      int minIndex = -1;
      float minVal = 0.0;
      for (int arg = rho - MAX_JUMP; arg <= rho + MAX_JUMP; arg++)
      {
        if (arg >= 0 && arg < POLAR_WIDTH)
        {
          if (minIndex == -1 || previousRow[arg] < minVal)
          {
            minIndex = arg;
            minVal = previousRow[arg];
          }
        }
      }

      currentRow[rho] = minVal + errFunc.horiz_error(rho, theta) - existingError;
      backPointers[theta][rho] = minIndex;
      existingError += errFunc.existing_error(rho, theta);
    }

    // put current into previous, use the memory in previous as the new current
    float *temp = previousRow;
    previousRow = currentRow;
    currentRow = temp;

  }

  // at this point, the seam costs should be stored in previousRow

  // pick the best seam which also wraps around
  int minSeam = -1;
  for (int i = 0; i < POLAR_WIDTH; i++)
  {
    int index = backPointers[POLAR_HEIGHT-1][i];
    for (int step = POLAR_HEIGHT - 2; step > 0; step--)
    {
      index = backPointers[step][index];
    }
    // if we have wraparound
    if (index == i)
    {
      if (minSeam == -1 || previousRow[i] < previousRow[minSeam])
      {
        minSeam = i;
      }
    }
  }

  // copy min seam to seam[]
  int index = minSeam;
  for (int step = POLAR_HEIGHT - 1; step >= 0; step--)
  {
    seam[step] = index;
    index = backPointers[step][index];
  }
}


