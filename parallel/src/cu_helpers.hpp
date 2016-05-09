/** Helper classes and functions for the serial implementation
 */

#ifndef CU_HELPERS_H
#define CU_HELPERS_H

#include "point.hpp"
#include "util_cu.hpp"
#include "mapping.hpp"
#include "constants.hpp"

#include <cstdio>

#define ETA 10.f

// Transforms offsets to whereever the center is defined into polar
inline __device__ Point offsetToPolar(int x, int y)
{
  Point out;
  // rho
  out.x = sqrtf(x*x + y*y) * RADIUS_FACTOR;
  // theta
  out.y = atan2f(y,x);
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
  out.x = rEff * cosf(thetaEff);
  out.y = rEff * sinf(thetaEff);
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

static inline __device__ float normFactor(int r)
{
  return 2 * M_PI * r / POLAR_HEIGHT / RADIUS_FACTOR / RADIUS_FACTOR;
}

inline __device__ float horiz_error(MappingData& mapping, int rho, int theta)
{
  Point src1 = polarToAbsolute(mapping.srcX, mapping.srcY, rho, theta);
  Point src2 = polarToAbsolute(mapping.srcX, mapping.srcY, rho+1, theta);
  Point map1 = polarToAbsolute(mapping.mapX, mapping.mapY, rho, theta);
  Point map2 = polarToAbsolute(mapping.mapX, mapping.mapY, rho+1, theta);

  ColorCu srcColor1 = imgGetColor(mapping.src, mapping.srcWidth, (int)round(src1.y), (int)round(src1.x));
  ColorCu srcColor2 = imgGetColor(mapping.src, mapping.srcWidth, (int)round(src2.y), (int)round(src2.x));


  int dst1Offset = imgGetRef(mapping.map, mapping.mapWidth, (int)round(map1.y), (int)round(map1.x));
  int dst2Offset = imgGetRef(mapping.map, mapping.mapWidth, (int)round(map2.y), (int)round(map2.x));
  ColorCu dstColor1 = imgGetColor(mapping.src, dst1Offset);
  ColorCu dstColor2 = imgGetColor(mapping.src, dst2Offset);

  float error1 = (colorSqDiff(srcColor1, dstColor1) +
                    colorSqDiff(srcColor2, dstColor2))
                  / (colorSqDiff(srcColor1, srcColor2) +
                    colorSqDiff(dstColor1, dstColor2) + ETA);
  
  return normFactor(rho) * error1 * error1;

}

inline __device__ float existing_error(MappingData& mapping, int rho, int theta)
{
  Point map1 = polarToAbsolute(mapping.mapX, mapping.mapY, rho, theta);
  Point map2 = polarToAbsolute(mapping.mapX, mapping.mapY, rho+1, theta);

  int map1Y = (int)round(map1.y);
  int map1X = (int)round(map1.x);

  int map2Y = (int)round(map2.y);
  int map2X = (int)round(map2.x);


  int xShift = map2X - map1X;
  int yShift = map2Y - map1Y;


  int dst1Offset = imgGetRef(mapping.map, mapping.mapWidth, map1Y, map1X);
  int dst2Offset = imgGetRef(mapping.map, mapping.mapWidth, map2Y, map2X);

  int src1X = refIndexCol(mapping.srcWidth, dst1Offset);
  int src1Y = refIndexRow(mapping.srcWidth, dst1Offset);



  ColorCu dstColor1 = imgGetColor(mapping.src, dst1Offset);
  ColorCu dstColor2 = imgGetColor(mapping.src, dst2Offset);

  ColorCu srcColor1 = dstColor1;
  ColorCu srcColor2 = imgGetColor(mapping.src, mapping.srcWidth, src1Y + yShift, src1X + xShift);

  // note that the square diff between srcColor1 and dstColor1 is 0 so we omit it
  float error1 = colorSqDiff(srcColor2, dstColor2)
                  / (colorSqDiff(srcColor1, srcColor2) +
                    colorSqDiff(dstColor1, dstColor2) + ETA);
  
  return normFactor(rho) * error1 * error1;
}


#endif
