/** Helper classes and functions for the serial implementation
 */

#ifndef SERIAL_HELPERS_H
#define SERIAL_HELPERS_H

#include "point.hpp"
#include "util.hpp"
#include "mapping.hpp"
#include "constants.hpp"

#define ETA 10.f


// Transforms offsets to whereever the center is defined into polar
inline Point offsetToPolar(int x, int y)
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
inline Point polarToOffset(int rho, int theta)
{
  Point out;
  float rEff = rho / RADIUS_FACTOR;
  float thetaEff = theta / ANGLE_FACTOR;
  out.x = rEff * cos(thetaEff);
  out.y = rEff * sin(thetaEff);
  return out;
}


inline Point absoluteToPolar(int cx, int cy, int x, int y)
{
  return offsetToPolar(x-cx, y-cy);
}


inline Point polarToAbsolute(int cx, int cy, int rho, int theta)
{
  Point out = polarToOffset(rho, theta);
  out.x += cx;
  out.y += cy;
  return out;
}

static inline float normFactor(int r)
{
  return 2 * M_PI * r / POLAR_HEIGHT / RADIUS_FACTOR / RADIUS_FACTOR;
}

inline float horiz_error(MappingData& mapping, int rho, int theta)
{
  Point src1 = polarToAbsolute(mapping.srcX, mapping.srcY, rho, theta);
  Point src2 = polarToAbsolute(mapping.srcX, mapping.srcY, rho+1, theta);
  Point map1 = polarToAbsolute(mapping.mapX, mapping.mapY, rho, theta);
  Point map2 = polarToAbsolute(mapping.mapX, mapping.mapY, rho+1, theta);

  Color srcColor1 = imgGetColor(mapping.src, mapping.srcWidth, (int)round(src1.y), (int)round(src1.x));
  Color srcColor2 = imgGetColor(mapping.src, mapping.srcWidth, (int)round(src2.y), (int)round(src2.x));


  int dst1Offset = imgGetRef(mapping.map, mapping.mapWidth, (int)round(map1.y), (int)round(map1.x));
  int dst2Offset = imgGetRef(mapping.map, mapping.mapWidth, (int)round(map2.y), (int)round(map2.x));
  Color dstColor1 = imgGetColor(mapping.src, dst1Offset);
  Color dstColor2 = imgGetColor(mapping.src, dst2Offset);

  float error1 = (colorSqDiff(srcColor1, dstColor1) +
                    colorSqDiff(srcColor2, dstColor2))
                  / (colorSqDiff(srcColor1, srcColor2) +
                    colorSqDiff(dstColor1, dstColor2) + ETA);
  
  return normFactor(rho) * error1 * error1;

}

inline float existing_error(MappingData& mapping, int rho, int theta)
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


  Color dstColor1 = imgGetColor(mapping.src, dst1Offset);
  Color dstColor2 = imgGetColor(mapping.src, dst2Offset);

  Color srcColor1 = dstColor1;
  Color srcColor2 = imgGetColor(mapping.src, mapping.srcWidth, src1Y + yShift, src1X + xShift);

  // note that the square diff between srcColor1 and dstColor1 is 0 so we omit it
  float error1 = colorSqDiff(srcColor2, dstColor2)
                  / (colorSqDiff(srcColor1, srcColor2) +
                    colorSqDiff(dstColor1, dstColor2) + ETA);
  
  return normFactor(rho) * error1 * error1;
}


bool seam_carve(MappingData&, int[POLAR_HEIGHT]);

void update_map(MappingData&, int[POLAR_HEIGHT]);

// This function basically does everything
// (except initializing random colors at the beginning)
void imagequilt_serial(unsigned char*, int, int, int*, int, int);


float test_rejection_rate(unsigned char*, int, int, int*, int, int);

#endif
