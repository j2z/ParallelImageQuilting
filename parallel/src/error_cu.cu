
#include "cu_helpers.hpp"
#include "color_cu.hpp"
#include "util_cu.hpp"

#define ETA 10.f
#define MAX_JUMP 1
#define CENTER_ERROR 100000.f
#define IMPROVE_THRESH 30.f

inline __device__ float normFactor(int r)
{
  return 2 * M_PI * r / POLAR_HEIGHT / RADIUS_FACTOR / RADIUS_FACTOR;
}

__device__ ErrorFunctionCu::ErrorFunctionCu(unsigned char* s, int sW, int cX, int cY, int* m, int mW, int mX, int mY, PolarTransformationCu& t):
    src(s), srcWidth(sW), srcX(cX), srcY(cY), map(m), mapWidth(mW), mapX(mX), mapY(mY), transform(t)
{
}

inline __device__ float ErrorFunctionCu::horiz_error(int rho, int theta)
{
  if (rho < 3)
  {
    return CENTER_ERROR;
  }
  Point src1 = transform.polarToAbsolute(srcX, srcY, rho, theta);
  Point src2 = transform.polarToAbsolute(srcX, srcY, rho+1, theta);
  Point map1 = transform.polarToAbsolute(mapX, mapY, rho, theta);
  Point map2 = transform.polarToAbsolute(mapX, mapY, rho+1, theta);

  ColorCu srcColor1 = imgGetColor(src, srcWidth, (int)round(src1.y), (int)round(src1.x));
  ColorCu srcColor2 = imgGetColor(src, srcWidth, (int)round(src2.y), (int)round(src2.x));


  int dst1Offset = imgGetRef(map, mapWidth, (int)round(map1.y), (int)round(map1.x));
  int dst2Offset = imgGetRef(map, mapWidth, (int)round(map2.y), (int)round(map2.x));
  ColorCu dstColor1 = imgGetColor(src, dst1Offset);
  ColorCu dstColor2 = imgGetColor(src, dst2Offset);

  float error1 = (ColorCu::sqDiff(srcColor1, dstColor1) +
                    ColorCu::sqDiff(srcColor2, dstColor2))
                  / (ColorCu::sqDiff(srcColor1, srcColor2) +
                    ColorCu::sqDiff(dstColor1, dstColor2) + ETA);
  
  return normFactor(rho) * error1 * error1;

}

inline __device__ float ErrorFunctionCu::existing_error(int rho, int theta)
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

  int src1X = refIndexCol(srcWidth, dst1Offset);
  int src1Y = refIndexRow(srcWidth, dst1Offset);


  ColorCu dstColor1 = imgGetColor(src, dst1Offset);
  ColorCu dstColor2 = imgGetColor(src, dst2Offset);

  ColorCu srcColor1 = dstColor1;
  ColorCu srcColor2 = imgGetColor(src, srcWidth, src1Y + yShift, src1X + xShift);

  // note that the square diff between srcColor1 and dstColor1 is 0 so we omit it
  float error1 = ColorCu::sqDiff(srcColor2, dstColor2)
                  / (ColorCu::sqDiff(srcColor1, srcColor2) +
                    ColorCu::sqDiff(dstColor1, dstColor2) + ETA);
  
  return normFactor(rho) * error1 * error1;

}
