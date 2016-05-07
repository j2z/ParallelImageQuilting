
#ifndef UTIL_H
#define UTIL_H

#include "color.hpp"

#define TILE_HEIGHT 80
#define TILE_WIDTH 80
#define CIRCLE_RADIUS 30
#define WIDTH_TILES 12
#define HEIGHT_TILES 8
#define NUM_ITERATIONS 20


inline unsigned char imgGet(unsigned char* img, int width, int row, int col, int channel)
{
  return *(img + (row*width + col)*3 + channel);
}

inline void imgSet(unsigned char* img, int width, int row, int col, int channel, unsigned char val)
{
  *(img + (row*width + col)*3 + channel) = val;
}

inline Color imgGetColor(unsigned char* img, int width, int row, int col)
{
  unsigned char* addr = img + (row*width + col) * 3;
  return Color(*addr, *(addr + 1), *(addr + 2));
}

/*
inline void imgSetColor(unsigned char* img, int width, int row, int col, Color val)
{
  unsigned char* addr = img + (row*width + col) * 3;
  *addr = val.red;
  *(addr + 1) = val.green;
  *(addr + 2) = val.blue;
}
*/

Color sample(unsigned char*, int, float, float);


#endif
