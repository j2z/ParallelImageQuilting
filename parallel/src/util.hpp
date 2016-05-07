
#ifndef UTIL_H
#define UTIL_H

#include "color.hpp"


inline int getRefIndx(int width, int row, int col)
{
  return row*width + col;
}

// getters/setters for the map
inline int imgGetRef(int* img, int width, int row, int col)
{
  return *(img + row*width + col);
}

inline void imgSetRef(int* img, int width, int row, int col, int val)
{
  *(img + row*width + col) = val;
}

// very granular getters/setters
inline unsigned char imgGet(unsigned char* img, int width, int row, int col, int channel)
{
  return *(img + (row*width + col)*3 + channel);
}

inline void imgSet(unsigned char* img, int width, int row, int col, int channel, unsigned char val)
{
  *(img + (row*width + col)*3 + channel) = val;
}

// access by row,col
inline Color imgGetColor(unsigned char* img, int width, int row, int col)
{
  unsigned char* addr = img + (row*width + col) * 3;
  return Color(*addr, *(addr + 1), *(addr + 2));
}

// access by raw offset
inline Color imgGetColor(unsigned char* img, int offset)
{
  unsigned char* addr = img + offset * 3;
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
