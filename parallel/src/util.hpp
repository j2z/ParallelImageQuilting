
#ifndef UTIL_H
#define UTIL_H


inline unsigned char imgGet(unsigned char* img, int width, int row, int col)
{
  return *(img + row*width + col);
}

inline void imgSet(unsigned char* img, int width, int row, int col, unsigned char val)
{
  *(img + row*width + col) = val;
}

unsigned char sample(unsigned char*, int, float, float);



#endif
