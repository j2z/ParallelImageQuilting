
#ifndef UTIL_H
#define UTIL_H

inline int getRefIndx(int width, int row, int col)
{
  return row*width + col;
}

inline int imgGetRef(int* img, int width, int row, int col)
{
  return *(img + row*width + col);
}

inline void imgSetRef(int* img, int width, int row, int col, int val)
{
  *(img + row*width + col) = val;
}

inline unsigned char imgGet(unsigned char* img, int width, int row, int col, int channel)
{
  return *(img + (row*width + col)*3 + channel);
}

inline void imgSet(unsigned char* img, int width, int row, int col, int channel, unsigned char val)
{
  *(img + (row*width + col)*3 + channel) = val;
}

unsigned char sample(unsigned char*, int, float, float);



#endif
