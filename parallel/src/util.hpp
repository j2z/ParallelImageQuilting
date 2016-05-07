
#ifndef UTIL_H
#define UTIL_H

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
