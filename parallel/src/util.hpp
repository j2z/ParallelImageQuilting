
#ifndef UTIL_H
#define UTIL_H

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

unsigned char sample(unsigned char*, int, float, float);



#endif
