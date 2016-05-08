
#include "image_data.hpp"
#include "util.hpp"


using namespace cimg_library;

void interleave_colors(unsigned char* dst, int height, int width, CImg<unsigned char>& src)
{
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      for (int channel = 0; channel < 3; channel++)
      {
        imgSet(dst, width, i, j, channel, src(j,i,channel));
      }
    }
  }
}

void generate_output(CImg<unsigned char>& out, int height, int width, unsigned char* src, int* map)
{
  //copy all the pixels to the actual output image
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      for (int channel = 0; channel < 3; channel++)
      {
        out(j,i,channel) = src[imgGetRef(map, width, i, j)*3 + channel];
      }
    }
  }
}
