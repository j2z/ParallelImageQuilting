
#include "CImg.h"
#include "constants.hpp"
#include "image_data.hpp"
#include "serial_helpers.hpp"
#include <ctime>
#include <iostream>

using namespace cimg_library;

void copy_patch(unsigned char* src, unsigned char* dst, int srcWidth, int dstWidth,
                int srcX, int srcY, int dstX, int dstY,
                PolarTransformation transform, int seam[]);


int main(int argc, char* argv[])
{
  srand(time(NULL));
  int seam[POLAR_HEIGHT];
  
  int seampos = POLAR_WIDTH / 2;
  
  for (int i = 0; i < POLAR_HEIGHT; i++)
  {
    switch (rand() % 3)
    {
      case 0:
        if (seampos == 0) seampos++;
        else              seampos--;
        break;
      case 1:
        if (seampos == POLAR_WIDTH-1) seampos--;
        else                          seampos++;
        break;
      default:
        break;
    }
    seam[i] = seampos;
  }

  CImg<unsigned char> image("bricks.jpg");
  CImg<unsigned char> output(500,500,1,3);
  CImg<float> err(POLAR_WIDTH,POLAR_HEIGHT,1,1);

  int imHeight = image.height();
  int imWidth = image.width();

  unsigned char* source_pixels = (unsigned char*)malloc(sizeof(unsigned char) * imHeight * imWidth * 3);

  interleave_colors(source_pixels, imHeight, imWidth, image);

  int* map = (int*)malloc(sizeof(int) * 500 * 500);

  for (int i = 0; i < 500 * 500; i++)
  {
    map[i] = 204*100 + 100;
  }

  PolarTransformation transform(MAX_RADIUS, RADIUS_FACTOR, ANGLE_FACTOR);

  ErrorFunction errFunc(source_pixels, imWidth, 100, 100, map, 500, 250, 250, transform);

  seam_carve(errFunc, seam);

  std::cout << "done carving" << std::endl;

  update_map(source_pixels, imWidth, 100, 100, map, 500, 250, 250, transform, seam);

  
  for (int j = 0; j < POLAR_HEIGHT; j++)
  {
    for (int i = 0; i < POLAR_WIDTH; i++)
    {
      err(i,j) = errFunc.horiz_error(i,j);
    }
  }
  
  std::cout << "map generated" << std::endl;


  generate_output(output, 500, 500, source_pixels, map);

  err.save("err.jpg");
  output.save("try.jpg");

  return 0;
}
