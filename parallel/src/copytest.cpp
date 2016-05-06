
#include "CImg.h"
#include "constants.hpp"
#include "polar_transform.hpp"
#include <ctime>


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

  CImg<unsigned char> image("duck.jpg");
  CImg<unsigned char> output(500,500,1,3);

  PolarTransformation transform(MAX_RADIUS, RADIUS_FACTOR, ANGLE_FACTOR);

  unsigned char* imageRed = image.data();
  unsigned char* outputRed = output.data();
  
  unsigned char* imageGreen = imageRed + image.width() * image.height();
  unsigned char* outputGreen = outputRed + 500 * 500;
  
  unsigned char* imageBlue = imageGreen + image.width() * image.height();
  unsigned char* outputBlue = outputGreen + 500 * 500;




  copy_patch(imageRed, outputRed, image.width(), 500, 150, 150, 150, 150, transform, seam);
  copy_patch(imageGreen, outputGreen, image.width(), 500, 150, 150, 150, 150, transform, seam);
  copy_patch(imageBlue, outputBlue, image.width(), 500, 150, 150, 150, 150, transform, seam);
  
  output.save("try.jpg");

  return 0;
}
