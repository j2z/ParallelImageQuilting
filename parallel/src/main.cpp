#include "CImg.h"
#include "util.hpp"

#include <iostream>
#include <ctime>


using namespace cimg_library;

void cuda_image_quilt(unsigned char* source, unsigned char* output);

//void vertical_stitch(CImg<int> im1, CImg<int> im2, int x1, int y1, int x2, int y2, int w, int h, int seam[]);
//void horizontal_stitch(CImg<int> im1, CImg<int> im2, int x1, int y1, int x2, int y2, int w, int h, int seam[]);

static inline void disp_help()
{
  std::cout << "Arg 1: filename of texture image (JPEG)" << std::endl;
  std::cout << "Arg 2: filename for output file (JPEG)" << std::endl;
}
// 2 images of same size 
//static inline float ssd(CImg<int> im1, CImg<int> im2)
//{
//  return (im1 - im2).get_sqr().sum();
//}
//


int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    disp_help();
    return 0;
  }

  // set up PRNG
  std::srand(time(NULL));
  
  CImg<unsigned char> texture_image(argv[1]);

  const int texture_height = texture_image.height();
  const int texture_width = texture_image.width();

  const int output_height = HEIGHT_TILES * TILE_HEIGHT;
  const int output_width = WIDTH_TILES * TILE_WIDTH;

  CImg<unsigned char> output(output_width, output_height, 1, 3);

  unsigned char* source_pixels = (unsigned char *)malloc(sizeof(unsigned char)*texture_height*texture_width*3);
  unsigned char* out_pixels = (unsigned char *)malloc(sizeof(char)*HEIGHT_TILES*TILE_HEIGHT*WIDTH_TILES*TILE_WIDTH*3);
  unsigned char* errors = (unsigned char *)malloc(sizeof(char)*HEIGHT_TILES*TILE_HEIGHT*WIDTH_TILES*TILE_WIDTH*3);

  //copy image right now it just copies every pixel until CImg's website comes back up
  for (int i = 0; i < texture_height; i++)
  {
    for (int j = 0; j < texture_width; j++)
    {
      for (int channel = 0; channel < 3; channel++)
      {
        imgSet(source_pixels, texture_width, i, j, channel, texture_image(j,i,channel));
      }
    }
  }

  //generate white noise image based on random pixels from the original image
  for (int i = 0; i < output_height; i++)
  {
    for (int j = 0; j < output_width; j++)
    {
      const int randX = std::rand() % (texture_width);
      const int randY = std::rand() % (texture_height);
      for (int channel = 0; channel < 3; channel++)
      {
        imgSet(out_pixels, output_width, i, j, channel, texture_image(randX,randY,channel));
      }
    }
  }

  cuda_image_quilt(source_pixels, out_pixels);
  
  //copy all the pixels to the actual output image
  for (int i = 0; i < output_height; i++)
  {
    for (int j = 0; j < output_width; j++)
    {
      for (int channel = 0; channel < 3; channel++)
      {
        output(j,i,channel) = imgGet(out_pixels, output_width, i, j, channel);
      }
    }
  }

  output.save("test.jpg");

  const int patchX = std::rand() % (texture_width - TILE_WIDTH);
  const int patchY = std::rand() % (texture_height - TILE_HEIGHT);

}
