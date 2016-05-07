#include "CImg.h"
#include "util.hpp"
#include "constants.hpp"
#include "CycleTimer.h"

#include <iostream>
#include <ctime>
#include <cstdio>
#include <string.h>


using namespace cimg_library;

void imagequilt_cuda(int texture_width, int texture_height, unsigned char* source, int* output);

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
  bool run_cuda = false;
  bool run_both = true;
  if (argc > 2)
  {
    if(strcmp(argv[2],"c") == 0 || strcmp(argv[2],"C") == 0)
    {
      run_both = false;
      run_cuda = true;
    }
    else if (strcmp(argv[3],"s") == 0 || strcmp(argv[2],"S") == 0)
    {
      run_both = false;
      run_cuda = false;
    }
  }
  
  const int texture_height = texture_image.height();
  const int texture_width = texture_image.width();

  const int output_height = (HEIGHT_TILES + 1) * (TILE_HEIGHT);
  const int output_width = (WIDTH_TILES + 1) * TILE_WIDTH;

  CImg<unsigned char> output(output_width, output_height, 1, 3);

  unsigned char* source_pixels = (unsigned char *)malloc(sizeof(unsigned char)*texture_height*texture_width*3);
  int* out_pixels = (int *)malloc(sizeof(int)*output_height*output_width);

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

  double seqTime = 1;
  double cudaTime = 1;

  if (run_both || !run_cuda)
  {
    printf("Running Serial\n");
    double startTime = CycleTimer::currentSeconds();
    //generate white noise image based on random pixels from the original image
    for (int i = 0; i < output_height; i++)
    {
      for (int j = 0; j < output_width; j++)
      {
        const int randX = std::rand() % (texture_width);
        const int randY = std::rand() % (texture_height);
        
        for (int channel = 0; channel < 3; channel++)
        {
          imgSetRef(out_pixels, output_width, i, j, getRefIndx(texture_width,randY, randX));
        }
      }
    }

    double endTime = CycleTimer::currentSeconds();
    seqTime = endTime - startTime;
    printf("Sequential Time: %.3f ms\n", 1000.f* seqTime);

    //copy all the pixels to the actual output image
    for (int i = 0; i < output_height; i++)
    {
      for (int j = 0; j < output_width; j++)
      {
        for (int channel = 0; channel < 3; channel++)
        {
          output(j,i,channel) = source_pixels[imgGetRef(out_pixels, output_width, i, j)*3 + channel];
        }
      }
    }
    output.save("serial_quilt.jpg");
  }

  if (run_both || run_cuda)
  {
    printf("Running in CUDA\n");
    double startTime = CycleTimer::currentSeconds();
    imagequilt_cuda(texture_width, texture_height, source_pixels, out_pixels);
    double endTime = CycleTimer::currentSeconds();
    cudaTime = endTime - startTime;
    printf("CUDA Time: %.3f ms\n", 1000.f* cudaTime);

    //copy all the pixels to the actual output image
    for (int i = 0; i < output_height; i++)
    {
      for (int j = 0; j < output_width; j++)
      {
        for (int channel = 0; channel < 3; channel++)
        {
          output(j,i,channel) = source_pixels[imgGetRef(out_pixels, output_width, i, j)*3 + channel];
        }
      }
    }

    output.save("cuda_quilt.jpg");
  }

  if (run_both)
  {
    double speedup = seqTime/cudaTime;
    printf("Speedup: %.3f\n", speedup);
  

  }
}
