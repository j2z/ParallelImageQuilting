#include "CImg.h"
#include "serial_helpers.hpp"
#include "image_data.hpp"
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
  std::cout << "Arg 2: c for cuda, s for sequential, b for both" << std::endl;
  std::cout << "Arg 3: num trials (optional)" << std::endl;
}
// 2 images of same size 
//static inline float ssd(CImg<int> im1, CImg<int> im2)
//{
//  return (im1 - im2).get_sqr().sum();
//}
//


int main(int argc, char* argv[])
{
  if (argc < 3)
  {
    disp_help();
    return 0;
  }

  // set up PRNG
  std::srand(time(NULL));
  
  CImg<unsigned char> texture_image(argv[1]);
  bool run_cuda = false;
  bool run_both = true;
  bool rejection_rate = false;
  if(strcmp(argv[2],"c") == 0 || strcmp(argv[2],"C") == 0)
  {
    run_both = false;
    run_cuda = true;
  }
  else if (strcmp(argv[2],"s") == 0 || strcmp(argv[2],"S") == 0)
  {
    run_both = false;
    run_cuda = false;
  }
  else if (strcmp(argv[2],"r") == 0)
  {
    rejection_rate = true;
  }
  else if (strcmp(argv[2], "b") == 0 || strcmp(argv[2], "B") == 0)
  {
    
  }
  else
  {
    disp_help();
    return 0;
  }

  int numTrials = DEFAULT_TRIALS;

  if (argc > 3)
  {
    numTrials = std::atoi(argv[3]);
  }
  
  const int texture_height = texture_image.height();
  const int texture_width = texture_image.width();

  CImg<unsigned char> output(OUTPUT_WIDTH, OUTPUT_HEIGHT, 1, 3);

  unsigned char* source_pixels = (unsigned char *)malloc(sizeof(unsigned char)*texture_height*texture_width*3);
  int* out_pixels = (int *)malloc(sizeof(int)*OUTPUT_HEIGHT*OUTPUT_WIDTH);

  interleave_colors(source_pixels, texture_height, texture_width, texture_image);

  if (rejection_rate)
  {
    //generate white noise image based on random pixels from the original image
    for (int i = 0; i < OUTPUT_HEIGHT; i++)
    {
      for (int j = 0; j < OUTPUT_WIDTH; j++)
      {
        const int randX = std::rand() % (texture_width);
        const int randY = std::rand() % (texture_height);
        
        for (int channel = 0; channel < 3; channel++)
        {
          imgSetRef(out_pixels, OUTPUT_WIDTH, i, j, getRefIndx(texture_width,randY, randX));
        }
      }
    }

    float rr = test_rejection_rate(source_pixels, texture_width, texture_height, out_pixels, OUTPUT_WIDTH, OUTPUT_HEIGHT);

    printf("Total rejection rate: %f\n", rr);

    generate_output(output, OUTPUT_HEIGHT, OUTPUT_WIDTH, source_pixels, out_pixels);
    
    output.save("serial_quilt.jpg");
    return 0;
  }


  double seqTime = 1;
  double cudaTime = 1;

  if (run_both || !run_cuda)
  {
    double totalTime = 0.0;
    for (int trial = 0; trial < numTrials; trial++)
    {
      printf("Running Serial trial %d\n", trial);
      double startTime = CycleTimer::currentSeconds();
      //generate white noise image based on random pixels from the original image
      for (int i = 0; i < OUTPUT_HEIGHT; i++)
      {
        for (int j = 0; j < OUTPUT_WIDTH; j++)
        {
          const int randX = std::rand() % (texture_width);
          const int randY = std::rand() % (texture_height);
          
          for (int channel = 0; channel < 3; channel++)
          {
            imgSetRef(out_pixels, OUTPUT_WIDTH, i, j, getRefIndx(texture_width,randY, randX));
          }
        }
      }

      imagequilt_serial(source_pixels, texture_width, texture_height, out_pixels, OUTPUT_WIDTH, OUTPUT_HEIGHT);

      double endTime = CycleTimer::currentSeconds();
      double trialTime = endTime - startTime;
      totalTime += trialTime;

      printf("Sequential Trial %d: %.3f ms\n", trial, 1000.f* trialTime);

      generate_output(output, OUTPUT_HEIGHT, OUTPUT_WIDTH, source_pixels, out_pixels);
      
      output.save("serial_quilt.jpg", trial, 1);
    }

    seqTime = totalTime / numTrials;
    printf("Average Sequential Time: %.3f ms\n", 1000.f* seqTime);
  }

  if (run_both || run_cuda)
  {
    double totalTime = 0.0;
    for (int trial = 0; trial < numTrials; trial++)
    {
      printf("Running CUDA trial %d\n", trial);
      double startTime = CycleTimer::currentSeconds();
      for (int i = 0; i < OUTPUT_HEIGHT; i++)
      {
        for (int j = 0; j < OUTPUT_WIDTH; j++)
        {
          const int randX = std::rand() % (texture_width);
          const int randY = std::rand() % (texture_height);
          
          for (int channel = 0; channel < 3; channel++)
          {
            imgSetRef(out_pixels, OUTPUT_WIDTH, i, j, getRefIndx(texture_width,randY, randX));
          }
        }
      }
      imagequilt_cuda(texture_width, texture_height, source_pixels, out_pixels);
      double endTime = CycleTimer::currentSeconds();
      double trialTime = endTime - startTime;
      totalTime += endTime - startTime;

      printf("CUDA Trial %d: %.3f ms\n", trial, 1000.f* trialTime);

      generate_output(output, OUTPUT_HEIGHT, OUTPUT_WIDTH, source_pixels, out_pixels);

      output.save("cuda_quilt.jpg", trial, 1);
    }
    cudaTime = totalTime / numTrials;
    printf("Average CUDA Time: %.3f ms\n", 1000.f* cudaTime);
  }

  if (run_both)
  {
    double speedup = seqTime/cudaTime;
    printf("Speedup: %.3f\n", speedup);
  
  }
}
