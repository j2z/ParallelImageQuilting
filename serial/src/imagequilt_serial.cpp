

#include "CImg.h"

#include <iostream>


#define TILE_HEIGHT 60
#define TILE_WIDTH 60
#define OVERLAP 10

#define TILE_HEIGHT_REM (TILE_HEIGHT - OVERLAP)
#define TILE_WIDTH_REM (TILE_WIDTH - OVERLAP)

#define WIDTH_TILES 20
#define HEIGHT_TILES 20


using namespace cimg_library;


static inline void disp_help()
{
  std::cout << "Needs 1 argument: filename of texture image (JPEG)" << std::endl;
}

// 2 images of same size 
static inline float ssd(CImg<int> im1, CImg<int> im2)
{
  return (im1 - im2).get_sqr().sum();
}


int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    disp_help();
    return 0;
  }
  
  CImg<int> image(argv[1]);
  CImg<int> im2(image);

  std::cout << ssd(image, im2) << std::endl;

  image.display();


  for (int j = 1; j < HEIGHT_TILES; j++)
  {
    for (int i = 1; i < WIDTH_TILES; i++)
    {
      
    }
  }

}
