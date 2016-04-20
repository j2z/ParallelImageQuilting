

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
  
  CImg<int> texture_image(argv[1]);

  const int texture_height = texture_image.height();
  const int texture_width = texture_image.width();

  const int output_height = HEIGHT_TILES * TILE_HEIGHT_REM;
  const int output_width = WIDTH_TILES * TILE_WIDTH_REM;

  CImg<int> output(output_width, output_height, 1, 3);

  

  for (int j = 0; j < HEIGHT_TILES; j++)
  {
    for (int i = 0; i < WIDTH_TILES; i++)
    {
      const int patchX = std::rand() % (texture_width - TILE_WIDTH);
      const int patchY = std::rand() % (texture_height - TILE_HEIGHT);

      const int outX = i * TILE_WIDTH_REM;
      const int outY = j * TILE_HEIGHT_REM;

      for (int x = 0; x < TILE_WIDTH_REM; x++)
      {
        for (int y = 0; y < TILE_HEIGHT_REM; y++)
        {
          output(outX + x, outY + y, 0) = texture_image(patchX + x, patchY + y, 0);
          output(outX + x, outY + y, 1) = texture_image(patchX + x, patchY + y, 1);
          output(outX + x, outY + y, 2) = texture_image(patchX + x, patchY + y, 2);
        }
      }

    }
  }

  output.display();

}
