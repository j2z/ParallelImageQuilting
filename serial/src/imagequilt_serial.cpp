

#include "CImg.h"

#include <iostream>


#define TILE_HEIGHT 60
#define TILE_WIDTH 60
#define OVERLAP 10

#define TILE_HEIGHT_REM (TILE_HEIGHT - OVERLAP)
#define TILE_WIDTH_REM (TILE_WIDTH - OVERLAP)

#define WIDTH_TILES 5
#define HEIGHT_TILES 5


using namespace cimg_library;


static inline void disp_help()
{
  std::cout << "Needs 1 argument: filename of texture image (JPEG)" << std::endl;
}

// 2 images of same size 
//static inline float ssd(CImg<int> im1, CImg<int> im2)
//{
//  return (im1 - im2).get_sqr().sum();
//}


// ssd of wxh region at (x1,y1) in im1 and (x2,y2) in im2 (in all 3 colors)
static inline float ssd(CImg<int> im1, CImg<int> im2, int x1, int y1, int x2, int y2, int w, int h)
{
  float sum = 0.0;
  for (int channel = 0; channel < 3; channel++)
  {
    for (int j = 0; j < h; j++)
    {
      for (int i = 0; i < w; i++)
      {
        float diff = im1(x1 + i, y1 + j, channel) - im2(x2 + i, y2 + j, channel);
        sum += diff * diff;
      }
    }
  }
  return sum;
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

  CImg<int> output(output_width + OVERLAP, output_height + OVERLAP, 1, 3);

  int patchSSDs[texture_width-TILE_WIDTH][texture_height-TILE_HEIGHT];

  // random seed patch
  const int patchX = std::rand() % (texture_width - TILE_WIDTH);
  const int patchY = std::rand() % (texture_height - TILE_HEIGHT);
  for (int channel = 0; channel < 3; channel++)
  {
    for (int y = 0; y < TILE_HEIGHT; y++)
    {
      for (int x = 0; x < TILE_WIDTH; x++)
      {
        output(x, y, channel) = texture_image(patchX + x, patchY + y, channel);
      }
    }
  }
  std::cout << "Tile 0, 0 done" << std::endl;
  std::cout << "Pulled from " << patchX << ", " << patchY << std::endl;
  output.save("output.jpg", 0, 2);


  for (int j = 0; j < HEIGHT_TILES; j++)
  {
    const int outY = j * TILE_HEIGHT_REM;

    for (int i = 0; i < WIDTH_TILES; i++)
    {
      const int outX = i * TILE_WIDTH_REM;

      // find best patch with SSD
      if (i == 0)
      {
        if (j == 0)
        {
          continue;
        }
        for (int y = 0; y < texture_height - TILE_HEIGHT; y++)
        {
          for (int x = 0; x < texture_width - TILE_WIDTH; x++)
          {
            // horizontal strip only
            patchSSDs[x][y] = ssd(texture_image, output, x,y, outX,outY, TILE_WIDTH, OVERLAP);
          }
        }
      }
      else if (j == 0)
      {
        for (int y = 0; y < texture_height - TILE_HEIGHT; y++)
        {
          for (int x = 0; x < texture_width - TILE_WIDTH; x++)
          {
            // vertical strip only
            patchSSDs[x][y] = ssd(texture_image, output, x,y, outX,outY, OVERLAP, TILE_HEIGHT);
          }
        }
      }
      else
      {
        for (int y = 0; y < texture_height - TILE_HEIGHT; y++)
        {
          for (int x = 0; x < texture_width - TILE_WIDTH; x++)
          {
            // vertical strip
            patchSSDs[x][y] = ssd(texture_image, output, x,y, outX,outY, OVERLAP, TILE_HEIGHT);
            // horizontal strip, but not including area already done by vertical strip
            patchSSDs[x][y] += ssd( texture_image, output,
                                    x+OVERLAP, y,
                                    outX+OVERLAP, outY,
                                    TILE_WIDTH_REM, OVERLAP );
          }
        }
      }


      // choose a good matching patch
      int patchX = 0;
      int patchY = 0;

      for (int x = 0; x < texture_width - TILE_WIDTH; x++)
      {
        for (int y = 0; y < texture_height - TILE_HEIGHT; y++)
        {
          if (patchSSDs[x][y] < patchSSDs[patchX][patchY])
          {
            patchX = x;
            patchY = y;
          }
        }
      }

      // copy patch over
      for (int channel = 0; channel < 3; channel++)
      {
        for (int y = 0; y < TILE_HEIGHT; y++)
        {
          for (int x = 0; x < TILE_WIDTH; x++)
          {
            output(outX + x, outY + y, channel) = texture_image(patchX + x, patchY + y, channel);
          }
        }
      }
      std::cout << "Tile " << i << ", " << j << " done" << std::endl;
      std::cout << "Pulled from " << patchX << ", " << patchY << std::endl;
      output.save("output.jpg", j*WIDTH_TILES + i, 2);
    }

  }


}
