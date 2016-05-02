

#include "CImg.h"

#include <iostream>
#include <ctime>


#define TILE_HEIGHT 36
#define TILE_WIDTH 36
#define OVERLAP 6

#define TILE_HEIGHT_REM (TILE_HEIGHT - OVERLAP)
#define TILE_WIDTH_REM (TILE_WIDTH - OVERLAP)

#define WIDTH_TILES 5
#define HEIGHT_TILES 5


using namespace cimg_library;

void vertical_stitch(CImg<int> im1, CImg<int> im2, int x1, int y1, int x2, int y2, int w, int h, int seam[]);
void horizontal_stitch(CImg<int> im1, CImg<int> im2, int x1, int y1, int x2, int y2, int w, int h, int seam[]);

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

  // set up PRNG
  std::srand(time(NULL));
  
  CImg<int> texture_image(argv[1]);

  const int texture_height = texture_image.height();
  const int texture_width = texture_image.width();

  const int output_height = HEIGHT_TILES * TILE_HEIGHT_REM + OVERLAP;
  const int output_width = WIDTH_TILES * TILE_WIDTH_REM + OVERLAP;

  CImg<int> output(output_width, output_height, 1, 3);

  int patchSSDs[texture_width-TILE_WIDTH][texture_height-TILE_HEIGHT];
  int vertical_seam[TILE_HEIGHT];
  int horizontal_seam[TILE_WIDTH];

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

      // copy based on seam
      if (i != 0)
      {
        vertical_stitch(texture_image, output,
                        patchX, patchY,
                        outX, outY,
                        OVERLAP, TILE_HEIGHT, vertical_seam);
      }
      if (j != 0)
      {
        horizontal_stitch(texture_image, output,
                          patchX, patchY,
                          outX, outY,
                          TILE_WIDTH, OVERLAP, horizontal_seam);
      }
      for (int channel = 0; channel < 3; channel++)
      {
        for (int y = 0; y < TILE_HEIGHT; y++)
        {
          int x_init = 0;
          if (i != 0)
          {
            x_init = vertical_seam[y];
          }
          for (int x = x_init; x < TILE_WIDTH; x++)
          {
            if (j == 0 || y > horizontal_seam[x])
            {
              output(outX + x, outY + y, channel) = texture_image(patchX + x, patchY + y, channel);
            }
            //if (j != 0) output(outX + x, outY + horizontal_seam[x], channel) = 255;
          }
          //if (i != 0) output(outX + vertical_seam[y], outY + y, channel) = 255;
        }
      }

      std::cout << "Tile " << i << ", " << j << " done" << std::endl;
      std::cout << "Pulled from " << patchX << ", " << patchY << std::endl;
      output.save("output.jpg", j*WIDTH_TILES + i, 2);
    }

  }

  if (argc < 3)
  {
    output.display();
  }
  else
  {
    output.save(argv[2]);
  }
}
