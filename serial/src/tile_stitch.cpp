

#include "CImg.h"

#include <cstring>

using namespace cimg_library;

void vertical_stitch(CImg<int> im1, CImg<int> im2, int x1, int y1, int x2, int y2, int w, int h, int seam[])
{
  float error_map[h][w];
  float min_seam[h][w];
  int back_pointers[h][w];

  memset(error_map, 0.f, h*w);

  for (int channel = 0; channel < 3; channel++)
  {
    for (int j = 0; j < h; j++)
    {
      for (int i = 0; i < w; i++)
      {
        float diff = im1(i,j,channel) - im2(i,j,channel);
        error_map[j][i] += diff * diff;
      }
    }
  }

  for (int i = 0; i < w; i++)
  {
    min_seam[0][i] = error_map[0][i];
  }

  for (int j = 1; j < h; j++)
  {
    if (min_seam[j-1][0] < min_seam[j-1][1])
    {
      // center is smallest
      min_seam[j][0] = min_seam[j-1][0] + error_map[j][0];
      back_pointers[j][0] = 0;
    }
    else
    {
      // right is smallest
      min_seam[j][0] = min_seam[j-1][1] + error_map[j][0];
      back_pointers[j][0] = 1;
    }

    for (int i = 1; i < w-1; i++)
    {
      if (min_seam[j-1][i-1] < min_seam[j-1][i])
      {
        if (min_seam[j-1][j-1] < min_seam[j-1][i+1])
        {
          // left is smallest
          min_seam[j][i] = min_seam[j-1][i-1] + error_map[j][i];
          back_pointers[j][i] = i-1;
        }
        else
        {
          // right is smallest
          min_seam[j][i] = min_seam[j-1][i+1] + error_map[j][i];
          back_pointers[j][i] = i+1;
        }
      }
      else
      {
        if (min_seam[j-1][i] < min_seam[j-1][i+1])
        {
          // center is smallest
          min_seam[j][i] = min_seam[j-1][i] + error_map[j][i];
          back_pointers[j][i] = i;
        }
        else
        {
          // right is smallest
          min_seam[j][i] = min_seam[j-1][i+1] + error_map[j][i];
          back_pointers[j][i] = i+1;
        }
      }
    }

    if (min_seam[j-1][w-2] < min_seam[j-1][w-1])
    {
      // left is smallest
      min_seam[j][w-1] = min_seam[j-1][w-2] + error_map[j][w-1];
      back_pointers[j][w-1] = w-2;
    }
    else
    {
      // center is smallest
      min_seam[j][w-1] = min_seam[j-1][w-1] + error_map[j][w-1];
      back_pointers[j][w-1] = w-1;
    }
  }

  int best_seam = 0;
  for (int i = 1; i < w; i++)
  {
    if (min_seam[h-1][i] < min_seam[h-1][best_seam])
    {
      best_seam = i;
    }
  }

  int index = best_seam;
  for (int j = h-1; j >= 0; j--)
  {
    seam[j] = index;
    index = back_pointers[j][index];
  }

}
