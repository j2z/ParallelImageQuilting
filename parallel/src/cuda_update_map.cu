
#include "cu_helpers.hpp"

__global__ void kernelUpdateMap(int srcWidth, int* map, int xOffset, int yOffset, char* minPaths, bool* improvements, short* samplesX, short* samplesY)
{
  
  int tileX = blockIdx.x;
  int tileY = blockIdx.y;
  int tileIdx = tileY * WIDTH_TILES + tileX;

  if (!improvements[tileIdx])
  {
    return;
  }

  int pixelX;
  int pixelY;

  switch (blockIdx.z)
  {
    case 0:
      pixelX = threadIdx.x - TILE_WIDTH / 2;
      pixelY = threadIdx.y - TILE_HEIGHT / 2;
      break;
    case 1:
      pixelX = threadIdx.x;
      pixelY = threadIdx.y - TILE_HEIGHT / 2;
      break;
    case 2:
      pixelX = threadIdx.x - TILE_WIDTH / 2;
      pixelY = threadIdx.y;
      break;
    case 3:
      pixelX = threadIdx.x;
      pixelY = threadIdx.y;
      break;
  }

  if (pixelX * pixelX + pixelY * pixelY <= MAX_RADIUS * MAX_RADIUS)
  {
    Point polar = offsetToPolar(pixelX, pixelY);

    int theta = (int)round(polar.y) % POLAR_HEIGHT;
    int rad = (int)round(polar.x);

    char* seam = minPaths + tileIdx * POLAR_HEIGHT;

    if (rad <= seam[theta])
    {
      int imgX = samplesX[tileIdx] + pixelX;
      int imgY = samplesY[tileIdx] + pixelY;
      imgSetRef(map, OUTPUT_WIDTH,
                tileY * TILE_HEIGHT + pixelY + TILE_HEIGHT / 2 + yOffset,
                tileX * TILE_WIDTH + pixelX + TILE_WIDTH / 2 + xOffset,
                imgY * srcWidth + imgX);
    }

  }


}
