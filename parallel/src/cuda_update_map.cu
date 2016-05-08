
#include "constants.hpp"
#include "point.hpp"
#include "util_cu.hpp"
#include "polar_transform_cu.hpp"
#include <cuda.h>


__global__ void kernelUpdateMap(int srcWidth, int* map, int mapWidth, int xOffset, int yOffset, char* minPaths, short* samplesX, short* samplesY)
{
  
  int tileX = blockIdx.x;
  int tileY = blockIdx.y;
  int pixelX = threadIdx.x - TILE_WIDTH / 2;
  int pixelY = threadIdx.y - TILE_HEIGHT / 2;

  int tileIdx = tileY * WIDTH_TILES + tileX;
  int maxDist = MAX_RADIUS * MAX_RADIUS;

  if (pixelX * pixelX + pixelY * pixelY <= maxDist)
  {
    Point polar = PolarTransformationCu(MAX_RADIUS, RADIUS_FACTOR, ANGLE_FACTOR).offsetToPolar(pixelX, pixelY);

    int theta = (int)round(polar.y) % POLAR_HEIGHT;
    int rad = (int)round(polar.x);

    char* seam = minPaths + tileIdx * POLAR_HEIGHT;

    if (rad <= seam[theta])
    {
      int imgX = samplesX[tileIdx] + pixelX;
      int imgY = samplesY[tileIdx] + pixelY;
      imgSetRef(map, mapWidth,
                tileY * TILE_WIDTH + threadIdx.y,
                tileX * TILE_WIDTH + threadIdx.x,
                imgY * srcWidth + imgX);
    }

  }


}
