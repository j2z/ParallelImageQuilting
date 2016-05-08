
#include "serial_helpers.hpp"
#include <cstdlib>

// assumes the constants used in constants.hpp
// assumes rand has been initialized
void imagequilt_serial(unsigned char* src, int srcWidth, int srcHeight, int* map, int mapWidth, int mapHeight)
{
  // right now, same radius for every iter
  PolarTransformation transform(MAX_RADIUS, RADIUS_FACTOR, ANGLE_FACTOR);

  for (int iter = 0; iter < ITERATIONS; iter++)
  {
    int xOffset = rand() % TILE_WIDTH;
    int yOffset = rand() % TILE_WIDTH;

    for (int tileY = 0; tileY < HEIGHT_TILES; tileY++)
    {
      for (int tileX = 0; tileX < WIDTH_TILES; tileX++)
      {
        int mapX = tileX * TILE_WIDTH + TILE_WIDTH / 2 + xOffset;
        int mapY = tileY * TILE_HEIGHT + TILE_HEIGHT / 2 + yOffset;

        int srcY = rand() % (srcHeight - 2*MAX_RADIUS) + MAX_RADIUS;
        int srcX = rand() % (srcWidth - 2*MAX_RADIUS) + MAX_RADIUS;

        ErrorFunction errFunc(src, srcWidth, srcX, srcY, map, mapWidth, mapX, mapY, transform);

        int seam[POLAR_HEIGHT];
      
        bool improvement = seam_carve(errFunc, seam);
        
        if (improvement)
        {
          update_map(src, srcWidth, srcX, srcY, map, mapWidth, mapX, mapY, transform, seam);
        }
      }
    }
  }
}
