
#include "serial_helpers.hpp"
#include <cstdio>
#include <cstdlib>

// assumes the constants used in constants.hpp
// assumes rand has been initialized
void imagequilt_serial(unsigned char* src, int srcWidth, int srcHeight, int* map, int mapWidth, int mapHeight)
{
  for (int iter = 0; iter < ITERATIONS; iter++)
  {
    int xOffset = rand() % TILE_WIDTH;
    int yOffset = rand() % TILE_WIDTH;

    for (int tileY = 0; tileY < HEIGHT_TILES; tileY++)
    {
      for (int tileX = 0; tileX < WIDTH_TILES; tileX++)
      {
        MappingData mapping;
        mapping.src = src;
        mapping.srcWidth = srcWidth;
        mapping.srcX = rand() % (srcWidth - 2*MAX_RADIUS) + MAX_RADIUS;
        mapping.srcY = rand() % (srcHeight - 2*MAX_RADIUS) + MAX_RADIUS;
       
        mapping.map = map;
        mapping.mapWidth = mapWidth;
        mapping.mapX = tileX * TILE_WIDTH + TILE_WIDTH / 2 + xOffset;
        mapping.mapY = tileY * TILE_HEIGHT + TILE_HEIGHT / 2 + yOffset;

        int seam[POLAR_HEIGHT];
      
        bool improvement = seam_carve(mapping, seam);
        
        if (improvement)
        {
          update_map(mapping, seam);
        }
      }
    }
  }
}

float test_rejection_rate(unsigned char* src, int srcWidth, int srcHeight, int* map, int mapWidth, int mapHeight)
{
  int sumTotal = 0;
  int sumRejected = 0;
  for (int iter = 0; iter < ITERATIONS; iter++)
  {
    int total = 0;
    int rejected = 0;
    int xOffset = rand() % TILE_WIDTH;
    int yOffset = rand() % TILE_WIDTH;

    for (int tileY = 0; tileY < HEIGHT_TILES; tileY++)
    {
      for (int tileX = 0; tileX < WIDTH_TILES; tileX++)
      {
        MappingData mapping;
        mapping.src = src;
        mapping.srcWidth = srcWidth;
        mapping.srcX = rand() % (srcWidth - 2*MAX_RADIUS) + MAX_RADIUS;
        mapping.srcY = rand() % (srcHeight - 2*MAX_RADIUS) + MAX_RADIUS;
       
        mapping.map = map;
        mapping.mapWidth = mapWidth;
        mapping.mapX = tileX * TILE_WIDTH + TILE_WIDTH / 2 + xOffset;
        mapping.mapY = tileY * TILE_HEIGHT + TILE_HEIGHT / 2 + yOffset;

        int seam[POLAR_HEIGHT];
      
        bool improvement = seam_carve(mapping, seam);
        
        if (improvement)
        {
          update_map(mapping, seam);
        }
        else
        {
          rejected++;
        }
        total++;
      }
    }

    printf("Iteration %d rejection rate: %f\n", iter, static_cast<float>(rejected) / total);
    sumTotal += total;
    sumRejected += rejected;
  }

  return static_cast<float>(sumRejected) / sumTotal;
}
