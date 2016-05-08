
#include "serial_helpers.hpp"
#include "util.hpp"
#include <cmath>

// updates the map to use the coordinates from src based on the polar cutout
// defined in the seam
void update_map(MappingData& mapping, int seam[POLAR_HEIGHT])
{
  int maxDist = MAX_RADIUS * MAX_RADIUS;
  for (int j = -MAX_RADIUS; j <= MAX_RADIUS; j++)
  {
    for (int i = -MAX_RADIUS; i <= MAX_RADIUS; i++)
    {
      if (i*i + j*j > maxDist)
      {
        continue;
      }

      Point polar = offsetToPolar(i, j);

      // right now just use rounding, can check if more advanced methods give
      // better results

      // need to wrap around if necessary
      int theta = static_cast<int>(round(polar.y)) % POLAR_HEIGHT;

      // rad should never exceed the max radius
      int rad = static_cast<int>(round(polar.x));

      if (rad <= seam[theta])
      {
        int imgX = mapping.srcX + i;
        int imgY = mapping.srcY + j;
        imgSetRef(mapping.map, mapping.mapWidth,
                  mapping.mapY + j, mapping.mapX + i,
                  imgY * mapping.srcWidth + imgX);
      }

    }
  }
  
}
