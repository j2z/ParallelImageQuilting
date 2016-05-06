
#include <cmath>
#include "constants.hpp"
#include "polar_transform.hpp"
#include "util.hpp"

// Copies circular patch at (srcX,srcY) in src to (dstX,dstY) in dst
// according to a seam defined in polar space
void copy_patch(unsigned char* src, unsigned char* dst, int srcWidth, int dstWidth,
                int srcX, int srcY, int dstX, int dstY,
                PolarTransformation transform, int seam[])
{
  int r = transform.getRadius();
  int maxDist = r*r;
  for (int j = -r; j <= r; j++)
  {
    for (int i = -r; i <= r; i++)
    {
      if (i*i + j*j > maxDist)
      {
        continue;
      }

      Point polar = transform.offsetToPolar(i, j);

      // right now just use rounding, can check if more advanced methods give
      // better results

      // need to wrap around if necessary
      int theta = static_cast<int>(round(polar.y)) % POLAR_HEIGHT;

      // rad should never exceed the max radius
      int rad = static_cast<int>(round(polar.x));

      if (rad <= seam[theta])
      {
        imgSet(dst, dstWidth, dstY + j, dstX + i,
          imgGet(src, srcWidth, srcY + j, srcX + i));
      }

    }
  }
}
