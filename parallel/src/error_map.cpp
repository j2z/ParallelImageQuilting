/** The error function is defined in the paper
 */

#include "constants.hpp"
#include "polar_transform.hpp"
#include <cmath>
#include "util.hpp"

// regularization factor to make sure denom isn't too big
#define ETA 2.f


// normalization factor for polar coordinates
static inline float normFactor(int r)
{
  return 2 * M_PI * r / POLAR_HEIGHT / RADIUS_FACTOR / RADIUS_FACTOR;
}



// fills a polar coordinate rectangle with error calculated from 
// the source image
void error_map( float error[POLAR_HEIGHT][POLAR_WIDTH],
                unsigned char* src, int srcWidth, int srcX, int srcY,
                unsigned char* dst, int dstWidth, int dstX, int dstY,
                PolarTransformation transform )
{
  for (int theta = 0; theta < POLAR_HEIGHT; theta++)
  {
    for (int rho = 0; rho < POLAR_WIDTH; rho++)
    {
      Point src1 = transform.polarToAbsolute(srcX,srcY,rho,theta);
      Point src2 = transform.polarToAbsolute(srcX,srcY,rho+1,theta);
      float srcColor1 = static_cast<float>(sample(src, srcWidth, src1.x, src1.y));
      float srcColor2 = static_cast<float>(sample(src, srcWidth, src2.x, src2.y));

      Point dst1 = transform.polarToAbsolute(dstX,dstY,rho,theta);
      Point dst2 = transform.polarToAbsolute(dstX,dstY,rho+1,theta);
      float dstColor1 = static_cast<float>(sample(dst, dstWidth, dst1.x, dst1.y));
      float dstColor2 = static_cast<float>(sample(dst, dstWidth, dst2.x, dst2.y));

      float diff1 = srcColor1 - dstColor2;
      float diff2 = srcColor2 - dstColor1;
      float diff3 = srcColor1 - srcColor2;
      float diff4 = dstColor1 - dstColor2;

      float error1 = (diff1*diff1 + diff2*diff2) / (ETA + diff3*diff3 + diff4*diff4);
      error[theta][rho] = error1 * error1;
    }
  }
}
