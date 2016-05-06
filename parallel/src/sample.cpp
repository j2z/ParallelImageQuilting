

#include <cmath>
#include "util.hpp"

// performs bilinear sampling of an image

unsigned char sample(unsigned char* img, int width, float x, float y)
{
  float left = floor(x);
  float right = ceil(x);
  float top = floor(y);
  float bot = ceil(y);

  float tl = (right - x) * (bot - y);
  float tr = (x - left) * (bot - y);
  float bl = (right - x) * (y - top);
  float br = (x - left) * (y - top);

  int l = static_cast<int>(left);
  int r = static_cast<int>(right);
  int b = static_cast<int>(bot);
  int t = static_cast<int>(top);

  // remember that images are stored in y,x
  float out = tl * static_cast<float>(imgGet(img, width, t, l)) +
              tr * static_cast<float>(imgGet(img, width, t, r)) +
              bl * static_cast<float>(imgGet(img, width, b, l)) +
              br * static_cast<float>(imgGet(img, width, b, r));

  return static_cast<unsigned char>(out);
}
