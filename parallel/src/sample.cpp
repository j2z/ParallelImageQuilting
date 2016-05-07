

#include <cmath>
#include "color.hpp"
#include "util.hpp"

// performs bilinear sampling of an image

Color sample(unsigned char* img, int width, float x, float y)
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

  Color out = tl * imgGetColor(img, width, t, l) +
              tr * imgGetColor(img, width, t, r) +
              bl * imgGetColor(img, width, b, l) +
              br * imgGetColor(img, width, b, r);

  return out;
}
