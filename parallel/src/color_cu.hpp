/** Handles arithmetic operations on colors 
 *  Although everything is floats, it assumes color values were
 *  casted from unsigned chars
 *  The rgb values don't need to be valid 0-255 for arithmetic,
 *  but if you're converting back to unsigned chars, they better be valid
 */

#ifndef COLOR_H
#define COLOR_H

class ColorCu
{
public:
  float red;
  float green;
  float blue;

  // construct from unsigned chars
  __device__ ColorCu(unsigned char r, unsigned char g, unsigned char b)
  {
    red = static_cast<float>(r);
    green = static_cast<float>(g);
    blue = static_cast<float>(b);
  }

  __device__ ColorCu(float r, float g, float b):
    red(r), green(g), blue(b)
  {
  }

  // copy (assignment) constructor
  __device__ ColorCu(const ColorCu& obj)
  {
    red = obj.red;
    green = obj.green;
    blue = obj.blue;
  }

  // basically vector addition
  __device__ friend ColorCu operator+(const ColorCu& c1, const ColorCu& c2)
  {
    return ColorCu(c1.red + c2.red, c1.green + c2.green, c1.blue + c2.blue);
  }

  // basically vector scalar multiplication
  __device__ friend ColorCu operator*(float scale, const ColorCu& c)
  {
    return ColorCu(scale * c.red, scale * c.green, scale * c.blue);
  }
};

// basically dot product squared
inline __device__ float colorSqDiff(ColorCu c1, ColorCu c2)
{
  float redDiff = c1.red - c2.red;
  float greenDiff = c1.green - c2.green;
  float blueDiff = c1.blue - c2.blue;
  return redDiff * redDiff + greenDiff * greenDiff + blueDiff * blueDiff;
}


#endif
