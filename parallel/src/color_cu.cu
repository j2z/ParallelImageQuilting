#include "color_cu.hpp"

// construct from unsigned chars
__device__ ColorCu::ColorCu(unsigned char r, unsigned char g, unsigned char b)
{
  red = static_cast<float>(r);
  green = static_cast<float>(g);
  blue = static_cast<float>(b);
}

__device__ ColorCu::ColorCu(float r, float g, float b):
  red(r), green(g), blue(b)
{
}

// copy (assignment) constructor
__device__ ColorCu::ColorCu(const ColorCu& obj)
{
  red = obj.red;
  green = obj.green;
  blue = obj.blue;
}

inline __device__ float ColorCu::sqDiff(ColorCu c1, ColorCu c2)
{
  float redDiff = c1.red - c2.red;
  float greenDiff = c1.green - c2.green;
  float blueDiff = c1.blue - c2.blue;
  return redDiff * redDiff + greenDiff * greenDiff + blueDiff * blueDiff;
}

inline __device__ ColorCu operator+(const ColorCu& c1, const ColorCu& c2)
{
  return ColorCu(c1.red + c2.red, c1.green + c2.green, c1.blue + c2.blue);
}

inline __device__ ColorCu operator*(float scale, const ColorCu& c)
{
  return ColorCu(scale * c.red, scale * c.green, scale * c.blue);
}

