#include "color.hpp"

// construct from unsigned chars
Color::Color(unsigned char r, unsigned char g, unsigned char b)
{
  red = static_cast<float>(r);
  green = static_cast<float>(g);
  blue = static_cast<float>(b);
}

Color::Color(float r, float g, float b):
  red(r), green(g), blue(b)
{
}

// copy (assignment) constructor
Color::Color(const Color& obj)
{
  red = obj.red;
  green = obj.green;
  blue = obj.blue;
}

float Color::sqDiff(Color c1, Color c2)
{
  float redDiff = c1.red - c2.red;
  float greenDiff = c1.green - c2.green;
  float blueDiff = c1.blue - c2.blue;
  return redDiff * redDiff + greenDiff * greenDiff + blueDiff * blueDiff;
}

Color operator+(const Color& c1, const Color& c2)
{
  return Color(c1.red + c2.red, c1.green + c2.green, c1.blue + c2.blue);
}

Color operator*(float scale, const Color& c)
{
  return Color(scale * c.red, scale * c.green, scale * c.blue);
}

