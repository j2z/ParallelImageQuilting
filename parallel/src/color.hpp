/** Handles arithmetic operations on colors 
 *  Although everything is floats, it assumes color values were
 *  casted from unsigned chars
 *  The rgb values don't need to be valid 0-255 for arithmetic,
 *  but if you're converting back to unsigned chars, they better be valid
 */

#ifndef COLOR_H
#define COLOR_H

class Color
{
public:
  float red;
  float green;
  float blue;
  
  // construct from unsigned chars
  Color(unsigned char r, unsigned char g, unsigned char b)
  {
    red = static_cast<float>(r);
    green = static_cast<float>(g);
    blue = static_cast<float>(b);
  }

  Color(float r, float g, float b):
    red(r), green(g), blue(b)
  {
  }

  // copy (assignment) constructor
  Color(const Color& obj)
  {
    red = obj.red;
    green = obj.green;
    blue = obj.blue;
  }

  // basically vector addition
  friend Color operator+(const Color& c1, const Color& c2)
  {
    return Color(c1.red + c2.red, c1.green + c2.green, c1.blue + c2.blue);
  }

  // basically vector scalar multiplication
  friend Color operator*(float scale, const Color& c)
  {
    return Color(scale * c.red, scale * c.green, scale * c.blue);
  }
};

// basically dot product squared
inline float colorSqDiff(Color c1, Color c2)
{
  float redDiff = c1.red - c2.red;
  float greenDiff = c1.green - c2.green;
  float blueDiff = c1.blue - c2.blue;
  return redDiff * redDiff + greenDiff * greenDiff + blueDiff * blueDiff;
}


#endif
