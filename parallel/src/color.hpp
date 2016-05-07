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

  Color(unsigned char, unsigned char, unsigned char);
  Color(float, float, float);
  Color(const Color&);

  // basically dot product squared
  static float sqDiff(Color, Color);

  // basically vector addition
  friend Color operator+(const Color&, const Color&);
  // basically vector scalar multiplication
  friend Color operator*(float, const Color&);

};

#endif
