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

  ColorCu(unsigned char, unsigned char, unsigned char);
  ColorCu(float, float, float);
  ColorCu(const ColorCu&);

  // basically dot product squared
  static float sqDiff(ColorCu, ColorCu);

  // basically vector addition
  friend ColorCu operator+(const ColorCu&, const ColorCu&);
  // basically vector scalar multiplication
  friend ColorCu operator*(float, const ColorCu&);

};

#endif
