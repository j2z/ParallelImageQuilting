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

  __device__ ColorCu(unsigned char, unsigned char, unsigned char);
  __device__ ColorCu(float, float, float);
  __device__ ColorCu(const ColorCu&);

  // basically dot product squared
  __device__ static float sqDiff(ColorCu, ColorCu);

  // basically vector addition
  __device__ friend ColorCu operator+(const ColorCu&, const ColorCu&);
  // basically vector scalar multiplication
  __device__ friend ColorCu operator*(float, const ColorCu&);

};

#endif
