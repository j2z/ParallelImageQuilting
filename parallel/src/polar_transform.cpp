// Care: these functions contain lots of implicit conversions
// between floats and ints and vice versa

#include <cmath>
#include "polar_transform.hpp"

PolarTransformation::PolarTransformation(int r, float rf, float af):
  radius(r),
  radiusFactor(rf),
  angleFactor(af)
{
}

// Transforms offsets to whereever the center is defined into polar
Point PolarTransformation::offsetToPolar(int x, int y)
{
  Point out;
  // r
  out.x = sqrt(x*x + y*y) * radiusFactor;
  // theta
  out.y = atan2(y,x);
  if (out.y < 0.f)
  {
    out.y += 2 * M_PI;
  }
  out.y *= angleFactor;
  return out;
}

// transforms polar to coordinates that are offsets to whereever the
// center is defined
Point PolarTransformation::polarToOffset(int r, int theta)
{
  Point out;
  float rEff = r / radiusFactor;
  float thetaEff = theta / angleFactor;
  out.x = rEff * cos(thetaEff);
  out.y = rEff * sin(thetaEff);
  return out;
}


Point PolarTransformation::absoluteToPolar(int cx, int cy, int x, int y)
{
  return offsetToPolar(x-cx, y-cy);
}


Point PolarTransformation::polarToAbsolute(int cx, int cy, int r, int theta)
{
  Point out = polarToOffset(r, theta);
  out.x += cx;
  out.y += cy;
  return out;
}

int PolarTransformation::getRadius()
{
  return this->radius;
}

