// Care: these functions contain lots of implicit conversions
// between floats and ints and vice versa

#include "cu_helpers.hpp"

PolarTransformationCu::PolarTransformationCu(int r, float rf, float af):
  radius(r),
  radiusFactor(rf),
  angleFactor(af)
{
}

// Transforms offsets to whereever the center is defined into polar
inline __device__ Point PolarTransformationCu::offsetToPolar(int x, int y)
{
  Point out;
  // r
  float fx = (float)x;
  float fy = (float)y;

  out.x = sqrt(fx*fx + fy*fy) * radiusFactor;
  // theta
  out.y = atan2(fy,fx);
  if (out.y < 0.f)
  {
    out.y += 2 * M_PI;
  }
  out.y *= angleFactor;
  return out;
}

// transforms polar to coordinates that are offsets to whereever the
// center is defined
inline __device__ Point PolarTransformationCu::polarToOffset(int r, int theta)
{
  Point out;
  float rEff = r / radiusFactor;
  float thetaEff = theta / angleFactor;
  out.x = round(rEff * cos(thetaEff));
  out.y = round(rEff * sin(thetaEff));
  return out;
}


inline __device__ Point PolarTransformationCu::absoluteToPolar(int cx, int cy, int x, int y)
{
  return offsetToPolar(x-cx, y-cy);
}


inline __device__ Point PolarTransformationCu::polarToAbsolute(int cx, int cy, int r, int theta)
{
  Point out = polarToOffset(r, theta);
  out.x += cx;
  out.y += cy;
  return out;
}

inline __device__ int PolarTransformationCu::getRadius()
{
  return this->radius;
}

