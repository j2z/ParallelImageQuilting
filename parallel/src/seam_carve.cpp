
#include "serial_helpers.hpp"
#include "color.hpp"
#include "util.hpp"
#include <cmath>

#include <iostream>

#define MAX_JUMP 1
#define IMPROVE_THRESH 0.f

// pass in an array for the polar coordinate error map
// with constants defined in constants.hpp

bool seam_carve(MappingData& mapping, int seam[POLAR_HEIGHT])
{
  // create 2 buffers for currentRow and previousRow
  float array1[POLAR_WIDTH];
  float array2[POLAR_WIDTH];

  float* currentRow = array1;

  // First row:
  // accumulate existing error on the left of the seam
  float existingError = 0.0;
  for (int rho = 0; rho < POLAR_WIDTH; rho++)
  {
    existingError += existing_error(mapping, rho, 0);
    // total error = h + v - epsilon
    // we don't use v currently
    currentRow[rho] = horiz_error(mapping, rho, 0) - existingError;
  }

  // backpointers array
  int backPointers[POLAR_HEIGHT][POLAR_WIDTH];


  float* previousRow = array1;
  currentRow = array2;

  // rest of the rows
  for (int theta = 1; theta < POLAR_HEIGHT; theta++)
  {
    existingError = 0.0;

    for (int rho = 0; rho < POLAR_WIDTH; rho++)
    {
      existingError += existing_error(mapping, rho, theta);
      
      // find the min from the previous row
      int minIndex = -1;
      float minVal = 0.0;
      for (int arg = rho - MAX_JUMP; arg <= rho + MAX_JUMP; arg++)
      {
        if (arg >= 0 && arg < POLAR_WIDTH)
        {
          if (minIndex == -1 || previousRow[arg] < minVal)
          {
            minIndex = arg;
            minVal = previousRow[arg];
          }
        }
      }

      currentRow[rho] = minVal + horiz_error(mapping, rho, theta) - existingError;
      backPointers[theta][rho] = minIndex;
    }

    // put current into previous, use the memory in previous as the new current
    float *temp = previousRow;
    previousRow = currentRow;
    currentRow = temp;

  }

  // at this point, the seam costs should be stored in previousRow

  // pick the best seam which also wraps around
  int minSeam = -1;
  float minVal = 0.0;
  for (int i = 0; i < POLAR_WIDTH; i++)
  {
    int index = backPointers[POLAR_HEIGHT-1][i];
    for (int step = POLAR_HEIGHT - 2; step > 0; step--)
    {
      index = backPointers[step][index];
    }
    // if we have wraparound
    if (index == i)
    {
      if (minSeam == -1 || previousRow[i] < minVal)
      {
        minSeam = i;
        minVal = previousRow[i];
      }
    }
  }

  // if the seams we cover up don't outweigh the cost of the new seam
  if (minVal >= IMPROVE_THRESH)
  {
    return false;
  }

  // copy min seam to seam[]
  int index = minSeam;
  for (int step = POLAR_HEIGHT - 1; step >= 0; step--)
  {
    seam[step] = index;
    index = backPointers[step][index];
  }
  return true;
}


