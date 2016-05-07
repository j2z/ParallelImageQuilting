

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>

#define TILE_WIDTH 120
#define TILE_HEIGHT TILE_WIDTH

#define MAX_RADIUS 60

#define POLAR_WIDTH 60
#define POLAR_HEIGHT 80

//our height for the polar coordinate space is ANGLE_FACTOR * 2 * PI
#define ANGLE_FACTOR (POLAR_HEIGHT / M_PI / 2.f)
//our width for the polar coordinate space is RADIUS_FACTOR * MAX_RADIUS
#define RADIUS_FACTOR (POLAR_WIDTH * 1.f / MAX_RADIUS)


#define HEIGHT_TILES 6
#define WIDTH_TILES 6

#define OUTPUT_HEIGHT (TILE_HEIGHT * HEIGHT_TILES + 2 * MAX_RADIUS)
#define OUTPUT_WIDTH (TILE_WIDTH * WIDTH_TILES + 2 * MAX_RADIUS)

// MAX_RADIUS acts as the amount of padding


#define ITERATIONS 5

#endif

