

#ifndef IMAGE_DATA_H
#define IMAGE_DATA_H

#include "CImg.h"

void interleave_colors(unsigned char*, int, int, cimg_library::CImg<unsigned char>&);

void generate_output(cimg_library::CImg<unsigned char>&, int, int, unsigned char*, int*);


#endif
