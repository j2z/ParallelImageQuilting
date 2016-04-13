
#include "CImg.h"


#include <iostream>
#include <typeinfo>


using namespace cimg_library;

int main()
{
  CImg<unsigned char> image("duck.png");
  image.display("duck");
  std::cout << "Width " << image.width() << std::endl;
  std::cout << "Height " << image.height() << std::endl;
  std::cout << "Depth " << image.depth() << std::endl;
  std::cout << "Color channels " << image.spectrum() << std::endl;

  return 0;
}
