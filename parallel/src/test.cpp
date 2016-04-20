
#include "CImg.h"

#include <cstdio>
#include <iostream>
#include <typeinfo>

double cudaScan(int* start, int* end, int* resultarray);


using namespace cimg_library;

int main()
{
  int* a = (int*)calloc(10,sizeof(int));
  a[0] = 1;
  a[1] = 0;
  a[2] = 1;
  a[3] = 1;
  cudaScan(a,a+10,a);
  printf("%d,%d,%d,%d",a[0],a[1],a[2],a[3]);

  /*CImg<unsigned char> image("duck.png");
  image.display("duck");
  std::cout << "Width " << image.width() << std::endl;
  std::cout << "Height " << image.height() << std::endl;
  std::cout << "Depth " << image.depth() << std::endl;
  std::cout << "Color channels " << image.spectrum() << std::endl;
*/
  return 0;
}
