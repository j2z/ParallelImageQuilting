

#ifndef MAPPING_H
#define MAPPING_H

struct MappingData
{
  unsigned char* src;
  int srcWidth;
  int srcX;
  int srcY;

  int* map;
  int mapWidth;
  int mapX;
  int mapY;
};

#endif
