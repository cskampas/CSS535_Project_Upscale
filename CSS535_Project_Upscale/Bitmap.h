#ifndef Bitmap_H
#define Bitmap_H
#include <string>

using namespace std;

class Bitmap
{
public:
	unsigned short* imageData;
	unsigned short width;
	unsigned short height;

	bool readFromFile(string filepath);
	bool writeToFile(string filepath);
};
#endif
