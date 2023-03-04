#ifndef Bitmap_H
#define Bitmap_H

using namespace std;

class Bitmap
{
public:
	unsigned short* imageData;
	unsigned short width;
	unsigned short height;

	const int fileHeaderSize = 14;
	const int metadataHeaderSize = 40;

	bool readFromFile(const char* filepath);
	bool writeToFile(const char* filepath);
};
#endif
