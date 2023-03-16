#ifndef Bitmap_H
#define Bitmap_H

using namespace std;

class Bitmap
{
public:
	unsigned char* imageData;
	unsigned short width;
	unsigned short height;

	bool readFromFile(const char* filepath);
	bool init();
	bool writeToFile(const char* filepath);

	unsigned int imageDataSize();
	unsigned char padSize();
	static unsigned int imageDataSize(unsigned short width, unsigned short height);
	static unsigned char padSize(unsigned short width);

	Bitmap();
	~Bitmap();
};
#endif
