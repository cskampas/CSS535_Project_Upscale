#ifndef Bitmap_H
#define Bitmap_H

using namespace std;

class Bitmap
{
public:
	unsigned char* imageData;
	unsigned int width;
	unsigned int height;

	bool readFromFile(const char* filepath);
	bool init();
	bool writeToFile(const char* filepath);

	unsigned int imageDataSize();
	unsigned char padSize();
	static unsigned int imageDataSize(unsigned int width, unsigned int height);
	static unsigned char padSize(unsigned int width);

	Bitmap();
	~Bitmap();
};
#endif
