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
	bool writeToFile(const char* filepath);

	int padSide();

	Bitmap();
	~Bitmap();
};
#endif
