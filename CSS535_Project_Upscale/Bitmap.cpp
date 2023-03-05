#include "bitmap.h"

#include <iostream>
#include <fstream>

using namespace std;

bool Bitmap::readFromFile(const char* filepath)
{
	cout << "read" << endl;
	cout << sizeof(unsigned short) << endl;
	cout << sizeof(char) << endl;
	ifstream infile;
	infile.open(filepath, ios::in | ios::binary);
	if (!infile.is_open())
	{
		cout << "error" << endl;
	}
	unsigned char fileHeader[14];
	unsigned char metadataHeader[40];
	infile.read(reinterpret_cast<char*>(fileHeader), 14);
	infile.read(reinterpret_cast<char*>(metadataHeader), 40);
	int fSize = fileHeader[2] + (fileHeader[3] << 8) + (fileHeader[4] << 16) + (fileHeader[5] << 24);
	this->width = metadataHeader[4] + (metadataHeader[5] << 8) + (metadataHeader[6] << 16) + (metadataHeader[7] << 24);
	this->height = metadataHeader[8] + (metadataHeader[9] << 8) + (metadataHeader[10] << 16) + (metadataHeader[11] << 24);
	this->imageData = new unsigned char[this->width * this->height * 3];
	int padsize = ((4 - (this->width * 3) % 4) % 4);

	for (int y = 0; y < this->height; ++y)
	{
		for (int x = 0; x < this->width; ++x)
		{
			unsigned char pixel[3];
			infile.read(reinterpret_cast<char*>(pixel), 3);
			if (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255)
				cout << " ";
			else if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
				cout << "X";
			else if (pixel[0] == 255)
				cout << "R";
			else if (pixel[1] == 255)
				cout << "G";
			else if (pixel[2] == 255)
				cout << "B";
			else
				cout << "?";
		}
		infile.ignore(padsize);
		cout << endl;
	}

	infile.close();
	cout << fileHeader << endl;

	return true;
}

bool Bitmap::writeToFile(const char* filepath)
{
	cout << "write" << endl;
	return false;
}
