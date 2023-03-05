#include "bitmap.h"

#include <iostream>
#include <fstream>

using namespace std;

unsigned int Bitmap::imageDataSize()
{
	return Bitmap::imageDataSize(this->width, this->height);
}

unsigned char Bitmap::padSize()
{
	return Bitmap::padSize(this->width);
}

unsigned int Bitmap::imageDataSize(unsigned short width, unsigned short height)
{
	return width * height * 3 + height * Bitmap::padSize(width);
}

unsigned char Bitmap::padSize(unsigned short width)
{
	return  (4 - (width * 3) % 4) % 4;
}

bool Bitmap::readFromFile(const char* filepath)
{
	cout << "read " << filepath << endl;
	if (this->imageData != NULL)
	{
		delete[] this->imageData;
	}
	ifstream infile;
	infile.open(filepath, ios::in | ios::binary);
	if (!infile.is_open())
	{
		cout << "error" << endl;
	}
	unsigned char metadataHeader[40];
	infile.ignore(14);
	infile.read(reinterpret_cast<char*>(metadataHeader), 40);
	this->width = metadataHeader[4] + (metadataHeader[5] << 8) + (metadataHeader[6] << 16) + (metadataHeader[7] << 24);
	this->height = metadataHeader[8] + (metadataHeader[9] << 8) + (metadataHeader[10] << 16) + (metadataHeader[11] << 24);
	this->imageData = new unsigned char[this->imageDataSize()];
	infile.read(reinterpret_cast<char*>(this->imageData), this->imageDataSize());
	/*
	for (int y = 0; y < this->height; ++y)
	{
		for (int x = 0; x < this->width; ++x)
		{
			unsigned char pixel[3];
			infile.read(reinterpret_cast<char*>(pixel), 3);

			int index = y * width * 3 + x * 3;
			imageData[index] = pixel[0];
			imageData[index + 1] = pixel[1];
        	imageData[index + 2] = pixel[2];
		}

		infile.ignore(padSize());
	}*/

	infile.close();

	return true;
}

bool Bitmap::init()
{
	if (this->imageData != NULL)
	{
		delete[] this->imageData;
	}
	if (this->width == 0 || this->height == 0)
	{
		this->imageData = NULL;
		return false;
	}
	this->imageData = new unsigned char[this->imageDataSize()];
	int size = this->imageDataSize();
	for (int i = 0; i < size; ++i)
	{
		this->imageData[i] = 0;
	}
	return true;
}

bool Bitmap::writeToFile(const char* filepath)
{
	if (this->imageData == NULL)
	{
		return false;
	}
	ofstream outfile;
	cout << "write " << filepath << endl;
	outfile.open(filepath, ios::out | ios::binary);
	unsigned char header[14];
	for (int i = 0; i < 14; ++i)
	{
		header[i] = 0x00;
	}
	header[0] = 0x42;
	header[1] = 0x4d;
	header[2] = 0x66;
	header[3] = 0x75;
	header[10] = 0x36;
	outfile.write(reinterpret_cast<char*>(header), 14);
	unsigned char metadata[40];
	for (int i = 0; i < 40; ++i)
	{
		metadata[i] = 0x00;
	}
	metadata[0] = 0x28;
	metadata[4] = this->width;
	metadata[5] = this->width >> 8;
	metadata[6] = this->width >> 16;
	metadata[7] = this->width >> 24;
	metadata[8] = this->height;
	metadata[9] = this->height >> 8;
	metadata[10] = this->height >> 16;
	metadata[11] = this->height >> 24;
	metadata[12] = 0x1;
	metadata[14] = 0x18;
	outfile.write(reinterpret_cast<char*>(metadata), 40);
	int size = 3 * sizeof(char) * this->width * this->height + this->width * padSize();
	outfile.write(reinterpret_cast<char*>(imageData), size);
	outfile.close();
	return true;
}

Bitmap::Bitmap()
{
	this->width = 0;
	this->height = 0;
	this->imageData = NULL;
}

Bitmap::~Bitmap()
{
	if (this->imageData != NULL)
	{
		delete[] this->imageData;
	}
}
