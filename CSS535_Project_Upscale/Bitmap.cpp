#include "bitmap.h"

#include <iostream>
#include <fstream>

using namespace std;

bool Bitmap::readFromFile(const char* filepath)
{
	cout << "read" << endl;
	ifstream infile;
	infile.open(filepath, ios::in | ios::binary);
	if (!infile.is_open())
	{
		cout << "error" << endl;
	}
	unsigned char* fileHeader = new unsigned char[Bitmap::fileHeaderSize];
	infile.read(reinterpret_cast<char*>(fileHeader), Bitmap::fileHeaderSize);
	infile.close();
	cout << fileHeader << endl;
	delete[] fileHeader;
	return true;
}

bool Bitmap::writeToFile(const char* filepath)
{
	cout << "write" << endl;
	return false;
}