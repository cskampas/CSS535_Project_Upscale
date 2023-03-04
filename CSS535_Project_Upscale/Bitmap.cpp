#include "bitmap.h"

#include <iostream>
#include <fstream>

using namespace std;

bool Bitmap::readFromFile(string filepath)
{
	cout << "read" << endl;
	ifstream infile;
	infile.open(filepath);

	infile.close();
	return true;
}

bool Bitmap::writeToFile(string filepath)
{
	cout << "write" << endl;
	return false;
}
