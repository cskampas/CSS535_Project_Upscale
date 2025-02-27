﻿// For Visual Studio intelisense, mainly
#if _MSC_VER
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include <iostream>
#include <math.h>
#include <chrono>

#include "bitmap.h"
#include "debugFeatures.h"

using namespace std;
using namespace std::chrono;

/// <summary>
/// Debug function.  Prints bitmap into standard output.
/// White pixels are spaces, black pixels are X,
/// pure red, green, and blue pixels are R, G, and B
/// all other mixed-channel color print as ?
/// This is useful for simple test images
/// </summary>
/// <param name="matrix">Color channel array</param>
/// <param name="width">Image width</param>
/// <param name="height">Image height</param>
/// <param name="pad">Padding dimension size</param>
void print_matrix(unsigned char* matrix, unsigned short width, unsigned short height, int pad){
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			unsigned char* pixel = matrix + (y * width * 3 + x * 3 + y * pad);

			if (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255)
				cout << " ";
			else if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
				cout << "X";
			else if (pixel[2] == 255)
				cout << "R";
			else if (pixel[1] == 255)
				cout << "G";
			else if (pixel[0] == 255)
				cout << "B";
			else
				cout << "?";
		}
		cout << endl;
	}
}

/// <summary>
/// Nearest neighbor CUDA kernel (Naive)
/// </summary>
/// <param name="source">Source image color channel matrix</param>
/// <param name="oWidth">Original image width</param>
/// <param name="oHeight">Original image height</param>
/// <param name="oPad">Original image padding size</param>
/// <param name="dest">Destination image color matrix</param>
/// <param name="nWidth">New image width</param>
/// <param name="nHeight">New image height</param>
/// <param name="nPad">New image padding size</param>
/// <returns></returns>
__global__ void NearestNeighbor(
	unsigned char* source,
	unsigned short oWidth,
	unsigned short oHeight,
	unsigned char oPad,
	unsigned char* dest,
	unsigned short nWidth,
	unsigned short nHeight,
	unsigned char nPad)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col >= nWidth || row >= nHeight)
	{
		return;
	}

	int index = ((col + row * nWidth) * 3) + row * nPad;

	int oCol = (int)(((float)col / (float)nWidth) * oWidth);
	int oRow = (int)(((float)row / (float)nHeight) * oHeight);

	if (oCol < 0)
	{
		oCol = 0;
	}
	if (oCol >= oHeight)
	{
		oCol = oHeight - 1;
	}
	if (oRow < 0)
	{
		oRow = 0;
	}
	if (oRow >= oHeight)
	{
		oRow = oHeight - 1;
	}

	int oIndex = ((oCol + oRow * oWidth) * 3) + oRow * oPad;

	dest[index] = source[oIndex];
	dest[index + 1] = source[oIndex + 1];
	dest[index + 2] = source[oIndex + 2];
}

/// <summary>
/// Bilinear neighbor CUDA kernel (Naive)
/// </summary>
/// <param name="source">Source image color channel matrix</param>
/// <param name="oWidth">Original image width</param>
/// <param name="oHeight">Original image height</param>
/// <param name="oPad">Original image padding size</param>
/// <param name="dest">Destination image color matrix</param>
/// <param name="nWidth">New image width</param>
/// <param name="nHeight">New image height</param>
/// <param name="nPad">New image padding size</param>
/// <returns></returns>
__global__ void Bilinear(
	unsigned char* source,
	unsigned short oWidth,
	unsigned short oHeight,
	unsigned char oPad,
	unsigned char* dest,
	unsigned short nWidth,
	unsigned short nHeight,
	unsigned char nPad)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col >= nWidth || row >= nHeight)
	{
		return;
	}
	int index = ((col + row * nWidth) * 3) + row * nPad;

	// Find left and right pixel from row above and row below
	// "Top" and "Left" here means towards 0, regardless of the reality of the image format

	float sourceRelativeRow = (float)row / (float)nHeight;
	float sourceRelativeCol = (float)col / (float)nWidth;

	int oRowTop = (int)(sourceRelativeRow * oHeight);
	int oRowBot = (int)(sourceRelativeRow * oHeight) + 1;
	int oColLeft = (int)(sourceRelativeCol * oWidth);
	int oColRight = (int)(sourceRelativeCol * oWidth) + 1;

	// Bilinear calculation
	unsigned char topLeft[3];
	unsigned char topRight[3];
	unsigned char botLeft[3];
	unsigned char botRight[3];
	int oColLeftSample = oColLeft;
	int oColRightSample = oColRight;
	int oRowTopSample = oRowTop;
	int oRowBotSample = oRowBot;

	if (oColLeft < 0)
	{
		oColLeftSample = 0;
	}
	if (oColRight >= oWidth)
	{
		oColRightSample = oWidth - 1;
	}
	if (oRowTop < 0)
	{
		oRowTopSample = 0;
	}
	if (oRowBot >= oHeight)
	{
		oRowBotSample = oHeight - 1;
	}
	int oIndexTL = ((oColLeftSample + oRowTopSample * oWidth) * 3) + oRowTopSample * oPad;
	int oIndexTR = ((oColRightSample + oRowTopSample * oWidth) * 3) + oRowTopSample * oPad;
	int oIndexBL = ((oColLeftSample + oRowBotSample * oWidth) * 3) + oRowBotSample * oPad;
	int oIndexBR = ((oColRightSample + oRowBotSample * oWidth) * 3) + oRowBotSample * oPad;
	unsigned char TL[3];
	unsigned char TR[3];
	unsigned char BL[3];
	unsigned char BR[3];

	float x = sourceRelativeCol * oWidth;
	float y = sourceRelativeRow * oHeight;
	float leftLinearFactor = oColRight - x;
	float rightLinearFactor = x - oColLeft;
	float topLinearFactor = oRowBot - y;
	float botLinearFactor = y - oRowTop;

	for (int i = 0; i < 3; ++i)
	{
		TL[i] = source[oIndexTL + i];
		TR[i] = source[oIndexTR + i];
		BL[i] = source[oIndexBL + i];
		BR[i] = source[oIndexBR + i];

		float top = leftLinearFactor * TL[i] + rightLinearFactor * TR[i];
		float bot = leftLinearFactor * BL[i] + rightLinearFactor * BR[i];
		float result = topLinearFactor * top + botLinearFactor * bot;
		dest[index + i] = static_cast<unsigned char>(result);
	}
}

/// <summary>
/// Bicubic CUDA kernel (Naive)
/// </summary>
/// <param name="source">Source image color channel matrix</param>
/// <param name="oWidth">Original image width</param>
/// <param name="oHeight">Original image height</param>
/// <param name="oPad">Original image padding size</param>
/// <param name="dest">Destination image color matrix</param>
/// <param name="nWidth">New image width</param>
/// <param name="nHeight">New image height</param>
/// <param name="nPad">New image padding size</param>
/// <returns></returns>
__global__ void Bicubic(
	unsigned char* source,
	unsigned short oWidth,
	unsigned short oHeight,
	unsigned char oPad,
	unsigned char* dest,
	unsigned short nWidth,
	unsigned short nHeight,
	unsigned char nPad)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col >= nWidth || row >= nHeight)
	{
		return;
	}
	int index = ((col + row * nWidth) * 3) + row * nPad;

	// Build mapping to 4x4 grid of nearby pixels in source image

	float sourceRelativeRow = (float)row / (float)nHeight;
	float sourceRelativeCol = (float)col / (float)nWidth;

	float oY = sourceRelativeRow * oHeight;
	float oX = sourceRelativeCol * oWidth;

	int oRow = (int)oY;
	int oCol = (int)oX;

	// Populate indices of colors to sample for 16 points
	unsigned int neighborhoodIndices[4][4];
	for (int x = 0; x < 4; ++x)
	{
		for (int y = 0; y < 4; ++y)
		{
			int oCurrentCol = oCol - 1 + x;
			if (oCurrentCol < 0)
			{
				oCurrentCol = 0;
			}
			if (oCurrentCol >= oWidth)
			{
				oCurrentCol = oWidth - 1;
			}
			int oCurrentRow = oRow - 1 + y;
			if (oCurrentRow < 0)
			{
				oCurrentRow = 0;
			}
			if (oCurrentRow >= oHeight)
			{
				oCurrentRow = oHeight - 1;
			}
			int oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
			neighborhoodIndices[x][y] = oIndex;
		}
	}

	// ranges from 0 to 1 representing location in unit box of desired pixel relative to known source information
	float rX = oX - oCol;
	float rY = oY - oRow;

	// Cubic interpolation on the 4 rows (times 3 color channels), each containing 4 points
	float rowCubics[12];
	for (int y = 0; y < 4; ++y)
	{
		// Cubic interpolation on a given row
		for (int c = 0; c < 3; ++c)
		{
			// interpolation per color channel
			unsigned char p0 = source[neighborhoodIndices[0][y] + c];
			unsigned char p1 = source[neighborhoodIndices[1][y] + c];
			unsigned char p2 = source[neighborhoodIndices[2][y] + c];
			unsigned char p3 = source[neighborhoodIndices[3][y] + c];

			// interpolate value
			// calculus
			rowCubics[y * 3 + c] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));
		}
	}

	// Cubic interpolation on the resulting intermediate collumn (times 3 color channels)
	for (int c = 0; c < 3; ++c)
	{
		// interpolation per color channel
		float p0 = rowCubics[c];
		float p1 = rowCubics[3 + c];
		float p2 = rowCubics[6 + c];
		float p3 = rowCubics[9 + c];
		unsigned char result;

		// interpolate value
		// calculus
		float rVal = p1 + 0.5f * rY * (p2 - p0 + rY * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rY * (3.0f * (p1 - p2) + p3 - p0)));

		// Bicubic interpolation can overshoot, so don't just cast to int, also cap to 0-255
		if (rVal <= 0.0f)
		{
			result = 0x00;
		}
		else if (rVal >= 255.0f)
		{
			result = 0xFF;
		}
		else
		{
			result = static_cast<unsigned char>(rVal);
		}
		dest[index + c] = result;
	}
}


/// <summary>
/// Nearest neighbor CUDA kernel (Conditional paths removed)
/// </summary>
/// <param name="source">Source image color channel matrix</param>
/// <param name="oWidth">Original image width</param>
/// <param name="oHeight">Original image height</param>
/// <param name="oPad">Original image padding size</param>
/// <param name="dest">Destination image color matrix</param>
/// <param name="nWidth">New image width</param>
/// <param name="nHeight">New image height</param>
/// <param name="nPad">New image padding size</param>
/// <returns></returns>
__global__ void Bicubic2(
	unsigned char* source,
	unsigned short oWidth,
	unsigned short oHeight,
	unsigned char oPad,
	unsigned char* dest,
	unsigned short nWidth,
	unsigned short nHeight,
	unsigned char nPad)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col >= nWidth || row >= nHeight)
	{
		return;
	}
	int index = ((col + row * nWidth) * 3) + row * nPad;

	// Build mapping to 4x4 grid of nearby pixels in source image

	float sourceRelativeRow = (float)row / (float)nHeight;
	float sourceRelativeCol = (float)col / (float)nWidth;

	float oY = sourceRelativeRow * oHeight;
	float oX = sourceRelativeCol * oWidth;

	int ioWidth = (int)oWidth;
	int ioHeight = (int)oHeight;

	int oRow = (int)oY;
	int oCol = (int)oX;

	// Populate indices of colors to sample for 16 points
	unsigned int neighborhoodIndices[4][4];
	for (int x = 0; x < 4; ++x)
	{
		for (int y = 0; y < 4; ++y)
		{
			int oCurrentCol = oCol - 1 + x;
			oCurrentCol = max(0, oCurrentCol);
			oCurrentCol = min(oCurrentCol, ioWidth - 1);
			int oCurrentRow = oRow - 1 + y;
			oCurrentRow = max(0, oCurrentRow);
			oCurrentRow = min(oCurrentRow, ioHeight - 1);
			int oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
			neighborhoodIndices[x][y] = oIndex;
		}
	}
	// ranges from 0 to 1 representing location in unit box of desired pixel relative to known source information
	float rX = oX - oCol;
	float rY = oY - oRow;

	// Cubic interpolation on the 4 rows (times 3 color channels), each containing 4 points
	float rowCubics[12];
	for (int y = 0; y < 4; ++y)
	{
		// Cubic interpolation on a given row
		for (int c = 0; c < 3; ++c)
		{
			// interpolation per color channel
			unsigned char p0 = source[neighborhoodIndices[0][y] + c];
			unsigned char p1 = source[neighborhoodIndices[1][y] + c];
			unsigned char p2 = source[neighborhoodIndices[2][y] + c];
			unsigned char p3 = source[neighborhoodIndices[3][y] + c];

			// interpolate value
			// calculus
			rowCubics[y * 3 + c] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));
		}
	}

	// Cubic interpolation on the resulting intermediate collumn (times 3 color channels)
	for (int c = 0; c < 3; ++c)
	{
		// interpolation per color channel
		float p0 = rowCubics[c];
		float p1 = rowCubics[3 + c];
		float p2 = rowCubics[6 + c];
		float p3 = rowCubics[9 + c];

		// interpolate value
		// calculus
		float rVal = p1 + 0.5f * rY * (p2 - p0 + rY * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rY * (3.0f * (p1 - p2) + p3 - p0)));

		// Bicubic interpolation can overshoot, so don't just cast to int, also cap to 0-255
		unsigned char result;
		rVal = fminf(255.0f, rVal);
		rVal = fmaxf(0.0f, rVal);
		result = static_cast<unsigned char>(rVal);
		dest[index + c] = result;
	}
}

/// <summary>
/// Nearest neighbor CUDA kernel (Conditional paths removed and loops unrolled)
/// </summary>
/// <param name="source">Source image color channel matrix</param>
/// <param name="oWidth">Original image width</param>
/// <param name="oHeight">Original image height</param>
/// <param name="oPad">Original image padding size</param>
/// <param name="dest">Destination image color matrix</param>
/// <param name="nWidth">New image width</param>
/// <param name="nHeight">New image height</param>
/// <param name="nPad">New image padding size</param>
/// <returns></returns>
__global__ void Bicubic3(
	unsigned char* source,
	unsigned short oWidth,
	unsigned short oHeight,
	unsigned char oPad,
	unsigned char* dest,
	unsigned short nWidth,
	unsigned short nHeight,
	unsigned char nPad)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col >= nWidth || row >= nHeight)
	{
		return;
	}
	int index = ((col + row * nWidth) * 3) + row * nPad;

	// Build mapping to 4x4 grid of nearby pixels in source image

	float sourceRelativeRow = (float)row / (float)nHeight;
	float sourceRelativeCol = (float)col / (float)nWidth;

	float oY = sourceRelativeRow * oHeight;
	float oX = sourceRelativeCol * oWidth;

	int ioWidth = (int)oWidth;
	int ioHeight = (int)oHeight;

	int oRow = (int)oY;
	int oCol = (int)oX;

	// Populate indices of colors to sample for 16 points
	unsigned int neighborhoodIndices[4][4];

	int oCurrentCol;
	int oCurrentRow;
	int oIndex;

	// 0,0

	oCurrentCol = oCol - 1;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow - 1;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[0][0] = oIndex;

	// 0,1

	oCurrentCol = oCol - 1;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[0][1] = oIndex;

	// 0,2

	oCurrentCol = oCol - 1;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow + 1;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[0][2] = oIndex;

	// 0,3

	oCurrentCol = oCol - 1;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow + 2;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[0][3] = oIndex;


	// 1,0

	oCurrentCol = oCol;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow - 1;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[1][0] = oIndex;

	// 1,1

	oCurrentCol = oCol;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[1][1] = oIndex;

	// 1,2

	oCurrentCol = oCol;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow + 1;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[1][2] = oIndex;

	// 1,3

	oCurrentCol = oCol;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow + 2;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[1][3] = oIndex;


	// 2,0

	oCurrentCol = oCol + 1;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow - 1;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[2][0] = oIndex;

	// 2,1

	oCurrentCol = oCol + 1;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[2][1] = oIndex;

	// 2,2

	oCurrentCol = oCol + 1;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow + 1;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[2][2] = oIndex;

	// 2,3

	oCurrentCol = oCol + 1;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow + 2;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[2][3] = oIndex;


	// 3,0

	oCurrentCol = oCol + 2;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow - 1;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[3][0] = oIndex;

	// 3,1

	oCurrentCol = oCol + 2;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[3][1] = oIndex;

	// 3,2

	oCurrentCol = oCol + 2;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow + 1;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[3][2] = oIndex;

	// 3,3

	oCurrentCol = oCol + 2;
	oCurrentCol = max(0, oCurrentCol);
	oCurrentCol = min(oCurrentCol, ioWidth - 1);
	oCurrentRow = oRow + 2;
	oCurrentRow = max(0, oCurrentRow);
	oCurrentRow = min(oCurrentRow, ioHeight - 1);
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[3][3] = oIndex;

	// ranges from 0 to 1 representing location in unit box of desired pixel relative to known source information
	float rX = oX - oCol;
	float rY = oY - oRow;

	// Cubic interpolation on the 4 rows (times 3 color channels), each containing 4 points
	float rowCubics[12];

	// horizantal cubics
	unsigned char p0;
	unsigned char p1;
	unsigned char p2;
	unsigned char p3;

	// y=0

	p0 = source[neighborhoodIndices[0][0]];
	p1 = source[neighborhoodIndices[1][0]];
	p2 = source[neighborhoodIndices[2][0]];
	p3 = source[neighborhoodIndices[3][0]];

	rowCubics[0] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));

	p0 = source[neighborhoodIndices[0][0] + 1];
	p1 = source[neighborhoodIndices[1][0] + 1];
	p2 = source[neighborhoodIndices[2][0] + 1];
	p3 = source[neighborhoodIndices[3][0] + 1];

	rowCubics[1] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));

	p0 = source[neighborhoodIndices[0][0] + 2];
	p1 = source[neighborhoodIndices[1][0] + 2];
	p2 = source[neighborhoodIndices[2][0] + 2];
	p3 = source[neighborhoodIndices[3][0] + 2];

	rowCubics[2] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));

	// y=1

	p0 = source[neighborhoodIndices[0][1]];
	p1 = source[neighborhoodIndices[1][1]];
	p2 = source[neighborhoodIndices[2][1]];
	p3 = source[neighborhoodIndices[3][1]];

	rowCubics[3] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));

	p0 = source[neighborhoodIndices[0][1] + 1];
	p1 = source[neighborhoodIndices[1][1] + 1];
	p2 = source[neighborhoodIndices[2][1] + 1];
	p3 = source[neighborhoodIndices[3][1] + 1];

	rowCubics[4] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));

	p0 = source[neighborhoodIndices[0][1] + 2];
	p1 = source[neighborhoodIndices[1][1] + 2];
	p2 = source[neighborhoodIndices[2][1] + 2];
	p3 = source[neighborhoodIndices[3][1] + 2];

	rowCubics[5] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));

	// y=2

	p0 = source[neighborhoodIndices[0][2]];
	p1 = source[neighborhoodIndices[1][2]];
	p2 = source[neighborhoodIndices[2][2]];
	p3 = source[neighborhoodIndices[3][2]];

	rowCubics[6] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));

	p0 = source[neighborhoodIndices[0][2] + 1];
	p1 = source[neighborhoodIndices[1][2] + 1];
	p2 = source[neighborhoodIndices[2][2] + 1];
	p3 = source[neighborhoodIndices[3][2] + 1];

	rowCubics[7] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));

	p0 = source[neighborhoodIndices[0][2] + 2];
	p1 = source[neighborhoodIndices[1][2] + 2];
	p2 = source[neighborhoodIndices[2][2] + 2];
	p3 = source[neighborhoodIndices[3][2] + 2];

	rowCubics[8] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));

	// y = 3

	p0 = source[neighborhoodIndices[0][3]];
	p1 = source[neighborhoodIndices[1][3]];
	p2 = source[neighborhoodIndices[2][3]];
	p3 = source[neighborhoodIndices[3][3]];

	rowCubics[9] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));

	p0 = source[neighborhoodIndices[0][3] + 1];
	p1 = source[neighborhoodIndices[1][3] + 1];
	p2 = source[neighborhoodIndices[2][3] + 1];
	p3 = source[neighborhoodIndices[3][3] + 1];

	rowCubics[10] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));

	p0 = source[neighborhoodIndices[0][3] + 2];
	p1 = source[neighborhoodIndices[1][3] + 2];
	p2 = source[neighborhoodIndices[2][3] + 2];
	p3 = source[neighborhoodIndices[3][3] + 2];

	rowCubics[11] = p1 + 0.5f * rX * (p2 - p0 + rX * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rX * (3.0f * (p1 - p2) + p3 - p0)));

	// vertical cubic
	float p0f = rowCubics[0];
	float p1f = rowCubics[3];
	float p2f = rowCubics[6];
	float p3f = rowCubics[9];

	float rVal;
	unsigned char result;

	rVal = p1f + 0.5f * rY * (p2f - p0f + rY * (2.0f * p0f - 5.0f * p1f + 4.0f * p2f - p3f + rY * (3.0f * (p1f - p2f) + p3f - p0f)));

	rVal = fminf(255.0f, rVal);
	rVal = fmaxf(0.0f, rVal);
	result = static_cast<unsigned char>(rVal);
	dest[index] = result;

	p0f = rowCubics[1];
	p1f = rowCubics[4];
	p2f = rowCubics[7];
	p3f = rowCubics[10];

	rVal = p1f + 0.5f * rY * (p2f - p0f + rY * (2.0f * p0f - 5.0f * p1f + 4.0f * p2f - p3f + rY * (3.0f * (p1f - p2f) + p3f - p0f)));

	rVal = fminf(255.0f, rVal);
	rVal = fmaxf(0.0f, rVal);
	result = static_cast<unsigned char>(rVal);
	dest[index + 1] = result;

	p0f = rowCubics[2];
	p1f = rowCubics[5];
	p2f = rowCubics[8];
	p3f = rowCubics[11];

	rVal = p1f + 0.5f * rY * (p2f - p0f + rY * (2.0f * p0f - 5.0f * p1f + 4.0f * p2f - p3f + rY * (3.0f * (p1f - p2f) + p3f - p0f)));

	rVal = fminf(255.0f, rVal);
	rVal = fmaxf(0.0f, rVal);
	result = static_cast<unsigned char>(rVal);
	dest[index + 2] = result;
}

/// <summary>
/// Nearest neighbor host setup code
/// </summary>
/// <param name="source">Source bitmap</param>
/// <param name="dest">Destination bitmap</param>
void NearestNeighbor(Bitmap* source, Bitmap* dest)
{
	const int NearestNeighborBlockSize = 16;
	dest->init();

	unsigned short oW = source->width;
	unsigned short oH = source->height;
	unsigned char oP = source->padSize();
	unsigned short nW = dest->width;
	unsigned short nH = dest->height;
	unsigned char nP = dest->padSize();

	unsigned char* original_image, * upscaled_image;
	unsigned char* original_image_device, * upscaled_image_device;

	int size_matrix = source->imageDataSize();
	int size_dest = dest->imageDataSize();
	original_image = source->imageData;
	upscaled_image = dest->imageData;

	cudaMalloc((void**)&original_image_device, size_matrix);
	cudaMalloc((void**)&upscaled_image_device, size_dest);

	cudaMemcpy(original_image_device, original_image, size_matrix, cudaMemcpyHostToDevice);

	dim3 dimBlock(NearestNeighborBlockSize, NearestNeighborBlockSize);
	dim3 dimGrid((nW / dimBlock.x) + 1, (nH / dimBlock.y) + 1);

	NearestNeighbor<<<dimGrid, dimBlock>>>(original_image_device, oW, oH, oP, upscaled_image_device, nW, nH, nP);

	cudaMemcpy(upscaled_image, upscaled_image_device, size_dest, cudaMemcpyDeviceToHost);

	cudaFree(original_image_device);
	cudaFree(upscaled_image_device);
}

/// <summary>
/// Bilinear host setup code
/// </summary>
/// <param name="source">Source bitmap</param>
/// <param name="dest">Destination bitmap</param>
void Bilinear(Bitmap* source, Bitmap* dest)
{
	const int BilinearBlockSize = 16;
	dest->init();

	unsigned short oW = source->width;
	unsigned short oH = source->height;
	unsigned char oP = source->padSize();
	unsigned short nW = dest->width;
	unsigned short nH = dest->height;
	unsigned char nP = dest->padSize();

	unsigned char* original_image, * upscaled_image;
	unsigned char* original_image_device, * upscaled_image_device;

	int size_matrix = source->imageDataSize();
	int size_dest = dest->imageDataSize();
	original_image = source->imageData;
	upscaled_image = dest->imageData;

	cudaMalloc((void**)&original_image_device, size_matrix);
	cudaMalloc((void**)&upscaled_image_device, size_dest);

	cudaMemcpy(original_image_device, original_image, size_matrix, cudaMemcpyHostToDevice);

	dim3 dimBlock(BilinearBlockSize, BilinearBlockSize);
	dim3 dimGrid((nW / dimBlock.x) + 1, (nH / dimBlock.y) + 1);

	Bilinear<<<dimGrid, dimBlock>>>(original_image_device, oW, oH, oP, upscaled_image_device, nW, nH, nP);

	cudaMemcpy(upscaled_image, upscaled_image_device, size_dest, cudaMemcpyDeviceToHost);

	cudaFree(original_image_device);
	cudaFree(upscaled_image_device);
}

/// <summary>
/// Bicubic (Naive) host setup code
/// </summary>
/// <param name="source">Source bitmap</param>
/// <param name="dest">Destination bitmap</param>
void Bicubic(Bitmap* source, Bitmap* dest)
{
	const int BicubicBlockSize = 16;
	dest->init();

	unsigned short oW = source->width;
	unsigned short oH = source->height;
	unsigned char oP = source->padSize();
	unsigned short nW = dest->width;
	unsigned short nH = dest->height;
	unsigned char nP = dest->padSize();

	unsigned char* original_image, * upscaled_image;
	unsigned char* original_image_device, * upscaled_image_device;

	int size_matrix = source->imageDataSize();
	int size_dest = dest->imageDataSize();
	original_image = source->imageData;
	upscaled_image = dest->imageData;

	cudaMalloc((void**)&original_image_device, size_matrix);
	cudaMalloc((void**)&upscaled_image_device, size_dest);

	cudaMemcpy(original_image_device, original_image, size_matrix, cudaMemcpyHostToDevice);

	dim3 dimBlock(BicubicBlockSize, BicubicBlockSize);
	dim3 dimGrid((nW / dimBlock.x) + 1, (nH / dimBlock.y) + 1);

	Bicubic<<<dimGrid, dimBlock>>>(original_image_device, oW, oH, oP, upscaled_image_device, nW, nH, nP);

	cudaMemcpy(upscaled_image, upscaled_image_device, size_dest, cudaMemcpyDeviceToHost);

	cudaFree(original_image_device);
	cudaFree(upscaled_image_device);
}

/// <summary>
/// Bicubic (conditional removal) host setup code
/// </summary>
/// <param name="source">Source bitmap</param>
/// <param name="dest">Destination bitmap</param>
void Bicubic2(Bitmap* source, Bitmap* dest)
{
	const int BicubicBlockSize = 16;
	dest->init();

	unsigned short oW = source->width;
	unsigned short oH = source->height;
	unsigned char oP = source->padSize();
	unsigned short nW = dest->width;
	unsigned short nH = dest->height;
	unsigned char nP = dest->padSize();

	unsigned char* original_image, * upscaled_image;
	unsigned char* original_image_device, * upscaled_image_device;

	int size_matrix = source->imageDataSize();
	int size_dest = dest->imageDataSize();
	original_image = source->imageData;
	upscaled_image = dest->imageData;

	cudaMalloc((void**)&original_image_device, size_matrix);
	cudaMalloc((void**)&upscaled_image_device, size_dest);

	cudaMemcpy(original_image_device, original_image, size_matrix, cudaMemcpyHostToDevice);

	dim3 dimBlock(BicubicBlockSize, BicubicBlockSize);
	dim3 dimGrid((nW / dimBlock.x) + 1, (nH / dimBlock.y) + 1);

	Bicubic2<<<dimGrid, dimBlock>>>(original_image_device, oW, oH, oP, upscaled_image_device, nW, nH, nP);

	cudaMemcpy(upscaled_image, upscaled_image_device, size_dest, cudaMemcpyDeviceToHost);

	cudaFree(original_image_device);
	cudaFree(upscaled_image_device);
}

/// <summary>
/// Bicubic (conditional removal + unrolling) host setup code
/// </summary>
/// <param name="source">Source bitmap</param>
/// <param name="dest">Destination bitmap</param>
void Bicubic3(Bitmap* source, Bitmap* dest)
{
	const int BicubicBlockSize = 16;
	dest->init();

	unsigned short oW = source->width;
	unsigned short oH = source->height;
	unsigned char oP = source->padSize();
	unsigned short nW = dest->width;
	unsigned short nH = dest->height;
	unsigned char nP = dest->padSize();

	unsigned char* original_image, * upscaled_image;
	unsigned char* original_image_device, * upscaled_image_device;

	int size_matrix = source->imageDataSize();
	int size_dest = dest->imageDataSize();
	original_image = source->imageData;
	upscaled_image = dest->imageData;

	cudaMalloc((void**)&original_image_device, size_matrix);
	cudaMalloc((void**)&upscaled_image_device, size_dest);

	cudaMemcpy(original_image_device, original_image, size_matrix, cudaMemcpyHostToDevice);

	dim3 dimBlock(BicubicBlockSize, BicubicBlockSize);
	dim3 dimGrid((nW / dimBlock.x) + 1, (nH / dimBlock.y) + 1);

	Bicubic3 << <dimGrid, dimBlock >> > (original_image_device, oW, oH, oP, upscaled_image_device, nW, nH, nP);

	cudaMemcpy(upscaled_image, upscaled_image_device, size_dest, cudaMemcpyDeviceToHost);

	cudaFree(original_image_device);
	cudaFree(upscaled_image_device);
}

/// <summary>
/// Run main pipeline
/// </summary>
/// <returns></returns>
int main()
{
	Bitmap* nearestNeighborImageRaytracer = new Bitmap();
	nearestNeighborImageRaytracer->readFromFile("TestContent/raytracer.bmp");
	Bitmap* nearestNeighborImageRaytracerUpscale = new Bitmap();
	nearestNeighborImageRaytracerUpscale->width = 5000;
	nearestNeighborImageRaytracerUpscale->height = 5000;
	NearestNeighbor(nearestNeighborImageRaytracer, nearestNeighborImageRaytracerUpscale);
	nearestNeighborImageRaytracerUpscale->writeToFile("TestContent/raytracer_newarest_neighbor.bmp");

	Bitmap* bilinearImageRaytracer = new Bitmap();
	bilinearImageRaytracer->readFromFile("TestContent/raytracer.bmp");
	Bitmap* bilinearImageRaytracerUpscale = new Bitmap();
	bilinearImageRaytracerUpscale->width = 5000;
	bilinearImageRaytracerUpscale->height = 5000;
	Bilinear(bilinearImageRaytracer, bilinearImageRaytracerUpscale);
	bilinearImageRaytracerUpscale->writeToFile("TestContent/raytracer_bilinear.bmp");

	Bitmap* bicubicImageRaytracer = new Bitmap();
	bicubicImageRaytracer->readFromFile("TestContent/raytracer.bmp");
	Bitmap* bicubicImageRaytracerUpscale = new Bitmap();
	bicubicImageRaytracerUpscale->width = 5000;
	bicubicImageRaytracerUpscale->height = 5000;
	Bicubic(bicubicImageRaytracer, bicubicImageRaytracerUpscale);
	bicubicImageRaytracerUpscale->writeToFile("TestContent/raytracer_bicubic.bmp");

	Bitmap* bicubicImageRaytracer2 = new Bitmap();
	bicubicImageRaytracer2->readFromFile("TestContent/raytracer.bmp");
	Bitmap* bicubicImageRaytracerUpscale2 = new Bitmap();
	bicubicImageRaytracerUpscale2->width = 5000;
	bicubicImageRaytracerUpscale2->height = 5000;
	Bicubic2(bicubicImageRaytracer2, bicubicImageRaytracerUpscale2);
	bicubicImageRaytracerUpscale2->writeToFile("TestContent/raytracer_bicubic2.bmp");

	Bitmap* bicubicImageRaytracer3 = new Bitmap();
	bicubicImageRaytracer3->readFromFile("TestContent/raytracer.bmp");
	Bitmap* bicubicImageRaytracerUpscale3 = new Bitmap();
	bicubicImageRaytracerUpscale3->width = 5000;
	bicubicImageRaytracerUpscale3->height = 5000;
	Bicubic3(bicubicImageRaytracer3, bicubicImageRaytracerUpscale3);
	bicubicImageRaytracerUpscale3->writeToFile("TestContent/raytracer_bicubic3.bmp");

	Bitmap* bicubicImageRaytracer4 = new Bitmap();
	bicubicImageRaytracer4->readFromFile("TestContent/raytracer.bmp");
	Bitmap* bicubicImageRaytracerUpscale4 = new Bitmap();
	bicubicImageRaytracerUpscale4->width = 5000;
	bicubicImageRaytracerUpscale4->height = 5000;
	chrono::steady_clock::time_point start = chrono::high_resolution_clock::now();
	DebugFeatures::emulator(bicubicImageRaytracer4, bicubicImageRaytracerUpscale4);
	chrono::steady_clock::time_point finish = chrono::high_resolution_clock::now();
	std::chrono::duration<long long,std::nano> elapsed = finish - start;
	std::chrono::milliseconds mili = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
	cout << "CPU emulation time: " << mili.count() << " miliseconds" << std::endl;
	bicubicImageRaytracerUpscale4->writeToFile("TestContent/raytracer_bicubicCPU.bmp");

	return 0;
}
