// For Visual Studio intelisense, mainly
#if _MSC_VER
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include <iostream>
// #include "math.h"
// #include <cmath>

#include "bitmap.h"
#include "debugFeatures.h"

// using namespace std;

// __device__ float (*device_fminf)(float, float) = fminf;
// __device__ float (*device_fmaxf)(float, float) = fmaxf;

__forceinline__ __device__ float my_fminf(float a, float b)
{
	return (a < b) * a + (b <= a) * b;
	// return device_fminf(a, b);
}
__forceinline__ __device__ float my_fmaxf(float a, float b)
{
	return (a > b) * a + (b >= a) * b;
	// return fmaxf(a, b);
}

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
			oCurrentCol += -(oCurrentCol >> 16);
			oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
			int oCurrentRow = oRow - 1 + y;
			oCurrentRow += -(oCurrentRow >> 16);
			oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
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
		rVal = my_fminf(255.0f, rVal);
		rVal = my_fmaxf(0.0f, rVal);
		result = static_cast<unsigned char>(rVal);
		dest[index + c] = result;
	}
}

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
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow - 1;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[0][0] = oIndex;

	// 0,1

	oCurrentCol = oCol - 1;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[0][1] = oIndex;

	// 0,2

	oCurrentCol = oCol - 1;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow + 1;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[0][2] = oIndex;

	// 0,3

	oCurrentCol = oCol - 1;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow + 2;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[0][3] = oIndex;


	// 1,0

	oCurrentCol = oCol;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow - 1;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[1][0] = oIndex;

	// 1,1

	oCurrentCol = oCol;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[1][1] = oIndex;

	// 1,2

	oCurrentCol = oCol;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow + 1;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[1][2] = oIndex;

	// 1,3

	oCurrentCol = oCol;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow + 2;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[1][3] = oIndex;


	// 2,0

	oCurrentCol = oCol + 1;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow - 1;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[2][0] = oIndex;

	// 2,1

	oCurrentCol = oCol + 1;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[2][1] = oIndex;

	// 2,2

	oCurrentCol = oCol + 1;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow + 1;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[2][2] = oIndex;

	// 2,3

	oCurrentCol = oCol + 1;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow + 2;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[2][3] = oIndex;


	// 3,0

	oCurrentCol = oCol + 2;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow - 1;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[3][0] = oIndex;

	// 3,1

	oCurrentCol = oCol + 2;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[3][1] = oIndex;

	// 3,2

	oCurrentCol = oCol + 2;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow + 1;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
	oIndex = ((oCurrentCol + oCurrentRow * oWidth) * 3) + oCurrentRow * oPad;
	neighborhoodIndices[3][2] = oIndex;

	// 3,3

	oCurrentCol = oCol + 2;
	oCurrentCol += -(oCurrentCol >> 16);
	oCurrentCol += (((ioWidth - 1) - oCurrentCol) >> 16) << -((ioWidth - oCurrentCol));
	oCurrentRow = oRow + 2;
	oCurrentRow += -(oCurrentRow >> 16);
	oCurrentRow += (((ioHeight - 1) - oCurrentRow) >> 16) << -((ioHeight - oCurrentRow));
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
	p0 = rowCubics[0];
	p1 = rowCubics[3];
	p2 = rowCubics[6];
	p3 = rowCubics[9];
	unsigned char result;

	float rVal = p1 + 0.5f * rY * (p2 - p0 + rY * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rY * (3.0f * (p1 - p2) + p3 - p0)));

	rVal = my_fminf(255.0f, rVal);
	rVal = my_fmaxf(0.0f, rVal);
	result = static_cast<unsigned char>(rVal);
	dest[index] = result;

	p0 = rowCubics[1];
	p1 = rowCubics[4];
	p2 = rowCubics[7];
	p3 = rowCubics[10];

	rVal = p1 + 0.5f * rY * (p2 - p0 + rY * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rY * (3.0f * (p1 - p2) + p3 - p0)));

	rVal = my_fminf(255.0f, rVal);
	rVal = my_fmaxf(0.0f, rVal);
	result = static_cast<unsigned char>(rVal);
	dest[index + 1] = result;

	p0 = rowCubics[2];
	p1 = rowCubics[5];
	p2 = rowCubics[8];
	p3 = rowCubics[11];

	rVal = p1 + 0.5f * rY * (p2 - p0 + rY * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + rY * (3.0f * (p1 - p2) + p3 - p0)));

	rVal = my_fminf(255.0f, rVal);
	rVal = my_fmaxf(0.0f, rVal);
	result = static_cast<unsigned char>(rVal);
	dest[index + 2] = result;
}

__global__ void Bicubic4(
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
			// use shared memory for these
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


void NearestNeighbor(Bitmap* source, Bitmap* dest)
{
	const int NearestNeighborBlockSize = 32;
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

	NearestNeighbor <<<dimGrid, dimBlock>>>(original_image_device, oW, oH, oP, upscaled_image_device, nW, nH, nP);

	cudaMemcpy(upscaled_image, upscaled_image_device, size_dest, cudaMemcpyDeviceToHost);

	cudaFree(original_image_device);
	cudaFree(upscaled_image_device);
}

void Bilinear(Bitmap* source, Bitmap* dest)
{
	const int BilinearBlockSize = 32;
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


void Bicubic(Bitmap* source, Bitmap* dest)
{
	const int BicubicBlockSize = 32;
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


void Bicubic2(Bitmap* source, Bitmap* dest)
{
	const int BicubicBlockSize = 32;
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


void Bicubic3(Bitmap* source, Bitmap* dest)
{
	const int BicubicBlockSize = 32;
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


int main()
{
	Bitmap* bicubicImageRaytracer = new Bitmap();
	bicubicImageRaytracer->readFromFile("TestContent/raytracer.bmp");
	Bitmap* bicubicImageRaytracerUpscale = new Bitmap();
	bicubicImageRaytracerUpscale->width = 2000;
	bicubicImageRaytracerUpscale->height = 2000;
	Bicubic(bicubicImageRaytracer, bicubicImageRaytracerUpscale);
	bicubicImageRaytracerUpscale->writeToFile("TestContent/raytracer_out.bmp");

	Bitmap* baseImage = new Bitmap();
	Bitmap* debugImage = new Bitmap();
	Bitmap* nearestNeighborImage = new Bitmap();
	Bitmap* bilinearImage = new Bitmap();
	Bitmap* bicubicImage = new Bitmap();
	Bitmap* bicubicImage2 = new Bitmap();
	Bitmap* bicubicImage3 = new Bitmap();
	debugImage->width = 2000;
	debugImage->height = 2000;
	nearestNeighborImage->width = 295;
	nearestNeighborImage->height = 295;
	bilinearImage->width = 2005;
	bilinearImage->height = 2005;
	bicubicImage->width = 2005;
	bicubicImage->height = 2005;
	bicubicImage2->width = 2005;
	bicubicImage2->height = 2005;
	bicubicImage3->width = 2005;
	bicubicImage3->height = 2005;
	baseImage->readFromFile("TestContent/raytracer.bmp");
	DebugFeatures::stopX = 5;
	DebugFeatures::stopY = 50;
	DebugFeatures::emulator(bicubicImageRaytracer, debugImage);
	NearestNeighbor(baseImage, nearestNeighborImage);
	Bilinear(baseImage, bilinearImage);
	Bicubic(baseImage, bicubicImage);
	Bicubic2(baseImage, bicubicImage2);
	Bicubic3(baseImage, bicubicImage3);
	debugImage->writeToFile("TestContent/Debug.bmp");
	nearestNeighborImage->writeToFile("TestContent/Test1NearestNeighbor.bmp");
	bilinearImage->writeToFile("TestContent/Test1Bilinear.bmp");
	bicubicImage->writeToFile("TestContent/TestBicubic.bmp");
	bicubicImage2->writeToFile("TestContent/TestBicubic2.bmp");
	bicubicImage3->writeToFile("TestContent/TestBicubic3.bmp");

	return 0;
}
