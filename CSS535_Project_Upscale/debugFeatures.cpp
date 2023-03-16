/*
* How to use:
* Copy and paste with absolutely no changes the GPU kernel code into CPUKernelDebug()
* If the host code which launches it needs any modification, adjust emulator()
* All other functions should be left unmodified
* 
* A debugger breakpoint can be inserted at the indicated place in KernelCPUEmulator() to stop
* at the x,y pixel given by stopX,stopY, initalized prior to execution
*/
#include "debugFeatures.h"

#include <iostream>
#include <cmath>

using namespace std;

struct mockUnit3
{
	unsigned int x;
	unsigned int y;
	unsigned int z;
	mockUnit3(const unsigned int X = 1, const unsigned int Y = 1, const unsigned int Z = 1)
	{
		x = X;
		y = Y;
		z = Z;
	}
};

mockUnit3 threadIdx;
mockUnit3 blockIdx;
mockUnit3 blockDim;
mockUnit3 gridDim;
unsigned int DebugFeatures::stopX;
unsigned int DebugFeatures::stopY;

void CPUKernelDebug(
	unsigned char* source,
	unsigned int oWidth,
	unsigned int oHeight,
	unsigned char oPad,
	unsigned char* dest,
	unsigned int nWidth,
	unsigned int nHeight,
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
	/*
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

	int oRowTop = (int)(sourceRelativeRow * oHeight - 0.5f);
	int oRowBot = (int)(sourceRelativeRow * oHeight + 0.5f);
	int oColLeft = (int)(sourceRelativeCol * oWidth - 0.5f);
	int oColRight = (int)(sourceRelativeCol * oWidth + 0.5f);

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
	float leftLinearFactor = (oColRightSample + 0.5f) - x;
	float rightLinearFactor = 1 - leftLinearFactor;
	float topLinearFactor = (oRowBotSample + 0.5f) - y;
	float botLinearFactor = 1 - topLinearFactor;

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
	}*/
}

void KernelCPUEmulator(unsigned char* source, unsigned int oWidth, unsigned int oHeight, unsigned char oPad, unsigned char* dest, unsigned int nWidth, unsigned int nHeight, unsigned char nPad)
{
	for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x)
	{
		for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y)
		{
			for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z)
			{
				for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x)
				{
					for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y)
					{
						for (threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z)
						{
							// Conditional breakpoints in Visual Studio introduce significant performance impact.  It is better here to do this:
							// Some copy-paste required for row/col calc to provide correct stopping pixel
							int col = threadIdx.x + blockIdx.x * blockDim.x;
							int row = threadIdx.y + blockIdx.y * blockDim.y;

							if (col == DebugFeatures::stopX && row == DebugFeatures::stopY)
							{
								// Set a breakpoint at this line to debug a particular pixel
								cout << "";
							}

							CPUKernelDebug(source, oWidth, oHeight, oPad, dest, nWidth, nHeight, nPad);
						}
					}
				}
			}
		}
	}
}

void DebugFeatures::emulator(Bitmap* source, Bitmap* dest)
{
	const int BlockSize = 32;
	dest->init();

	unsigned int oW = source->width;
	unsigned int oH = source->height;
	unsigned char oP = source->padSize();
	unsigned int nW = dest->width;
	unsigned int nH = dest->height;
	unsigned char nP = dest->padSize();

	unsigned char* original_image, * upscaled_image;
	unsigned char* original_image_device, * upscaled_image_device;

	int size_matrix = source->imageDataSize();
	int size_dest = dest->imageDataSize();
	original_image = source->imageData;
	upscaled_image = dest->imageData;

	// Pretending to cudaMalloc and cudaMemcpy
	original_image_device = original_image;
	upscaled_image_device = upscaled_image;

	// Mocked threads per block and blocks per grid.  Modify if needed
	blockDim = mockUnit3(BlockSize, BlockSize);
	gridDim = mockUnit3((nW / blockDim.x) + 1, (nH / blockDim.y) + 1);

	KernelCPUEmulator(original_image_device, oW, oH, oP, upscaled_image_device, nW, nH, nP);
}
