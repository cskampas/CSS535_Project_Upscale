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
unsigned short DebugFeatures::stopX;
unsigned short DebugFeatures::stopY;

void CPUKernelDebug(
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
	}
}

void KernelCPUEmulator(unsigned char* source, unsigned short oWidth, unsigned short oHeight, unsigned char oPad, unsigned char* dest, unsigned short nWidth, unsigned short nHeight, unsigned char nPad)
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

	// Pretending to cudaMalloc and cudaMemcpy
	original_image_device = original_image;
	upscaled_image_device = upscaled_image;

	// Mocked threads per block and blocks per grid.  Modify if needed
	blockDim = mockUnit3(BlockSize, BlockSize);
	gridDim = mockUnit3((nW / blockDim.x) + 1, (nH / blockDim.y) + 1);

	KernelCPUEmulator(original_image_device, oW, oH, oP, upscaled_image_device, nW, nH, nP);
}
