// For Visual Studio intelisense, mainly
#if _MSC_VER
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include <iostream>

#include "bitmap.h"
#include "debugFeatures.h"

using namespace std;

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

	int oCol = (int)(((float)col / (float)nWidth) * oWidth + 0.5f);
	int oRow = (int)(((float)row / (float)nHeight) * oHeight + 0.5f);

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
/*
#define BLOCK_SIZE 2
__global__ void CopyImage(unsigned char* a, unsigned char* b, unsigned short width, unsigned short height, int pad) {

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	int index = ((col + row * width) * 3) + row * pad;

	if (row < height && col < width) {
		b[index] = a[index];
        b[index + 1] = a[index + 1];
        b[index + 2] = a[index + 2];
	}
}
*/

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

int main()
{
    Bitmap* baseImage = new Bitmap();
	Bitmap* debugImage = new Bitmap();
	Bitmap* nearestNeighborImage = new Bitmap();
	Bitmap* bilinearImage = new Bitmap();
	debugImage->width = 2005;
	debugImage->height = 2005;
	nearestNeighborImage->width = 295;
	nearestNeighborImage->height = 295;
	bilinearImage->width = 2005;
	bilinearImage->height = 2005;
	baseImage->readFromFile("TestContent/Test1.bmp");
	DebugFeatures::stopX = 990;
	DebugFeatures::stopY = 10;
	DebugFeatures::emulator(baseImage, debugImage);
	NearestNeighbor(baseImage, nearestNeighborImage);
	Bilinear(baseImage, bilinearImage);
	debugImage->writeToFile("TestContent/Debug.bmp");
	nearestNeighborImage->writeToFile("TestContent/Test1NearestNeighbor.bmp");
	bilinearImage->writeToFile("TestContent/Test1Bilinear.bmp");

	return 0;
}
