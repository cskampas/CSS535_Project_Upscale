#if _MSC_VER
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include "bitmap.h"

#include <iostream>

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

	// int oCol = (int)(sourceRelativeCol * oWidth + 0.5f);
	// int oRow = (int)(sourceRelativeRow * oHeight + 0.5f);
	// int oIndex = ((oCol + oRow * oWidth) * 3) + oRow * oPad;

	int oRowTop = (int)(sourceRelativeRow * oHeight);
	int oRowBot = (int)(sourceRelativeRow * oHeight) + 1;
	int oColLeft = (int)(sourceRelativeCol * oWidth);
	int oColRight = (int)(sourceRelativeCol * oWidth) + 1;
	/*
	if (oColLeft < 0)
	{
		oColLeft = 0;
	}
	if (oColRight >= oHeight)
	{
		oColRight = oHeight - 1;
	}
	if (oRowTop < 0)
	{
		oRowTop = 0;
	}
	if (oRowBot >= oHeight)
	{
		oRowBot = oHeight - 1;
	}*/

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

	float leftLinearFactor = sourceRelativeCol - oColLeft;
	float rightLinearFactor = oColRight - sourceRelativeCol;
	float topLinearFactor = sourceRelativeRow - oRowTop;
	float botLinearFactor = oRowBot - sourceRelativeRow;

	for (int i = 0; i < 3; ++i)
	{
		TL[i] = source[oIndexTL + i];
		TR[i] = source[oIndexTR + i];
		BL[i] = source[oIndexBL + i];
		BR[i] = source[oIndexBR + i];

		float top = leftLinearFactor * TL[i] + rightLinearFactor * TR[i];
		float bot = leftLinearFactor * BL[i] + rightLinearFactor * BR[i];
		float result = topLinearFactor * top + botLinearFactor * bot;
		dest[index + i] = static_cast<unsigned char>(oRowBot * 128);
	}

	/*
	float bTop = leftLinearFactor * TL[0] + rightLinearFactor * TR[0];
	float gTop = leftLinearFactor * TL[1] + rightLinearFactor * TR[1];
	float rTop = leftLinearFactor * TL[2] + rightLinearFactor * TR[2];

	float bBot = leftLinearFactor * BL[0] + rightLinearFactor * BR[0];
	float gBot = leftLinearFactor * BL[1] + rightLinearFactor * BR[1];
	float rBot = leftLinearFactor * BL[2] + rightLinearFactor * BR[2];

	float b = topLinearFactor * bTop + botLinearFactor * bBot;
	float g = topLinearFactor * gTop + botLinearFactor * gBot;
	float r = topLinearFactor * rTop + botLinearFactor * rBot;


	dest[index] = (unsigned char)bTop;
	dest[index + 1] = (unsigned char)gTop;
	dest[index + 2] = (unsigned char)rTop;
	*/
	/*
	dest[index] = (unsigned char)b;
	dest[index + 1] = (unsigned char)g;
	dest[index + 2] = (unsigned char)r;
	*/
	/*

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
	*/
}

void BilinearCPUKernel(
	unsigned int row,
	unsigned int col,
	unsigned char* source,
	unsigned short oWidth,
	unsigned short oHeight,
	unsigned char oPad,
	unsigned char* dest,
	unsigned short nWidth,
	unsigned short nHeight,
	unsigned char nPad)
{
if (col >= nWidth || row >= nHeight)
{
	return;
}

int index = ((col + row * nWidth) * 3) + row * nPad;

// Find left and right pixel from row above and row below
// "Top" and "Left" here means towards 0, regardless of the reality of the image format

float sourceRelativeRow = (float)row / (float)nHeight;
float sourceRelativeCol = (float)col / (float)nWidth;

// int oCol = (int)(sourceRelativeCol * oWidth + 0.5f);
// int oRow = (int)(sourceRelativeRow * oHeight + 0.5f);
// int oIndex = ((oCol + oRow * oWidth) * 3) + oRow * oPad;

int oRowTop = (int)(sourceRelativeRow * oHeight);
int oRowBot = (int)(sourceRelativeRow * oHeight) + 1;
int oColLeft = (int)(sourceRelativeCol * oWidth);
int oColRight = (int)(sourceRelativeCol * oWidth) + 1;
/*
if (oColLeft < 0)
{
	oColLeft = 0;
}
if (oColRight >= oHeight)
{
	oColRight = oHeight - 1;
}
if (oRowTop < 0)
{
	oRowTop = 0;
}
if (oRowBot >= oHeight)
{
	oRowBot = oHeight - 1;
}*/

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

float leftLinearFactor = sourceRelativeCol - oColLeft;
float rightLinearFactor = oColRight - sourceRelativeCol;
float topLinearFactor = sourceRelativeRow - oRowTop;
float botLinearFactor = oRowBot - sourceRelativeRow;

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

/*
float bTop = leftLinearFactor * TL[0] + rightLinearFactor * TR[0];
float gTop = leftLinearFactor * TL[1] + rightLinearFactor * TR[1];
float rTop = leftLinearFactor * TL[2] + rightLinearFactor * TR[2];

float bBot = leftLinearFactor * BL[0] + rightLinearFactor * BR[0];
float gBot = leftLinearFactor * BL[1] + rightLinearFactor * BR[1];
float rBot = leftLinearFactor * BL[2] + rightLinearFactor * BR[2];

float b = topLinearFactor * bTop + botLinearFactor * bBot;
float g = topLinearFactor * gTop + botLinearFactor * gBot;
float r = topLinearFactor * rTop + botLinearFactor * rBot;


dest[index] = (unsigned char)bTop;
dest[index + 1] = (unsigned char)gTop;
dest[index + 2] = (unsigned char)rTop;
*/
/*
dest[index] = (unsigned char)b;
dest[index + 1] = (unsigned char)g;
dest[index + 2] = (unsigned char)r;
*/
/*

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
*/
}

void BilinearCPU(
	unsigned int blockIdxX,
	unsigned int blockIdxY,
	unsigned int threadIdxX,
	unsigned int threadIdxY,
	unsigned char* source,
	unsigned short oWidth,
	unsigned short oHeight,
	unsigned char oPad,
	unsigned char* dest,
	unsigned short nWidth,
	unsigned short nHeight,
	unsigned char nPad)
{
	for (unsigned int gX = 0; gX < blockIdxX; ++gX)
	{
		for (unsigned int gY = 0; gY < blockIdxY; ++gY)
		{
			for (unsigned int bX = 0; bX < threadIdxX; ++bX)
			{
				for (unsigned int bY = 0; bY < threadIdxY; ++bY)
				{
					int col = bX + gX * 32;
					int row = bY + gY * 32;
					BilinearCPUKernel(row, col, source, oWidth, oHeight, oPad, dest, nWidth, nHeight, nPad);
				}
			}
		}
	}
}

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
	Bitmap* nearestNeighborImage = new Bitmap();
	Bitmap* bilinearImage = new Bitmap();
	nearestNeighborImage->width = 295;
	nearestNeighborImage->height = 295;
	bilinearImage->width = 1005;
	bilinearImage->height = 1005;
	baseImage->readFromFile("TestContent/Test1.bmp");
	NearestNeighbor(baseImage, nearestNeighborImage);
	const int BilinearBlockSize = 32;
	bilinearImage->init();

	unsigned short oW = baseImage->width;
	unsigned short oH = baseImage->height;
	unsigned char oP = baseImage->padSize();
	unsigned short nW = bilinearImage->width;
	unsigned short nH = bilinearImage->height;
	unsigned char nP = bilinearImage->padSize();

	unsigned char* original_image, * upscaled_image;
	unsigned char* original_image_device, * upscaled_image_device;

	int size_matrix = baseImage->imageDataSize();
	int size_dest = bilinearImage->imageDataSize();
	original_image = baseImage->imageData;
	upscaled_image = bilinearImage->imageData;

	//dim3 dimBlock(BilinearBlockSize, BilinearBlockSize);
	//dim3 dimGrid((nW / dimBlock.x) + 1, (nH / dimBlock.y) + 1);
	//uint3 dB = dimBlock;
	//uint3 dG = dimGrid;

	BilinearCPU(32, 32, (nW / 32) + 1, (nH / 32) + 1, original_image, oW, oH, oP, upscaled_image, nW, nH, nP);

	//Bilinear(baseImage, bilinearImage);
	nearestNeighborImage->writeToFile("TestContent/Test1NearestNeighbor.bmp");
	bilinearImage->writeToFile("TestContent/Test1Bilinear.bmp");

	return 0;

	// result->init();

	// int oW = b->width;
	// int oH = b->height;
	// int oP = b->padSize();
	// int nW = result->width;
	// int nH = result->height;
	// int nP = result->padSize();

    // b->writeToFile("TestContent/Test2.bmp");
	
    // unsigned short width = b->width;
	// unsigned short height = b->height;
    
	// unsigned char *original_image, *upscaled_image;
    // unsigned char *original_image_device, *upscaled_image_device;
    
	// int size_matrix = b->imageDataSize();
	// int size_dest = result->imageDataSize();
	// original_image = b->imageData;
	// upscaled_image = result->imageData;

    // upscaled_image = (unsigned char*)malloc(size_matrix);

    // cudaMalloc((void**)&original_image_device, size_matrix);
    // cudaMalloc((void**)&upscaled_image_device, size_dest);

    // cudaMemcpy(original_image_device, original_image, size_matrix, cudaMemcpyHostToDevice);

    //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	//dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
    //CopyImage <<<dimGrid, dimBlock >>>(original_image_device, upscaled_image_device, width, height, b->padSize());

	// dim3 dimBlock(32, 32);
	// dim3 dimGrid((nW / dimBlock.x) + 1, (nH / dimBlock.y) + 1);
	// NearestNeighbor <<<dimGrid, dimBlock >>> (original_image_device, oW, oH, oP, upscaled_image_device, nW, nH, nP);

    // cudaMemcpy(upscaled_image, upscaled_image_device, size_dest, cudaMemcpyDeviceToHost);
	/*
    cout << "ORIGINAL" << endl;
    print_matrix(original_image, width, height, b->padSize());
    cout << "COPY" << endl;
    print_matrix(upscaled_image, width, height, b->padSize());
	*/

	// result->writeToFile("TestContent/Test1NearestNeighbor.bmp");

    // cudaFree(original_image_device);
    // cudaFree(upscaled_image_device);

	// ------------------------

    // return 0;
}

// Kernel test code
/*
unsigned char r = index % 256;
unsigned char g = (index / 256) % 256;
unsigned char b = (index / 65536) % 256;

dest[index] = b;
dest[index + 1] = g;
dest[index + 2] = r;
*/

/*
int oCol = (int)(((float)col / (float)nWidth) * oWidth + 0.5f);
int oRow = (int)(((float)row / (float)nHeight) * oHeight + 0.5f);

int oIndex = ((oCol + oRow * nWidth) * 3) + oRow * oPad;

dest[index] = source[oIndex];
dest[index + 1] = source[oIndex + 1];
dest[index + 2] = source[oIndex + 2];
*/

// int oCol = col / 4;
// int oRow = row / 4;
