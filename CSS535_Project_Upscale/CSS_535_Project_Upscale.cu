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

#define BLOCK_SIZE 2
__global__ void UpscaleImage(unsigned char* a, unsigned char* b, unsigned short width, unsigned short height, int pad) {

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	int index = ((col + row * width) * 3) + row * pad;

	if (row < height && col < width) {
		b[index] = a[index];
        b[index + 1] = a[index + 1];
        b[index + 2] = a[index + 2];
	}
}

int main()
{
    cout << "Hello, World!" << endl;
    Bitmap* b = new Bitmap();
    b->readFromFile("TestContent/Test1.bmp");
    b->writeToFile("TestContent/Test2.bmp");


	// ------- Dummy code -----
	
    unsigned short width = b->width;
	unsigned short height = b->height;
    
	unsigned char *original_image, *upscaled_image;
    unsigned char *original_image_device, *upscaled_image_device;
    
	int size_matrix = 3 * width * height * sizeof(unsigned char) + height * b->padSide();

	original_image = b->imageData;    

    upscaled_image = (unsigned char*)malloc(size_matrix);

    cudaMalloc((void**)&original_image_device, size_matrix);
    cudaMalloc((void**)&upscaled_image_device, size_matrix);

    cudaMemcpy(original_image_device, original_image, size_matrix, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
    UpscaleImage <<<dimGrid, dimBlock >>>(original_image_device, upscaled_image_device, width, height, b->padSide());

    cudaMemcpy(upscaled_image, upscaled_image_device, size_matrix, cudaMemcpyDeviceToHost);

    cout << "ORIGINAL" << endl;
    print_matrix(original_image, width, height, b->padSide());
    cout << "COPY" << endl;
    print_matrix(upscaled_image, width, height, b->padSide());

    cudaFree(original_image_device);
    cudaFree(upscaled_image_device);

    free(original_image);
    free(upscaled_image);

	// ------------------------

    return 0;
}
