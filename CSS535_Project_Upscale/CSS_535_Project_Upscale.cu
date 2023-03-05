#if _MSC_VER
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include "Bitmap.h"

#include <iostream>

using namespace std;


void print_matrix(short* matrix, int size){
	for(int i =0; i < size; i++){
		for(int j=0; j < size; j++){
			cout << matrix[i*size + j] << " ";
		}
		cout << endl; 
	}
}

#define BLOCK_SIZE 2
__global__ void UpscaleImage(short* a, short* b, int n) {

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

    int index = col + row * n;

	if (row < n && col < n) {
		b[index] = a[index];
	}
}

int main()
{
    cout << "Hello, World!" << endl;
    Bitmap* b = new Bitmap();
    // b->readFromFile("TestContent\\Test1.bmp");
    // b->writeToFile("TestContent\\Test2.bmp");


	// ------- Dummy code -----
	
    int N = 4;
    
	short *original_image, *upscaled_image;
    short *original_image_device, *upscaled_image_device;
    
	int size_matrix = N * N * sizeof(short);

    ///////
	original_image = (short*)malloc(size_matrix);
	for(short i = 0; i < N * N; i++){
		original_image[i] = i;
	}
    ////

    upscaled_image = (short*)malloc(size_matrix);

    cudaMalloc((void**)&original_image_device, size_matrix);
    cudaMalloc((void**)&upscaled_image_device, size_matrix);

    cudaMemcpy(original_image_device, original_image, size_matrix, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
    UpscaleImage <<<dimGrid, dimBlock >>>(original_image_device, upscaled_image_device, N);

    cudaMemcpy(upscaled_image, upscaled_image_device, size_matrix, cudaMemcpyDeviceToHost);

    print_matrix(original_image, N);
    print_matrix(upscaled_image, N);

    cudaFree(original_image_device);
    cudaFree(upscaled_image_device);

    free(original_image);
    free(upscaled_image);

	// ------------------------

    return 0;
}
