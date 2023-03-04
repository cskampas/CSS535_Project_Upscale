#if _MSC_VER
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include "bitmap.h"

#include <iostream>

using namespace std;

int main()
{
    cout << "Hello, World!" << endl;
    Bitmap* b = new Bitmap();
    b->readFromFile("C:\\Users\\cskam\\source\repos\\CSS535_Project_Upscale\\CSS535_Project_Upscale\\TestContent\\Test1.bmp");
    b->writeToFile("TestContent\\Test2.bmp");
    return 0;
}
