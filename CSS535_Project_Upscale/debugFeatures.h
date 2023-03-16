#ifndef DebugFeatures_H
#define DebugFeatures_H

#include "bitmap.h"

class DebugFeatures
{
public:
	static unsigned int stopX;
	static unsigned int stopY;
	static void emulator(Bitmap* source, Bitmap* dest);
};
#endif
