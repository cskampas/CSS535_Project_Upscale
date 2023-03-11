#ifndef DebugFeatures_H
#define DebugFeatures_H

#include "bitmap.h"

class DebugFeatures
{
public:
	static unsigned short stopX;
	static unsigned short stopY;
	static void emulator(Bitmap* source, Bitmap* dest);
};
#endif
