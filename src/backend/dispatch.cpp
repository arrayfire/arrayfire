#include "dispatch.hpp"

unsigned nextpow2(unsigned x)
{
	   x = x - 1;
	   x = x | (x >> 1);
	   x = x | (x >> 2);
	   x = x | (x >> 4);
	   x = x | (x >> 8);
	   x = x | (x >>16);
	   return x + 1;
}
