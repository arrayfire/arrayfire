#include <cuComplex.h>
#include <math_functions.h>
typedef unsigned char uchar;
typedef unsigned int uint;
typedef cuFloatComplex cfloat;
typedef cuDoubleComplex cdouble;
typedef long long intl;
typedef unsigned long long uintl;

__device__ __inline__ float cabs2(cfloat in) { return in.x * in.x + in.y * in.y;}
__device__ __inline__ double cabs2(cdouble in) { return in.x * in.x + in.y * in.y; }
