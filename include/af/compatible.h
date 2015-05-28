/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/image.h>

#ifdef __cplusplus
namespace af
{
class array;

DEPRECATED("Use getDeviceCount instead")
AFAPI int devicecount();

DEPRECATED("Use getDevice instead")
AFAPI int deviceget();

DEPRECATED("Use setDevice instead")
AFAPI void deviceset(const int device);

DEPRECATED("Use loadImage instead")
AFAPI array loadimage(const char* filename, const bool is_color=false);

DEPRECATED("Use saveImage instead")
AFAPI void saveimage(const char* filename, const array& in);

DEPRECATED("Use gaussianKernel instead")
AFAPI array gaussiankernel(const int rows, const int cols, const double sig_r = 0, const double sig_c = 0);

template<typename T>
DEPRECATED("Use allTrue instead")
T alltrue(const array &in);

template<typename T>
DEPRECATED("Use anyTrue instead")
T anytrue(const array &in);

DEPRECATED("Use allTrue instead")
AFAPI array alltrue(const array &in, const int dim = -1);

DEPRECATED("Use anyTrue instead")
AFAPI array anytrue(const array &in, const int dim = -1);

DEPRECATED("Use setUnique instead")
AFAPI array setunique(const array &in, const bool is_sorted=false);

DEPRECATED("Use setUnion instead")
AFAPI array setunion(const array &first, const array &second, const bool is_unique=false);

DEPRECATED("Use setIntersect instead")
AFAPI array setintersect(const array &first, const array &second, const bool is_unique=false);

DEPRECATED("Use histEqual instead")
AFAPI array histequal(const array& in, const array& hist);

DEPRECATED("Use colorSpace instead")
AFAPI array colorspace(const array& image, const CSpace to, const CSpace from);

DEPRECATED("Use deviceInfo instead")
AFAPI void deviceprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

}
#endif
