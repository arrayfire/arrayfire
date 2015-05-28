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

AFAPI int devicecount();

AFAPI int deviceget();

AFAPI void deviceset(const int device);

AFAPI array loadimage(const char* filename, const bool is_color=false);

AFAPI void saveimage(const char* filename, const array& in);

AFAPI array gaussiankernel(const int rows, const int cols, const double sig_r = 0, const double sig_c = 0);

template<typename T> T alltrue(const array &in);

template<typename T> T anytrue(const array &in);

AFAPI array alltrue(const array &in, const int dim = -1);

AFAPI array anytrue(const array &in, const int dim = -1);

AFAPI array setunique(const array &in, const bool is_sorted=false);

AFAPI array setunion(const array &first, const array &second, const bool is_unique=false);

AFAPI array setintersect(const array &first, const array &second, const bool is_unique=false);

AFAPI array histequal(const array& in, const array& hist);

AFAPI array colorspace(const array& image, const CSpace to, const CSpace from);

AFAPI void deviceprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

}
#endif
