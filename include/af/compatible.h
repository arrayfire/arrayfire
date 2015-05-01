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


}
#endif
