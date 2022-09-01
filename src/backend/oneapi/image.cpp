/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <GraphicsResourceManager.hpp>
#include <err_oneapi.hpp>
#include <image.hpp>

#include <stdexcept>
#include <vector>

namespace oneapi {

template<typename T>
void copy_image(const Array<T> &in, fg_image image) {
    ONEAPI_NOT_SUPPORTED("");
}

#define INSTANTIATE(T) template void copy_image<T>(const Array<T> &, fg_image);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace oneapi
