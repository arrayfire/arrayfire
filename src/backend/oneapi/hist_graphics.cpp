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
#include <hist_graphics.hpp>

namespace oneapi {

template<typename T>
void copy_histogram(const Array<T> &data, fg_histogram hist) {
    ONEAPI_NOT_SUPPORTED("");
}

#define INSTANTIATE(T) \
    template void copy_histogram<T>(const Array<T> &, fg_histogram);

INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}  // namespace oneapi
