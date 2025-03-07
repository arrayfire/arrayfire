/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <kernel/tile.hpp>
#include <tile.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <stdexcept>

using arrayfire::common::half;

namespace arrayfire {
namespace opencl {
template<typename T>
Array<T> tile(const Array<T> &in, const af::dim4 &tileDims) {
    const af::dim4 &iDims = in.dims();
    af::dim4 oDims        = iDims;
    oDims *= tileDims;

    Array<T> out = createEmptyArray<T>(oDims);

    kernel::tile<T>(out, in);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> tile<T>(const Array<T> &in, const af::dim4 &tileDims);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace opencl
}  // namespace arrayfire
