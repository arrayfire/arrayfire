/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <diff.hpp>
#include <err_cuda.hpp>
#include <kernel/diff.hpp>
#include <stdexcept>

namespace arrayfire {
namespace cuda {

template<typename T>
Array<T> diff(const Array<T> &in, const int dim, const bool isDiff2) {
    const af::dim4 &iDims = in.dims();
    af::dim4 oDims        = iDims;
    oDims[dim] -= (isDiff2 + 1);

    if (iDims.elements() == 0 || oDims.elements() == 0) {
        AF_ERROR("Elements are 0", AF_ERR_SIZE);
    }

    Array<T> out = createEmptyArray<T>(oDims);

    kernel::diff<T>(out, in, in.ndims(), dim, isDiff2);

    return out;
}

template<typename T>
Array<T> diff1(const Array<T> &in, const int dim) {
    return diff<T>(in, dim, false);
}

template<typename T>
Array<T> diff2(const Array<T> &in, const int dim) {
    return diff<T>(in, dim, true);
}

#define INSTANTIATE(T)                                             \
    template Array<T> diff1<T>(const Array<T> &in, const int dim); \
    template Array<T> diff2<T>(const Array<T> &in, const int dim);

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

}  // namespace cuda
}  // namespace arrayfire
