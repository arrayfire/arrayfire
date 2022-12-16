/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <diff.hpp>

#include <Array.hpp>
#include <kernel/diff.hpp>
#include <platform.hpp>

#include <af/dim4.hpp>

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> diff1(const Array<T> &in, const int dim) {
    // Decrement dimension of select dimension
    af::dim4 dims = in.dims();
    dims[dim]--;

    Array<T> outArray = createEmptyArray<T>(dims);

    getQueue().enqueue(kernel::diff1<T>, outArray, in, dim);

    return outArray;
}

template<typename T>
Array<T> diff2(const Array<T> &in, const int dim) {
    // Decrement dimension of select dimension
    af::dim4 dims = in.dims();
    dims[dim] -= 2;

    Array<T> outArray = createEmptyArray<T>(dims);

    getQueue().enqueue(kernel::diff2<T>, outArray, in, dim);

    return outArray;
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
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace cpu
}  // namespace arrayfire
