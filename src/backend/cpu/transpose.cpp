/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <kernel/transpose.hpp>
#include <platform.hpp>
#include <transpose.hpp>
#include <af/dim4.hpp>
#include <cassert>
#include <utility>

using af::dim4;

namespace cpu {

template<typename T>
Array<T> transpose(const Array<T> &in, const bool conjugate) {
    in.eval();

    const dim4 inDims  = in.dims();
    const dim4 outDims = dim4(inDims[1], inDims[0], inDims[2], inDims[3]);
    // create an array with first two dimensions swapped
    Array<T> out = createEmptyArray<T>(outDims);

    getQueue().enqueue(kernel::transpose<T>, out, in, conjugate);

    return out;
}

template<typename T>
void transpose_inplace(Array<T> &in, const bool conjugate) {
    in.eval();
    getQueue().enqueue(kernel::transpose_inplace<T>, in, conjugate);
}

#define INSTANTIATE(T)                                                     \
    template Array<T> transpose(const Array<T> &in, const bool conjugate); \
    template void transpose_inplace(Array<T> &in, const bool conjugate);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cpu
