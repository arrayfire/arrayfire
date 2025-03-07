/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <diagonal.hpp>
#include <kernel/diagonal.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <platform.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>

#include <algorithm>
#include <cstdlib>

using arrayfire::common::half;  // NOLINT(misc-unused-using-decls) bug in
                                // clang-tidy
using std::abs;  // NOLINT(misc-unused-using-decls) bug in clang-tidy
using std::min;  // NOLINT(misc-unused-using-decls) bug in clang-tidy

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> diagCreate(const Array<T> &in, const int num) {
    int size     = in.dims()[0] + abs(num);
    int batch    = in.dims()[1];
    Array<T> out = createEmptyArray<T>(dim4(size, size, batch));

    getQueue().enqueue(kernel::diagCreate<T>, out, in, num);

    return out;
}

template<typename T>
Array<T> diagExtract(const Array<T> &in, const int num) {
    const dim4 &idims = in.dims();
    dim_t size        = min(idims[0], idims[1]) - abs(num);
    Array<T> out      = createEmptyArray<T>(dim4(size, 1, idims[2], idims[3]));

    getQueue().enqueue(kernel::diagExtract<T>, out, in, num);

    return out;
}

#define INSTANTIATE_DIAGONAL(T)                                          \
    template Array<T> diagExtract<T>(const Array<T> &in, const int num); \
    template Array<T> diagCreate<T>(const Array<T> &in, const int num);

INSTANTIATE_DIAGONAL(float)
INSTANTIATE_DIAGONAL(double)
INSTANTIATE_DIAGONAL(cfloat)
INSTANTIATE_DIAGONAL(cdouble)
INSTANTIATE_DIAGONAL(int)
INSTANTIATE_DIAGONAL(uint)
INSTANTIATE_DIAGONAL(intl)
INSTANTIATE_DIAGONAL(uintl)
INSTANTIATE_DIAGONAL(char)
INSTANTIATE_DIAGONAL(uchar)
INSTANTIATE_DIAGONAL(short)
INSTANTIATE_DIAGONAL(ushort)
INSTANTIATE_DIAGONAL(half)

}  // namespace cpu
}  // namespace arrayfire
