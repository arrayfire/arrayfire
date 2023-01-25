/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <kernel/shift.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <shift.hpp>

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> shift(const Array<T> &in, const int sdims[4]) {
    Array<T> out = createEmptyArray<T>(in.dims());
    const af::dim4 temp(sdims[0], sdims[1], sdims[2], sdims[3]);

    getQueue().enqueue(kernel::shift<T>, out, in, temp);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> shift<T>(const Array<T> &in, const int sdims[4]);

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

}  // namespace cpu
}  // namespace arrayfire
