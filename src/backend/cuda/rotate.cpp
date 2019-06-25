/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <kernel/rotate.hpp>
#include <rotate.hpp>
#include <utility.hpp>

namespace cuda {

template<typename T>
Array<T> rotate(const Array<T> &in, const float theta, const af::dim4 &odims,
                const af_interp_type method) {
    Array<T> out      = createEmptyArray<T>(odims);
    auto interpParams = toInternalEnum(method);
    kernel::rotate<T>(out, in, theta, interpParams.first, interpParams.second);
    return out;
}

#define INSTANTIATE(T)                                              \
    template Array<T> rotate(const Array<T> &in, const float theta, \
                             const af::dim4 &odims,                 \
                             const af_interp_type method);

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
