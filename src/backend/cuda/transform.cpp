/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <transform.hpp>

#include <kernel/transform.hpp>
#include <utility.hpp>

namespace cuda {
template<typename T>
Array<T> transform(const Array<T> &in, const Array<float> &tf,
                   const af::dim4 &odims, const af::interpType method,
                   const bool inverse, const bool perspective) {
    Array<T> out = createEmptyArray<T>(odims);
    kernel::transform<T>(out, in, tf, inverse, perspective, method,
                         interpOrder(method));
    return out;
}

#define INSTANTIATE(T)                                                      \
    template Array<T> transform(const Array<T> &in, const Array<float> &tf, \
                                const af::dim4 &odims,                      \
                                const af_interp_type method,                \
                                const bool inverse, const bool perspective);

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
