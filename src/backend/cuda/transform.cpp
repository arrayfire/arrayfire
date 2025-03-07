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

namespace arrayfire {
namespace cuda {

template<typename T>
void transform(Array<T> &out, const Array<T> &in, const Array<float> &tf,
               const af::interpType method, const bool inverse,
               const bool perspective) {
    kernel::transform<T>(out, in, tf, inverse, perspective, method,
                         interpOrder(method));
}

#define INSTANTIATE(T)                                                       \
    template void transform(Array<T> &out, const Array<T> &in,               \
                            const Array<float> &tf,                          \
                            const af_interp_type method, const bool inverse, \
                            const bool perspective);

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
