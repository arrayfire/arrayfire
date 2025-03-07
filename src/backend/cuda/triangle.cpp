/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <triangle.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <kernel/triangle.hpp>
#include <af/dim4.hpp>

using af::dim4;
using arrayfire::common::half;

namespace arrayfire {
namespace cuda {

template<typename T>
void triangle(Array<T> &out, const Array<T> &in, const bool is_upper,
              const bool is_unit_diag) {
    kernel::triangle<T>(out, in, is_upper, is_unit_diag);
}

template<typename T>
Array<T> triangle(const Array<T> &in, const bool is_upper,
                  const bool is_unit_diag) {
    Array<T> out = createEmptyArray<T>(in.dims());
    triangle<T>(out, in, is_upper, is_unit_diag);
    return out;
}

#define INSTANTIATE(T)                                                  \
    template void triangle<T>(Array<T> &, const Array<T> &, const bool, \
                              const bool);                              \
    template Array<T> triangle<T>(const Array<T> &, const bool, const bool);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(char)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace cuda
}  // namespace arrayfire
