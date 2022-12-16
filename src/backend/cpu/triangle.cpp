/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <triangle.hpp>

#include <common/half.hpp>
#include <kernel/triangle.hpp>
#include <platform.hpp>
#include <af/dim4.hpp>

#include <functional>

using arrayfire::common::half;

namespace arrayfire {
namespace cpu {

template<typename T>
using triangleFunc = std::function<void(Param<T>, CParam<T>)>;

template<typename T>
void triangle(Array<T> &out, const Array<T> &in, const bool is_upper,
              const bool is_unit_diag) {
    static const triangleFunc<T> funcs[4] = {
        kernel::triangle<T, false, false>,
        kernel::triangle<T, false, true>,
        kernel::triangle<T, true, false>,
        kernel::triangle<T, true, true>,
    };
    const int funcIdx = is_upper * 2 + is_unit_diag;
    getQueue().enqueue(funcs[funcIdx], out, in);
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

}  // namespace cpu
}  // namespace arrayfire
