/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_cuda.hpp>
#include <af/dim4.hpp>

#undef _GLIBCXX_USE_INT128
#include <kernel/where.hpp>
#include <where.hpp>
#include <complex>

namespace arrayfire {
namespace cuda {
template<typename T>
Array<uint> where(const Array<T> &in) {
    Param<uint> out;
    kernel::where<T>(out, in);
    return createParamArray<uint>(out, true);
}

#define INSTANTIATE(T) template Array<uint> where<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cuda
}  // namespace arrayfire
