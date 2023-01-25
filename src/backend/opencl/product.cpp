/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/half.hpp>
#include "reduce_impl.hpp"

using arrayfire::common::half;

namespace arrayfire {
namespace opencl {
// sum
INSTANTIATE(af_mul_t, float, float)
INSTANTIATE(af_mul_t, double, double)
INSTANTIATE(af_mul_t, cfloat, cfloat)
INSTANTIATE(af_mul_t, cdouble, cdouble)
INSTANTIATE(af_mul_t, int, int)
INSTANTIATE(af_mul_t, uint, uint)
INSTANTIATE(af_mul_t, intl, intl)
INSTANTIATE(af_mul_t, uintl, uintl)
INSTANTIATE(af_mul_t, char, int)
INSTANTIATE(af_mul_t, uchar, uint)
INSTANTIATE(af_mul_t, short, int)
INSTANTIATE(af_mul_t, ushort, uint)
INSTANTIATE(af_mul_t, half, float)
}  // namespace opencl
}  // namespace arrayfire
