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
INSTANTIATE(af_add_t, float, float)
INSTANTIATE(af_add_t, double, double)
INSTANTIATE(af_add_t, cfloat, cfloat)
INSTANTIATE(af_add_t, cdouble, cdouble)
INSTANTIATE(af_add_t, int, int)
INSTANTIATE(af_add_t, int, float)
INSTANTIATE(af_add_t, uint, uint)
INSTANTIATE(af_add_t, uint, float)
INSTANTIATE(af_add_t, intl, intl)
INSTANTIATE(af_add_t, intl, double)
INSTANTIATE(af_add_t, uintl, uintl)
INSTANTIATE(af_add_t, uintl, double)
INSTANTIATE(af_add_t, char, int)
INSTANTIATE(af_add_t, char, float)
INSTANTIATE(af_add_t, uchar, uint)
INSTANTIATE(af_add_t, uchar, float)
INSTANTIATE(af_add_t, short, int)
INSTANTIATE(af_add_t, short, float)
INSTANTIATE(af_add_t, ushort, uint)
INSTANTIATE(af_add_t, ushort, float)
INSTANTIATE(af_add_t, half, half)
INSTANTIATE(af_add_t, half, float)
}  // namespace opencl
}  // namespace arrayfire
