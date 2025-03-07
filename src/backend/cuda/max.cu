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
namespace cuda {
// max
INSTANTIATE(af_max_t, float, float)
INSTANTIATE(af_max_t, double, double)
INSTANTIATE(af_max_t, cfloat, cfloat)
INSTANTIATE(af_max_t, cdouble, cdouble)
INSTANTIATE(af_max_t, int, int)
INSTANTIATE(af_max_t, uint, uint)
INSTANTIATE(af_max_t, intl, intl)
INSTANTIATE(af_max_t, uintl, uintl)
INSTANTIATE(af_max_t, char, char)
INSTANTIATE(af_max_t, uchar, uchar)
INSTANTIATE(af_max_t, short, short)
INSTANTIATE(af_max_t, ushort, ushort)
INSTANTIATE(af_max_t, half, half)
}  // namespace cuda
}  // namespace arrayfire
