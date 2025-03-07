/*******************************************************
 * Copyright (c) 2022, ArrayFire
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
namespace oneapi {
// min
INSTANTIATE(af_min_t, float, float)
INSTANTIATE(af_min_t, double, double)
INSTANTIATE(af_min_t, cfloat, cfloat)
INSTANTIATE(af_min_t, cdouble, cdouble)
INSTANTIATE(af_min_t, int, int)
INSTANTIATE(af_min_t, uint, uint)
INSTANTIATE(af_min_t, intl, intl)
INSTANTIATE(af_min_t, uintl, uintl)
INSTANTIATE(af_min_t, char, char)
INSTANTIATE(af_min_t, uchar, uchar)
INSTANTIATE(af_min_t, short, short)
INSTANTIATE(af_min_t, ushort, ushort)
INSTANTIATE(af_min_t, half, half)
}  // namespace oneapi
}  // namespace arrayfire
