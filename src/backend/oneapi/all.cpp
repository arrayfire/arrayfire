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
// alltrue
INSTANTIATE(af_and_t, float, char)
INSTANTIATE(af_and_t, double, char)
INSTANTIATE(af_and_t, cfloat, char)
INSTANTIATE(af_and_t, cdouble, char)
INSTANTIATE(af_and_t, int, char)
INSTANTIATE(af_and_t, uint, char)
INSTANTIATE(af_and_t, intl, char)
INSTANTIATE(af_and_t, uintl, char)
INSTANTIATE(af_and_t, char, char)
INSTANTIATE(af_and_t, uchar, char)
INSTANTIATE(af_and_t, short, char)
INSTANTIATE(af_and_t, ushort, char)
INSTANTIATE(af_and_t, half, char)
}  // namespace oneapi
}  // namespace arrayfire
