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
// count
INSTANTIATE(af_notzero_t, float, uint)
INSTANTIATE(af_notzero_t, double, uint)
INSTANTIATE(af_notzero_t, cfloat, uint)
INSTANTIATE(af_notzero_t, cdouble, uint)
INSTANTIATE(af_notzero_t, int, uint)
INSTANTIATE(af_notzero_t, uint, uint)
INSTANTIATE(af_notzero_t, intl, uint)
INSTANTIATE(af_notzero_t, uintl, uint)
INSTANTIATE(af_notzero_t, short, uint)
INSTANTIATE(af_notzero_t, ushort, uint)
INSTANTIATE(af_notzero_t, char, uint)
INSTANTIATE(af_notzero_t, uchar, uint)
INSTANTIATE(af_notzero_t, half, uint)
}  // namespace cuda
}  // namespace arrayfire
