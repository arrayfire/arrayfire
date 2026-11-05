/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/sort_by_key_impl.hpp>

// SBK_TYPES:float double int uint intl uintl short ushort char schar uchar half

namespace arrayfire {
namespace oneapi {
namespace kernel {
// uint values are what sort with indices uses, so they get their own object
// and their own device image. See the CMakeLists.txt next to this file.
#if SBK_UINT_VALUES
INSTANTIATE(TYPE, uint)
#else
INSTANTIATE(TYPE, float)
INSTANTIATE(TYPE, double)
INSTANTIATE(TYPE, cfloat)
INSTANTIATE(TYPE, cdouble)
INSTANTIATE(TYPE, int)
INSTANTIATE(TYPE, short)
INSTANTIATE(TYPE, ushort)
INSTANTIATE(TYPE, char)
INSTANTIATE(TYPE, schar)
INSTANTIATE(TYPE, uchar)
INSTANTIATE(TYPE, intl)
INSTANTIATE(TYPE, uintl)
#endif
}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
