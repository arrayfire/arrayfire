/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "morph_impl.hpp"

namespace cuda {

INSTANTIATE(float, false)
INSTANTIATE(double, false)
INSTANTIATE(char, false)
INSTANTIATE(int, false)
INSTANTIATE(uint, false)
INSTANTIATE(uchar, false)
INSTANTIATE(short, false)
INSTANTIATE(ushort, false)

}  // namespace cuda
