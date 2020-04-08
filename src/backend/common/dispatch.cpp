/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "dispatch.hpp"

unsigned nextpow2(unsigned x) {
    x = x - 1U;
    x = x | (x >> 1U);
    x = x | (x >> 2U);
    x = x | (x >> 4U);
    x = x | (x >> 8U);
    x = x | (x >> 16U);
    return x + 1U;
}
