/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <scan_impl.hpp>

namespace cpu
{
    INSTANTIATE_SCAN(af_notzero_t, char, uint)
    INSTANTIATE_SCAN_ALL(af_add_t)
}
