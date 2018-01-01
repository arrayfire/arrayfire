/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/scan.hpp>

namespace cuda {
template Array<uint> scan<af_notzero_t, char, uint>(const Array<char> &in, const int dim, bool inclusive_scan);
}