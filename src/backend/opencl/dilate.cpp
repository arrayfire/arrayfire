/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "morph_impl.hpp"

namespace opencl
{

INSTANTIATE(float , true)
INSTANTIATE(double, true)
INSTANTIATE(char  , true)
INSTANTIATE(int   , true)
INSTANTIATE(uint  , true)
INSTANTIATE(uchar , true)
INSTANTIATE(short , true)
INSTANTIATE(ushort, true)

}
