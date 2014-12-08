/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <cl.hpp>
namespace opencl
{

    cl::Buffer *memAlloc(const size_t &bytes);
    void memFree(cl::Buffer *buf);
}
