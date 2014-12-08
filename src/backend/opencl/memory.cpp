/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memory.hpp>
#include <platform.hpp>

namespace opencl
{

    cl::Buffer *memAlloc(const size_t &bytes)
    {
        cl::Buffer *res = NULL;
        if (bytes > 0) res = new cl::Buffer(getContext(), CL_MEM_READ_WRITE, bytes);
        return res;
    }

    void memFree(cl::Buffer *buf)
    {
        delete buf;
    }
}
