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

    cl::Buffer *bufferAlloc(const size_t &bytes);
    void bufferFree(cl::Buffer *buf);

    template<typename T> T *memAlloc(const size_t &elements);
    template<typename T> void memFree(T *ptr);
}
