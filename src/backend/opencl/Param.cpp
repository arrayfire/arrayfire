/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <Param.hpp>
#include <platform.hpp>
#include <kernel/KParam.hpp>

namespace opencl
{
    Param makeParam(cl_mem mem, int off, int dims[4], int strides[4])
    {
        Param out;
        out.data = new cl::Buffer(mem);
        out.info.offset = off;
        for (int i = 0; i < 4; i++) {
            out.info.dims[i] = dims[i];
            out.info.strides[i] = strides[i];
        }
        return out;
    }
}
