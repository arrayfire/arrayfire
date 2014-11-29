/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <graphics.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
    int image(const Array<float> &in, const int wId, const char *title)
    {
        AF_ERROR("Graphics not implemented on OpenCL backend", AF_ERR_NOT_SUPPORTED);
        return -1;
    }
}
