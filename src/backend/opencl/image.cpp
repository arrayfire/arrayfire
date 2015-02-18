/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_GRAPHICS)

#include <Array.hpp>
#include <graphics.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename T>
    void draw_image(const Array<T> &in, const ImageHandle &image)
    {
        printf("Error: Graphics not available for OpenCL backend.\n");
        AF_ERROR("Graphics not Available", AF_ERR_NOT_CONFIGURED);
    }

    #define INSTANTIATE(T)      \
        template void draw_image<T>(const Array<T> &in, const ImageHandle &image);

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}

#endif
