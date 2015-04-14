/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined (WITH_GRAPHICS)

#include <Array.hpp>
#include <plot.hpp>
#include <err_opencl.hpp>
#include <graphics_common.hpp>

using af::dim4;

namespace opencl
{
    template<typename T>
    void copy_plot(const Array<T> &X, const Array<T> &Y, const fg_plot_handle plot)
    {
        printf("Error: Graphics not available for OpenCL backend.\n");
        AF_ERROR("Graphics not Available", AF_ERR_NOT_CONFIGURED);

    }

    #define INSTANTIATE(T)  \
        template void copy_plot<T>(const Array<T> &X, const Array<T> &Y, const fg_plot_handle plot);

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
}

#endif  // WITH_GRAPHICS
