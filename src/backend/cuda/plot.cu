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
#include <graphics.hpp>
#include <err_cpu.hpp>
#include <cstdio>
#include <stdexcept>
#include <graphics_common.hpp>
#include <join.hpp>
#include <reduce.hpp>
#include <iostream>
#include <memory.hpp>

using namespace std;
using af::dim4;

namespace cuda
{
    template<typename T>
    void copy_plot(const Array<T> &X, const Array<T> &Y, const fg_plot_handle plot)
    {
        printf("Error: Plot not available for CUDA backend.\n");
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
