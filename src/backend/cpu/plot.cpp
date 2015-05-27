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
#include <plot.hpp>
#include <err_cpu.hpp>
#include <stdexcept>
#include <graphics_common.hpp>
#include <reduce.hpp>
#include <memory.hpp>

using af::dim4;

namespace cpu
{
    template<typename T>
    void copy_plot(const Array<T> &P, fg::Plot* plot)
    {
        CheckGL("Before CopyArrayToVBO");

        glBindBuffer(GL_ARRAY_BUFFER, plot->vbo());
        glBufferSubData(GL_ARRAY_BUFFER, 0, plot->size(), P.get());
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        CheckGL("In CopyArrayToVBO");
    }

    #define INSTANTIATE(T)  \
        template void copy_plot<T>(const Array<T> &P, fg::Plot* plot);

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
}

#endif  // WITH_GRAPHICS
