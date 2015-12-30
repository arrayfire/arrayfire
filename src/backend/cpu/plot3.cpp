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
#include <plot3.hpp>
#include <err_cpu.hpp>
#include <graphics_common.hpp>
#include <platform.hpp>
#include <queue.hpp>

using af::dim4;

namespace cpu
{

template<typename T>
void copy_plot3(const Array<T> &P, fg::Plot3* plot3)
{
    P.eval();
    getQueue().sync();
    CheckGL("Before CopyArrayToVBO");

    glBindBuffer(GL_ARRAY_BUFFER, plot3->vbo());
    glBufferSubData(GL_ARRAY_BUFFER, 0, plot3->size(), P.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    CheckGL("In CopyArrayToVBO");
}

#define INSTANTIATE(T)  \
    template void copy_plot3<T>(const Array<T> &P, fg::Plot3* plot3);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}

#endif  // WITH_GRAPHICS
