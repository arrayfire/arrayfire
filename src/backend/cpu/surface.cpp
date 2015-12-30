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
#include <surface.hpp>
#include <err_cpu.hpp>
#include <graphics_common.hpp>
#include <platform.hpp>
#include <queue.hpp>

using af::dim4;

namespace cpu
{

template<typename T>
void copy_surface(const Array<T> &P, fg::Surface* surface)
{
    P.eval();
    getQueue().sync();
    CheckGL("Before CopyArrayToVBO");

    glBindBuffer(GL_ARRAY_BUFFER, surface->vbo());
    glBufferSubData(GL_ARRAY_BUFFER, 0, surface->size(), P.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    CheckGL("In CopyArrayToVBO");
}

#define INSTANTIATE(T)  \
    template void copy_surface<T>(const Array<T> &P, fg::Surface* surface);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}

#endif  // WITH_GRAPHICS
