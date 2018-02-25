/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined (WITH_GRAPHICS)

#include <hist_graphics.hpp>
#include <err_cpu.hpp>
#include <platform.hpp>
#include <queue.hpp>

namespace cpu
{
using namespace gl;

template<typename T>
void copy_histogram(const Array<T> &data, const forge::Histogram* hist)
{
    data.eval();
    getQueue().sync();

    CheckGL("Begin copy_histogram");

    glBindBuffer(GL_ARRAY_BUFFER, hist->vertices());
    glBufferSubData(GL_ARRAY_BUFFER, 0, hist->verticesSize(), data.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    CheckGL("End copy_histogram");
}

#define INSTANTIATE(T)  \
    template void copy_histogram<T>(const Array<T> &data, const forge::Histogram* hist);

INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

#undef INSTANTIATE
}

#endif  // WITH_GRAPHICS
