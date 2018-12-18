/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <plot.hpp>
#include <err_cpu.hpp>
#include <common/graphics_common.hpp>
#include <platform.hpp>
#include <queue.hpp>

using af::dim4;

namespace cpu {

template<typename T>
void copy_plot(const Array<T> &P, fg_plot plot)
{
    ForgeModule& _ = graphics::forgePlugin();
    P.eval();
    getQueue().sync();

    CheckGL("Before CopyArrayToVBO");
    unsigned bytes = 0, buffer = 0;
    FG_CHECK(fg_get_plot_vertex_buffer(&buffer, plot));
    FG_CHECK(fg_get_plot_vertex_buffer_size(&bytes, plot));

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, P.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    CheckGL("In CopyArrayToVBO");
}

#define INSTANTIATE(T)  \
template void copy_plot<T>(const Array<T> &, fg_plot);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}
