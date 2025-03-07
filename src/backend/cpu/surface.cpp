/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/graphics_common.hpp>
#include <err_cpu.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <surface.hpp>

using af::dim4;
using arrayfire::common::ForgeManager;
using arrayfire::common::ForgeModule;
using arrayfire::common::forgePlugin;

namespace arrayfire {
namespace cpu {

template<typename T>
void copy_surface(const Array<T> &P, fg_surface surface) {
    ForgeModule &_ = common::forgePlugin();
    P.eval();
    getQueue().sync();

    CheckGL("Before CopyArrayToVBO");
    unsigned bytes = 0, buffer = 0;
    FG_CHECK(_.fg_get_surface_vertex_buffer(&buffer, surface));
    FG_CHECK(_.fg_get_surface_vertex_buffer_size(&bytes, surface));

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, P.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    CheckGL("In CopyArrayToVBO");
}

#define INSTANTIATE(T) \
    template void copy_surface<T>(const Array<T> &, fg_surface);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cpu
}  // namespace arrayfire
