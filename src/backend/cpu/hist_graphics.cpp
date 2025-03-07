/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_cpu.hpp>
#include <hist_graphics.hpp>
#include <platform.hpp>
#include <queue.hpp>

using arrayfire::common::ForgeManager;
using arrayfire::common::ForgeModule;
using arrayfire::common::forgePlugin;

namespace arrayfire {
namespace cpu {

template<typename T>
void copy_histogram(const Array<T> &data, fg_histogram hist) {
    ForgeModule &_ = forgePlugin();
    data.eval();
    getQueue().sync();

    CheckGL("Begin copy_histogram");
    unsigned bytes = 0, buffer = 0;
    FG_CHECK(_.fg_get_histogram_vertex_buffer(&buffer, hist));
    FG_CHECK(_.fg_get_histogram_vertex_buffer_size(&bytes, hist));

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, data.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    CheckGL("End copy_histogram");
}

#define INSTANTIATE(T) \
    template void copy_histogram<T>(const Array<T> &, fg_histogram);

INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cpu
}  // namespace arrayfire
