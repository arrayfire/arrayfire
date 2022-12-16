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
#include <vector_field.hpp>

using af::dim4;
using arrayfire::common::ForgeManager;
using arrayfire::common::ForgeModule;
using arrayfire::common::forgePlugin;

namespace arrayfire {
namespace cpu {

template<typename T>
void copy_vector_field(const Array<T> &points, const Array<T> &directions,
                       fg_vector_field vfield) {
    ForgeModule &_ = forgePlugin();
    points.eval();
    directions.eval();
    getQueue().sync();

    CheckGL("Before CopyArrayToVBO");

    unsigned size1 = 0, size2 = 0;
    unsigned buff1 = 0, buff2 = 0;
    FG_CHECK(_.fg_get_vector_field_vertex_buffer_size(&size1, vfield));
    FG_CHECK(_.fg_get_vector_field_direction_buffer_size(&size2, vfield));
    FG_CHECK(_.fg_get_vector_field_vertex_buffer(&buff1, vfield));
    FG_CHECK(_.fg_get_vector_field_direction_buffer(&buff2, vfield));

    glBindBuffer(GL_ARRAY_BUFFER, buff1);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size1, points.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, buff2);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size2, directions.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    CheckGL("In CopyArrayToVBO");
}

#define INSTANTIATE(T)                                                     \
    template void copy_vector_field<T>(const Array<T> &, const Array<T> &, \
                                       fg_vector_field);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cpu
}  // namespace arrayfire
