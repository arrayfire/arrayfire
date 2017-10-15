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
#include <vector_field.hpp>
#include <err_cpu.hpp>
#include <common/graphics_common.hpp>
#include <platform.hpp>
#include <queue.hpp>

using af::dim4;

namespace cpu
{
using namespace gl;

template<typename T>
void copy_vector_field(const Array<T> &points, const Array<T> &directions,
                       forge::VectorField* vector_field)
{
    points.eval();
    directions.eval();
    getQueue().sync();

    CheckGL("Before CopyArrayToVBO");

    glBindBuffer(GL_ARRAY_BUFFER, vector_field->vertices());
    glBufferSubData(GL_ARRAY_BUFFER, 0, vector_field->verticesSize(), points.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, vector_field->directions());
    glBufferSubData(GL_ARRAY_BUFFER, 0, vector_field->directionsSize(), directions.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    CheckGL("In CopyArrayToVBO");
}

#define INSTANTIATE(T)                                                                      \
    template void copy_vector_field<T>(const Array<T> &points, const Array<T> &directions,  \
                                       forge::VectorField* vector_field);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}

#endif  // WITH_GRAPHICS
