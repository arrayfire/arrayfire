/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Parts of this code sourced from SnopyDogy
// https://gist.github.com/SnopyDogy/a9a22497a893ec86aa3e

#if defined (WITH_GRAPHICS)

#include <Array.hpp>
#include <image.hpp>
#include <err_cpu.hpp>
#include <cstdio>
#include <stdexcept>
#include <graphics_common.hpp>

using af::dim4;

namespace cpu
{
    template<typename T>
    void copy_image(const Array<T> &in, const fg::Image* image)
    {
        CheckGL("Before CopyArrayToPBO");
        const T *d_X = in.get();
        size_t data_size = image->size();

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, image->pbo());
        glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, data_size, d_X);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        CheckGL("In CopyArrayToPBO");
    }

    #define INSTANTIATE(T)  \
        template void copy_image<T>(const Array<T> &in, const fg::Image* image);

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}

#endif  // WITH_GRAPHICS
