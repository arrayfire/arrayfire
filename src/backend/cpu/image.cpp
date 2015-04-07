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
    void copy_image(const Array<T> &in, const fg_image_handle image)
    {
        CheckGL("Before CopyArrayToPBO");
        const T *d_X = in.get();

        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, image->src_width * image->src_height * image->window->mode * sizeof(T),
                     d_X, GL_STREAM_COPY);

        // Unlock array
        // Not implemented yet
        // X.unlock();
        CheckGL("In CopyArrayToPBO");
    }

    #define INSTANTIATE(T)  \
        template void copy_image<T>(const Array<T> &in, const fg_image_handle image);

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}

#endif  // WITH_GRAPHICS
