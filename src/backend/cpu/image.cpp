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
#include <common/graphics_common.hpp>
#include <platform.hpp>
#include <queue.hpp>

using af::dim4;

namespace cpu {

template<typename T>
void copy_image(const Array<T> &in, fg_image image)
{
    ForgeModule& _ = graphics::forgePlugin();
    in.eval();
    getQueue().sync();

    CheckGL("Before CopyArrayToImage");
    const T *d_X = in.get();
    unsigned data_size = 0, buffer = 0;
    FG_CHECK(fg_get_pixel_buffer(&buffer, image));
    FG_CHECK(fg_get_image_size(&data_size, image));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
    glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, data_size, d_X);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CheckGL("In CopyArrayToImage");
}

#define INSTANTIATE(T)  \
template void copy_image<T>(const Array<T> &, fg_image);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)

}

#endif  // WITH_GRAPHICS
