/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <GraphicsResourceManager.hpp>
#include <debug_opencl.hpp>
#include <err_opencl.hpp>
#include <image.hpp>

#include <stdexcept>
#include <vector>

using arrayfire::common::ForgeModule;
using arrayfire::common::forgePlugin;

namespace arrayfire {
namespace opencl {

template<typename T>
void copy_image(const Array<T> &in, fg_image image) {
    ForgeModule &_ = forgePlugin();
    if (isGLSharingSupported()) {
        CheckGL("Begin opencl resource copy");

        auto res = interopManager().getImageResources(image);

        const cl::Buffer *d_X = in.get();

        unsigned bytes = 0;
        FG_CHECK(_.fg_get_image_size(&bytes, image));

        std::vector<cl::Memory> shared_objects;
        shared_objects.push_back(*(res[0].get()));

        glFinish();

        // Use of events:
        // https://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueReleaseGLObjects.html
        cl::Event event;

        getQueue().enqueueAcquireGLObjects(&shared_objects, NULL, &event);
        event.wait();
        getQueue().enqueueCopyBuffer(*d_X, *(res[0].get()), 0, 0, bytes, NULL,
                                     &event);
        getQueue().enqueueReleaseGLObjects(&shared_objects, NULL, &event);
        event.wait();

        CL_DEBUG_FINISH(getQueue());
        CheckGL("End opencl resource copy");
    } else {
        CheckGL("Begin OpenCL fallback-resource copy");
        unsigned bytes = 0, buffer = 0;
        FG_CHECK(_.fg_get_image_size(&bytes, image));
        FG_CHECK(_.fg_get_pixel_buffer(&buffer, image));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, bytes, 0, GL_STREAM_DRAW);
        auto *ptr = static_cast<GLubyte *>(
            glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY));
        if (ptr) {
            getQueue().enqueueReadBuffer(*in.get(), CL_TRUE, 0, bytes, ptr);
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        CheckGL("End OpenCL fallback-resource copy");
    }
}

#define INSTANTIATE(T) template void copy_image<T>(const Array<T> &, fg_image);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace opencl
}  // namespace arrayfire
