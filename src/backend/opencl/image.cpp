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
#include <debug_opencl.hpp>
#include <err_opencl.hpp>
#include <image.hpp>
#include <GraphicsResourceManager.hpp>

#include <stdexcept>
#include <vector>

namespace opencl
{
using namespace gl;

template<typename T>
void copy_image(const Array<T> &in, const forge::Image* image)
{
    if (isGLSharingSupported()) {
        CheckGL("Begin opencl resource copy");

        ShrdResVector res = interopManager().getBufferResource(image);

        const cl::Buffer *d_X = in.get();
        size_t num_bytes = image->size();

        std::vector<cl::Memory> shared_objects;
        shared_objects.push_back(*(res[0].get()));

        glFinish();

        // Use of events:
        // https://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueReleaseGLObjects.html
        cl::Event event;

        getQueue().enqueueAcquireGLObjects(&shared_objects, NULL, &event);
        event.wait();
        getQueue().enqueueCopyBuffer(*d_X, *(res[0].get()), 0, 0, num_bytes, NULL, &event);
        getQueue().enqueueReleaseGLObjects(&shared_objects, NULL, &event);
        event.wait();

        CL_DEBUG_FINISH(getQueue());
        CheckGL("End opencl resource copy");
    } else {
        CheckGL("Begin OpenCL fallback-resource copy");
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, image->pixels());
        glBufferData(GL_PIXEL_UNPACK_BUFFER, image->size(), 0, GL_STREAM_DRAW);
        GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
        if (ptr) {
            getQueue().enqueueReadBuffer(*in.get(), CL_TRUE, 0, image->size(), ptr);
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        CheckGL("End OpenCL fallback-resource copy");
    }
}

#define INSTANTIATE(T)      \
    template void copy_image<T>(const Array<T> &in, const forge::Image* image);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)

}

#endif
