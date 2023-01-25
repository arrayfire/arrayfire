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
#include <hist_graphics.hpp>

using arrayfire::common::ForgeModule;
using arrayfire::common::forgePlugin;

namespace arrayfire {
namespace opencl {

template<typename T>
void copy_histogram(const Array<T> &data, fg_histogram hist) {
    ForgeModule &_ = forgePlugin();
    if (isGLSharingSupported()) {
        CheckGL("Begin OpenCL resource copy");
        const cl::Buffer *d_P = data.get();
        unsigned bytes        = 0;
        FG_CHECK(_.fg_get_histogram_vertex_buffer_size(&bytes, hist));

        auto res = interopManager().getHistogramResources(hist);

        std::vector<cl::Memory> shared_objects;
        shared_objects.push_back(*(res[0].get()));

        glFinish();

        // Use of events:
        // https://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueReleaseGLObjects.html
        cl::Event event;

        getQueue().enqueueAcquireGLObjects(&shared_objects, NULL, &event);
        event.wait();
        getQueue().enqueueCopyBuffer(*d_P, *(res[0].get()), 0, 0, bytes, NULL,
                                     &event);
        getQueue().enqueueReleaseGLObjects(&shared_objects, NULL, &event);
        event.wait();

        CL_DEBUG_FINISH(getQueue());
        CheckGL("End OpenCL resource copy");
    } else {
        unsigned bytes = 0, buffer = 0;
        FG_CHECK(_.fg_get_histogram_vertex_buffer(&buffer, hist));
        FG_CHECK(_.fg_get_histogram_vertex_buffer_size(&bytes, hist));

        CheckGL("Begin OpenCL fallback-resource copy");
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        auto *ptr =
            static_cast<GLubyte *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
        if (ptr) {
            getQueue().enqueueReadBuffer(*data.get(), CL_TRUE, 0, bytes, ptr);
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        CheckGL("End OpenCL fallback-resource copy");
    }
}

#define INSTANTIATE(T) \
    template void copy_histogram<T>(const Array<T> &, fg_histogram);

INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}  // namespace opencl
}  // namespace arrayfire
