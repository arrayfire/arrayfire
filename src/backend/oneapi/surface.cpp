/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
// #include <GraphicsResourceManager.hpp>
// #include <debug_oneapi.hpp>
#include <err_oneapi.hpp>
#include <surface.hpp>

using af::dim4;
// using cl::Memory;
using std::vector;

namespace arrayfire {
namespace oneapi {

template<typename T>
void copy_surface(const Array<T> &P, fg_surface surface) {
    ONEAPI_NOT_SUPPORTED("copy_surface Not supported");
    // ForgeModule &_ = common::forgePlugin();
    // if (isGLSharingSupported()) {
    //     CheckGL("Begin OpenCL resource copy");
    //     const cl::Buffer *d_P = P.get();
    //     unsigned bytes        = 0;
    //     FG_CHECK(_.fg_get_surface_vertex_buffer_size(&bytes, surface));

    //     auto res = interopManager().getSurfaceResources(surface);

    //     vector<Memory> shared_objects;
    //     shared_objects.push_back(*(res[0].get()));

    //     glFinish();

    //     // Use of events:
    //     //
    //     https://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueReleaseGLObjects.html
    //     cl::Event event;

    //     getQueue().enqueueAcquireGLObjects(&shared_objects, NULL, &event);
    //     event.wait();
    //     getQueue().enqueueCopyBuffer(*d_P, *(res[0].get()), 0, 0, bytes,
    //     NULL,
    //                                  &event);
    //     getQueue().enqueueReleaseGLObjects(&shared_objects, NULL, &event);
    //     event.wait();

    //     CL_DEBUG_FINISH(getQueue());
    //     CheckGL("End OpenCL resource copy");
    // } else {
    //     unsigned bytes = 0, buffer = 0;
    //     FG_CHECK(_.fg_get_surface_vertex_buffer(&buffer, surface));
    //     FG_CHECK(_.fg_get_surface_vertex_buffer_size(&bytes, surface));

    //     CheckGL("Begin OpenCL fallback-resource copy");
    //     glBindBuffer(GL_ARRAY_BUFFER, buffer);
    //     auto *ptr =
    //         static_cast<GLubyte *>(glMapBuffer(GL_ARRAY_BUFFER,
    //         GL_WRITE_ONLY));
    //     if (ptr) {
    //         getQueue().enqueueReadBuffer(*P.get(), CL_TRUE, 0, bytes, ptr);
    //         glUnmapBuffer(GL_ARRAY_BUFFER);
    //     }
    //     glBindBuffer(GL_ARRAY_BUFFER, 0);
    //     CheckGL("End OpenCL fallback-resource copy");
    // }
}

#define INSTANTIATE(T) \
    template void copy_surface<T>(const Array<T> &, fg_surface);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}  // namespace oneapi
}  // namespace arrayfire
