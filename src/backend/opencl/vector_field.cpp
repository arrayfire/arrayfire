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
#include <vector_field.hpp>

using af::dim4;
using arrayfire::common::ForgeModule;
using arrayfire::common::forgePlugin;

namespace arrayfire {
namespace opencl {

template<typename T>
void copy_vector_field(const Array<T> &points, const Array<T> &directions,
                       fg_vector_field vfield) {
    ForgeModule &_ = common::forgePlugin();
    if (isGLSharingSupported()) {
        CheckGL("Begin OpenCL resource copy");
        const cl::Buffer *d_points     = points.get();
        const cl::Buffer *d_directions = directions.get();
        unsigned pBytes                = 0;
        unsigned dBytes                = 0;
        FG_CHECK(_.fg_get_vector_field_vertex_buffer_size(&pBytes, vfield));
        FG_CHECK(_.fg_get_vector_field_direction_buffer_size(&dBytes, vfield));

        auto res = interopManager().getVectorFieldResources(vfield);

        std::vector<cl::Memory> shared_objects;
        shared_objects.push_back(*(res[0].get()));
        shared_objects.push_back(*(res[1].get()));

        glFinish();

        // Use of events:
        // https://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueReleaseGLObjects.html
        cl::Event event;

        getQueue().enqueueAcquireGLObjects(&shared_objects, NULL, &event);
        event.wait();
        getQueue().enqueueCopyBuffer(*d_points, *(res[0].get()), 0, 0, pBytes,
                                     NULL, &event);
        getQueue().enqueueCopyBuffer(*d_directions, *(res[1].get()), 0, 0,
                                     dBytes, NULL, &event);
        getQueue().enqueueReleaseGLObjects(&shared_objects, NULL, &event);
        event.wait();

        CL_DEBUG_FINISH(getQueue());
        CheckGL("End OpenCL resource copy");
    } else {
        unsigned size1 = 0, size2 = 0;
        unsigned buff1 = 0, buff2 = 0;
        FG_CHECK(_.fg_get_vector_field_vertex_buffer_size(&size1, vfield));
        FG_CHECK(_.fg_get_vector_field_direction_buffer_size(&size2, vfield));
        FG_CHECK(_.fg_get_vector_field_vertex_buffer(&buff1, vfield));
        FG_CHECK(_.fg_get_vector_field_direction_buffer(&buff2, vfield));

        CheckGL("Begin OpenCL fallback-resource copy");

        // Points
        glBindBuffer(GL_ARRAY_BUFFER, buff1);
        auto *pPtr =
            static_cast<GLubyte *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
        if (pPtr) {
            getQueue().enqueueReadBuffer(*points.get(), CL_TRUE, 0, size1,
                                         pPtr);
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Directions
        glBindBuffer(GL_ARRAY_BUFFER, buff2);
        auto *dPtr =
            static_cast<GLubyte *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
        if (dPtr) {
            getQueue().enqueueReadBuffer(*directions.get(), CL_TRUE, 0, size2,
                                         dPtr);
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        CheckGL("End OpenCL fallback-resource copy");
    }
}

#define INSTANTIATE(T)                                                     \
    template void copy_vector_field<T>(const Array<T> &, const Array<T> &, \
                                       fg_vector_field);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}  // namespace opencl
}  // namespace arrayfire
