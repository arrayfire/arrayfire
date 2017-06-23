/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined (WITH_GRAPHICS)

#include <Array.hpp>
#include <debug_opencl.hpp>
#include <err_opencl.hpp>
#include <GraphicsResourceManager.hpp>
#include <vector_field.hpp>

using af::dim4;

namespace opencl
{
using namespace gl;

template<typename T>
void copy_vector_field(const Array<T> &points, const Array<T> &directions,
                       forge::VectorField* vector_field)
{
    if (isGLSharingSupported()) {
        CheckGL("Begin OpenCL resource copy");
        const cl::Buffer *d_points      = points.get();
        const cl::Buffer *d_directions  = directions.get();
        size_t pBytes = vector_field->verticesSize();
        size_t dBytes = vector_field->directionsSize();

        ShrdResVector res = interopManager().getBufferResource(vector_field);

        std::vector<cl::Memory> shared_objects;
        shared_objects.push_back(*(res[0].get()));
        shared_objects.push_back(*(res[1].get()));

        glFinish();

        // Use of events:
        // https://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueReleaseGLObjects.html
        cl::Event event;

        getQueue().enqueueAcquireGLObjects(&shared_objects, NULL, &event);
        event.wait();
        getQueue().enqueueCopyBuffer(*d_points    , *(res[0].get()), 0, 0, pBytes, NULL, &event);
        getQueue().enqueueCopyBuffer(*d_directions, *(res[1].get()), 0, 0, dBytes, NULL, &event);
        getQueue().enqueueReleaseGLObjects(&shared_objects, NULL, &event);
        event.wait();

        CL_DEBUG_FINISH(getQueue());
        CheckGL("End OpenCL resource copy");
    } else {
        CheckGL("Begin OpenCL fallback-resource copy");
        glBindBuffer(GL_ARRAY_BUFFER, vector_field->vertices());
        GLubyte* pPtr = (GLubyte*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        if (pPtr) {
            getQueue().enqueueReadBuffer(*points.get(), CL_TRUE, 0, vector_field->verticesSize(), pPtr);
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindBuffer(GL_ARRAY_BUFFER, vector_field->directions());
        GLubyte* dPtr = (GLubyte*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        if (dPtr) {
            getQueue().enqueueReadBuffer(*directions.get(), CL_TRUE, 0, vector_field->directionsSize(), dPtr);
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        CheckGL("End OpenCL fallback-resource copy");
    }
}

#define INSTANTIATE(T)                                                                      \
    template void copy_vector_field<T>(const Array<T> &points, const Array<T> &directions,  \
                                       forge::VectorField* vector_field);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}

#endif  // WITH_GRAPHICS
